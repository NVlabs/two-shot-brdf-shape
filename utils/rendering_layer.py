# -----------------------------------------------------------------------
# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
#
# Official Implementation of the CVPR2020 Paper
# Two-shot Spatially-varying BRDF and Shape Estimation
# Mark Boss, Varun Jampani, Kihwan Kim, Hendrik P. A. Lensch, Jan Kautz
# -----------------------------------------------------------------------

import os
import sys
from typing import Any, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape

import utils.common_layers as cl
import utils.layer_helper as layer_helper
from utils.common_layers import (
    gamma_to_linear,
    isclose,
    linear_to_gamma,
    mix,
    safe_sqrt,
    saturate,
    srgb_to_linear,
)
from utils.dataflow_utils import apply_mask, chwToHwc, ensureSingleChannel, hwcToChw

EPS = 1e-7


class RenderingLayer:
    def __init__(
        self,
        fov: int,
        distanceToZero: float,
        output_shape: tensor_shape.TensorShape,
        data_format: str = "channels_last",
    ):
        self.distanceToZero = distanceToZero
        self.fov = fov
        self.data_format = layer_helper.normalize_data_format(data_format)

        self.build(output_shape)

    def build(self, output_shape):
        with tf.variable_scope("Prep"):
            if self.data_format == "channels_first":
                channel_axis = 1
                height_axis = 2
                width_axis = 3
            else:
                channel_axis = -1
                height_axis = 1
                width_axis = 2

            if (
                output_shape.dims[height_axis].value is None
                or output_shape.dims[width_axis].value is None
            ):
                raise ValueError(
                    "The width and height dimension of the inputs "
                    "should be defined. Found `None`."
                )

            height = int(output_shape[height_axis])
            width = int(output_shape[width_axis])

            yRange = xRange = self.distanceToZero * np.tan((self.fov * np.pi / 180) / 2)

            x, y = np.meshgrid(
                np.linspace(-xRange, xRange, height),
                np.linspace(-yRange, yRange, width),
            )
            y = np.flipud(y)
            x = np.fliplr(x)

            z = np.ones((height, width), dtype=np.float32)
            coord = np.stack([x, y, z]).astype(np.float32)
            if self.data_format == "channels_last":
                coord = chwToHwc(coord)

            self.base_mesh = tf.convert_to_tensor(
                np.expand_dims(coord, 0), dtype=tf.float32
            )

    def call(
        self,
        diffuse: tf.Tensor,
        specular: tf.Tensor,
        roughness: tf.Tensor,
        normal: tf.Tensor,
        depth: tf.Tensor,
        mask: tf.Tensor,
        camera_pos: tf.Tensor,
        light_pos: tf.Tensor,
        light_color: tf.Tensor,
        sgs: tf.Tensor,
    ) -> tf.Tensor:
        """ Evaluate the rendering equation
        """
        with tf.variable_scope("Setup"):
            assert (
                sgs.shape[self._get_channel_axis()] == 7 and len(sgs.shape) == 3
            )  # n, sgs, c
            assert (
                diffuse.shape[self._get_channel_axis()] == 3 and len(diffuse.shape) == 4
            )
            assert (
                specular.shape[self._get_channel_axis()] == 3
                and len(specular.shape) == 4
            )
            assert (
                roughness.shape[self._get_channel_axis()] == 1
                and len(roughness.shape) == 4
            )
            assert (
                normal.shape[self._get_channel_axis()] == 3 and len(normal.shape) == 4
            )
            assert mask.shape[self._get_channel_axis()] == 1 and len(mask.shape) == 4
            assert (
                camera_pos.shape[self._get_channel_axis()] == 3
                and len(camera_pos.shape) == 2
            )
            assert (
                light_pos.shape[self._get_channel_axis()] == 3
                and len(light_pos.shape) == 2
            )
            assert (
                light_color.shape[self._get_channel_axis()] == 3
                and len(light_color.shape) == 2
            )

            realDepth = self._uncompressDepth(depth)
            perturbed_mesh = self.base_mesh * realDepth

            if self.data_format == "channels_first":
                reshapeShape = [-1, 3, 1, 1]
            else:
                reshapeShape = [-1, 1, 1, 3]

            lp = tf.reshape(light_pos, reshapeShape)
            vp = tf.reshape(camera_pos, reshapeShape)
            lc = tf.reshape(light_color, reshapeShape)

            l_vec = lp - perturbed_mesh

            v = self._normalize(vp - perturbed_mesh)
            l = self._normalize(l_vec)
            h = self._normalize(l + v)

            axis_flip = tf.constant([-1, 1, -1], dtype=tf.float32)
            axis_flip = tf.reshape(axis_flip, reshapeShape)
            n = self._normalize(normal * 2.0 - 1.0) * axis_flip

            ndl = saturate(self._dot(n, l))
            ndv = saturate(self._dot(n, v), 1e-5)
            ndh = saturate(self._dot(n, h))
            ldh = saturate(self._dot(l, h))
            vdh = saturate(self._dot(v, h))

            sqrLightDistance = self._dot(l_vec, l_vec)
            light = tf.div_no_nan(lc, sqrLightDistance)

            diff = srgb_to_linear(diffuse)
            spec = srgb_to_linear(specular)

        directSpecular = self.spec(ndl, ndv, ndh, ldh, vdh, spec, roughness)
        with tf.variable_scope("Diffuse"):
            directDiffuse = diff * (1.0 / np.pi) * ndv * (1.0 - self.F(spec, ldh))

        with tf.variable_scope("Direct_light"):
            brdf = directSpecular + directDiffuse
            direct = brdf * light

            direct = tf.where(
                tf.math.less(self._to_vec3(ndl), EPS), tf.zeros_like(direct), direct
            )
            direct = tf.where(
                tf.math.less(self._to_vec3(ndv), EPS), tf.zeros_like(direct), direct
            )

        with tf.variable_scope("SG_light"):
            sg_stack_axis = 2 if self.data_format == "channels_first" else 1
            number_of_sgs = sgs.shape[sg_stack_axis]

            env_direct = tf.zeros_like(direct)
            for i in range(number_of_sgs):
                if self.data_format == "channels_first":
                    sg = sgs[:, :, i]
                else:
                    sg = sgs[:, i]

                evaled = self.sg_eval(
                    sg, diffuse, specular, roughness, n, mask, perturbed_mesh, vp
                )

                env_direct = env_direct + evaled

        with tf.variable_scope("Blending"):
            return direct + env_direct

    def F(self, F0: tf.Tensor, ldh: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("Fresnel"):
            ct = 1 - ldh
            ctsq = ct * ct
            ct5 = ctsq * ctsq * ct
            return F0 + (1 - F0) * ct5

    def _G(self, a2: tf.Tensor, ndx: tf.Tensor) -> tf.Tensor:
        return tf.div_no_nan(2 * ndx, ndx + safe_sqrt(a2 + (1 - a2) * ndx * ndx))

    def G(self, alpha: tf.Tensor, ndl: tf.Tensor, ndv: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("Geometry"):
            a2 = alpha * alpha
            return self._G(a2, ndl) * self._G(a2, ndv)

    def D(self, alpha: tf.Tensor, ndh: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("Distribution"):
            a2 = alpha * alpha

            denom = (ndh * ndh) * (a2 - 1) + 1.0
            denom2 = denom * denom

            return tf.div_no_nan(a2, np.pi * denom2)

    def spec(
        self,
        ndl: tf.Tensor,
        ndv: tf.Tensor,
        ndh: tf.Tensor,
        ldh: tf.Tensor,
        vdh: tf.Tensor,
        F0: tf.Tensor,
        roughness: tf.Tensor,
    ) -> tf.Tensor:
        with tf.variable_scope("Specular"):
            alpha = saturate(roughness * roughness, 1e-3)

            F = self.F(F0, ldh)
            G = self.G(alpha, ndl, ndv)
            D = self.D(alpha, ndh)

            ret = tf.div_no_nan(F * G * D, 4.0 * ndl)

            ret = tf.where(
                tf.math.less(self._to_vec3(ndh), EPS), tf.zeros_like(ret), ret
            )
            ret = tf.where(
                tf.math.less(self._to_vec3(ldh * ndl), EPS), tf.zeros_like(ret), ret
            )
            ret = tf.where(
                tf.math.less(self._to_vec3(vdh * ndv), EPS), tf.zeros_like(ret), ret
            )
            return ret

    def _sg_integral(self, sg: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("Integral"):
            assert sg.shape[self._get_channel_axis()] == 7 and len(sg.shape) == 4

            s_amplitude, s_axis, s_sharpness = self._extract_sg_components(sg)

            expTerm = 1.0 - tf.exp(-2.0 * s_sharpness)
            return 2 * np.pi * tf.div_no_nan(s_amplitude, s_sharpness) * expTerm

    def _sg_evaluate(self, sg: tf.Tensor, d: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("Evaluate"):
            assert sg.shape[self._get_channel_axis()] == 7 and len(sg.shape) == 4
            assert d.shape[self._get_channel_axis()] == 3 and len(d.shape) == 4

            s_amplitude, s_axis, s_sharpness = self._extract_sg_components(sg)

            cosAngle = self._dot(d, s_axis)
            return s_amplitude * tf.exp(s_sharpness * (cosAngle - 1.0))

    def _sg_inner_product(self, sg1: tf.Tensor, sg2: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("InnerProd"):
            assert sg1.shape[self._get_channel_axis()] == 7 and len(sg1.shape) == 4
            assert sg2.shape[self._get_channel_axis()] == 7 and len(sg2.shape) == 4

            s1_amplitude, s1_axis, s1_sharpness = self._extract_sg_components(sg1)
            s2_amplitude, s2_axis, s2_sharpness = self._extract_sg_components(sg2)

            umLength = self._magnitude(s1_sharpness * s1_axis + s2_sharpness * s2_axis)
            expo = (
                tf.exp(umLength - s1_sharpness - s2_sharpness)
                * s1_amplitude
                * s2_amplitude
            )

            other = 1.0 - tf.exp(-2.0 * umLength)

            return tf.div_no_nan(2.0 * np.pi * expo * other, umLength)

    def _sg_evaluate_diffuse(
        self, sg: tf.Tensor, diffuse: tf.Tensor, normal: tf.Tensor
    ) -> tf.Tensor:
        with tf.variable_scope("Diffuse"):
            assert (
                sg.shape[self._get_channel_axis()] == 7 and len(sg.shape) == 4
            )  # b, h, w, c | b, c, h, w
            assert (
                diffuse.shape[self._get_channel_axis()] == 3 and len(diffuse.shape) == 4
            )
            assert (
                normal.shape[self._get_channel_axis()] == 3 and len(normal.shape) == 4
            )

            diff = tf.div_no_nan(diffuse, np.pi)

            s_amplitude, s_axis, s_sharpness = self._extract_sg_components(sg)

            mudn = saturate(self._dot(s_axis, normal))

            c0 = 0.36
            c1 = 1.0 / (4.0 * c0)

            eml = tf.exp(-s_sharpness)
            em2l = eml * eml
            rl = tf.div_no_nan(1.0, s_sharpness)

            scale = 1.0 + 2.0 * em2l - rl
            bias = (eml - em2l) * rl - em2l

            x = safe_sqrt(1.0 - scale)
            x0 = c0 * mudn
            x1 = c1 * x

            n = x0 + x1

            y = tf.where(tf.less_equal(tf.abs(x0), x1), n * tf.div_no_nan(n, x), mudn)

            res = scale * y + bias

            res = res * self._sg_integral(sg) * diff

            return res

    def _sg_distribution_term(self, d: tf.Tensor, roughness: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("Distribution"):
            assert d.shape[self._get_channel_axis()] == 3 and len(d.shape) == 4
            assert (
                roughness.shape[self._get_channel_axis()] == 1
                and len(roughness.shape) == 4
            )

            a2 = saturate(roughness * roughness, 1e-3)

            ret = tf.concat(
                [
                    self._to_vec3(tf.div_no_nan(1.0, np.pi * a2)),
                    d,
                    tf.div_no_nan(2.0, a2),
                ],
                self._get_channel_axis(),
            )

            return ret

    def _sg_warp_distribution(self, ndfs: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("WarpDistribution"):
            assert ndfs.shape[self._get_channel_axis()] == 7 and len(ndfs.shape) == 4
            assert v.shape[self._get_channel_axis()] == 3 and len(v.shape) == 4

            ndf_amplitude, ndf_axis, ndf_sharpness = self._extract_sg_components(ndfs)

            ret = tf.concat(
                [
                    ndf_amplitude,
                    self._reflect(-v, ndf_axis),
                    tf.div_no_nan(
                        ndf_sharpness, (4.0 * saturate(self._dot(ndf_axis, v), 1e-4))
                    ),
                ],
                self._get_channel_axis(),
            )

            return ret

    def _sg_ggx(self, a2: tf.Tensor, ndx: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("Geometric"):
            return tf.div_no_nan(1.0, (ndx + safe_sqrt(a2 + (1 - a2) * ndx * ndx)))

    def _sg_evaluate_specular(
        self,
        sg: tf.Tensor,
        specular: tf.Tensor,
        roughness: tf.Tensor,
        warped_ndf: tf.Tensor,
        ndl: tf.Tensor,
        ndv: tf.Tensor,
        ldh: tf.Tensor,
        vdh: tf.Tensor,
        ndh: tf.Tensor,
    ) -> tf.Tensor:
        with tf.variable_scope("Specular"):
            assert sg.shape[self._get_channel_axis()] == 7 and len(sg.shape) == 4
            assert (
                warped_ndf.shape[self._get_channel_axis()] == 7
                and len(warped_ndf.shape) == 4
            )
            assert (
                specular.shape[self._get_channel_axis()] == 3
                and len(specular.shape) == 4
            )
            assert (
                roughness.shape[self._get_channel_axis()] == 1
                and len(roughness.shape) == 4
            )
            assert ndl.shape[self._get_channel_axis()] == 1 and len(ndl.shape) == 4
            assert ndv.shape[self._get_channel_axis()] == 1 and len(ndv.shape) == 4
            assert ldh.shape[self._get_channel_axis()] == 1 and len(ldh.shape) == 4
            assert vdh.shape[self._get_channel_axis()] == 1 and len(vdh.shape) == 4
            assert ndh.shape[self._get_channel_axis()] == 1 and len(ndh.shape) == 4

            a2 = saturate(roughness * roughness, 1e-3)

            with tf.variable_scope("Distribution"):
                D = self._sg_inner_product(warped_ndf, sg)

            G = self._sg_ggx(a2, ndl) * self._sg_ggx(a2, ndv)

            with tf.variable_scope("Fresnel"):
                powTerm = tf.pow(1.0 - ldh, 5)
                F = specular + (1.0 - specular) * powTerm

            output = D * G * F * ndl

            shadowed = tf.zeros_like(output)
            zero_vec = tf.zeros_like(ndh)
            output = tf.where(self._to_vec3(isclose(ndh, zero_vec)), shadowed, output)
            output = tf.where(
                self._to_vec3(isclose(ldh * ndl, zero_vec)), shadowed, output
            )
            output = tf.where(
                self._to_vec3(isclose(vdh * ndv, zero_vec)), shadowed, output
            )

            return tf.maximum(output, 0.0)

    def sg_eval(
        self,
        sg: tf.Tensor,
        diffuse: tf.Tensor,
        specular: tf.Tensor,
        roughness: tf.Tensor,
        normal: tf.Tensor,
        mask: tf.Tensor,
        perturbed_mesh: tf.Tensor,
        camera_pos: tf.Tensor,
    ) -> tf.Tensor:
        with tf.variable_scope("SG"):
            assert sg.shape[self._get_channel_axis()] == 7 and len(sg.shape) == 2
            assert (
                diffuse.shape[self._get_channel_axis()] == 3 and len(diffuse.shape) == 4
            )
            assert (
                specular.shape[self._get_channel_axis()] == 3
                and len(specular.shape) == 4
            )
            assert (
                roughness.shape[self._get_channel_axis()] == 1
                and len(roughness.shape) == 4
            )
            assert (
                normal.shape[self._get_channel_axis()] == 3 and len(normal.shape) == 4
            )
            assert (
                normal.shape[self._get_channel_axis()] == 3 and len(normal.shape) == 4
            )
            assert mask.shape[self._get_channel_axis()] == 1 and len(mask.shape) == 4
            assert (
                camera_pos.shape[self._get_channel_axis()] == 3
                and len(camera_pos.shape) == 4
            )

            if self.data_format == "channels_first":
                sgShape = [-1, 7, 1, 1]
            else:
                sgShape = [-1, 1, 1, 7]

            sg = tf.reshape(sg, sgShape)

            v = self._normalize(camera_pos - perturbed_mesh)
            diff = srgb_to_linear(diffuse)
            spec = srgb_to_linear(specular)
            norm = normal
            rogh = roughness

            ndf = self._sg_distribution_term(norm, rogh)

            warped_ndf = self._sg_warp_distribution(ndf, v)
            _, wndf_axis, _ = self._extract_sg_components(warped_ndf)

            warpDir = wndf_axis

            ndl = saturate(self._dot(norm, warpDir))
            ndv = saturate(self._dot(norm, v), 1e-5)
            h = self._normalize(warpDir + v)
            ndh = saturate(self._dot(norm, h))
            ldh = saturate(self._dot(warpDir, h))
            vdh = saturate(self._dot(v, h))

            diffuse_eval = self._sg_evaluate_diffuse(sg, diff, norm) * ndl
            specular_eval = self._sg_evaluate_specular(
                sg, spec, rogh, warped_ndf, ndl, ndv, ldh, vdh, ndh
            )

            shadowed = tf.zeros_like(diffuse_eval)
            zero_vec = tf.zeros_like(ndl)
            diffuse_eval = tf.where(
                self._to_vec3(isclose(ndl, zero_vec)), shadowed, diffuse_eval
            )
            diffuse_eval = tf.where(
                self._to_vec3(isclose(ndv, zero_vec)), shadowed, diffuse_eval
            )

            specular_eval = tf.where(
                self._to_vec3(isclose(ndl, zero_vec)), shadowed, specular_eval
            )
            specular_eval = tf.where(
                self._to_vec3(isclose(ndv, zero_vec)), shadowed, specular_eval
            )

            brdf_eval = diffuse_eval + specular_eval

            return tf.where(
                self._to_vec3(tf.equal(mask, zero_vec)), shadowed, brdf_eval
            )

    def _extract_sg_components(
        self, sg: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        if self.data_format == "channels_first":
            s_amplitude = sg[:, 0:3]
            s_axis = sg[:, 3:6]
            s_sharpness = sg[:, 6:7]
        else:
            s_amplitude = sg[..., 0:3]
            s_axis = sg[..., 3:6]
            s_sharpness = sg[..., 6:7]

        return (s_amplitude, s_axis, s_sharpness)

    def visualize_sgs(self, sgs: tf.Tensor, output: tf.Tensor, name: str = "sgs"):
        with tf.variable_scope("Visualize"):
            us, vs = tf.meshgrid(
                tf.linspace(0.0, 1.0, output.shape[2]),
                tf.linspace(0.0, 1.0, output.shape[1]),
            )  # OK

            uvs = tf.stack([us, vs], -1)
            # q   f

            theta = 2.0 * np.pi * uvs[..., 0] - (np.pi / 2)
            phi = np.pi * uvs[..., 1]

            d = tf.expand_dims(
                tf.stack(
                    [
                        tf.cos(theta) * tf.sin(phi),
                        tf.cos(phi),
                        tf.sin(theta) * tf.sin(phi),
                    ],
                    -1,
                ),
                0,
            )

            for i in range(sgs.shape[1]):
                output = output + self._sg_evaluate(
                    tf.reshape(sgs[:, i], [-1, 1, 1, 7]), d
                )

            tf.summary.image(name, output, max_outputs=10)

    def _magnitude(self, x: tf.Tensor) -> tf.Tensor:
        return cl.magnitude(x, data_format=self.data_format)

    def _normalize(self, x: tf.Tensor) -> tf.Tensor:
        return cl.normalize(x, data_format=self.data_format)

    def _dot(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return cl.dot(x, y, data_format=self.data_format)

    def _reflect(self, d: tf.Tensor, n: tf.Tensor) -> tf.Tensor:
        return d - 2 * self._dot(d, n) * n

    def _to_vec3(self, x: tf.Tensor) -> tf.Tensor:
        return cl.to_vec3(x, data_format=self.data_format)

    def _get_channel_axis(self) -> int:
        return cl.get_channel_axis(data_format=self.data_format)

    def _uncompressDepth(
        self, d: tf.Tensor, sigma: float = 2.5, epsilon: float = 0.7
    ) -> tf.Tensor:
        """From 0-1 values to full depth range. The possible depth range
        is modelled by sigma and epsilon and with sigma=2.5 and epsilon=0.7
        it is between 0.17 and 1.4.
        """
        return tf.div_no_nan(1.0, 2.0 * sigma * d + epsilon)
