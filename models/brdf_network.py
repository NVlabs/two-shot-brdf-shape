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
from math import log2
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorpack import (
    ModelDesc,
    ModelSaver,
    QueueInput,
    SaverRestore,
    ScheduledHyperParamSetter,
    SimpleTrainer,
    StagingInput,
    SyncMultiGPUTrainer,
    SyncMultiGPUTrainerParameterServer,
    SyncMultiGPUTrainerReplicated,
    TrainConfig,
    launch_train_with_config,
)
from tensorpack.callbacks import (
    EstimatedTimeLeft,
    GPUUtilizationTracker,
    MinSaver,
    ModelSaver,
)
from tensorpack.tfutils import gradproc, optimizer
from tensorpack.tfutils.argscope import (
    argscope,
    enable_argscope_for_function,
    enable_argscope_for_module,
    get_arg_scope,
)
from tensorpack.tfutils.scope_utils import (
    auto_reuse_variable_scope,
    under_name_scope,
    under_variable_scope,
)
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils import logger
from tensorpack.utils.gpu import change_gpu, get_num_gpu

import dataflow.dataflow as df
import utils.sg_utils as sg
from models.base import (
    BaseNetwork,
    BaseTrainer,
    graySquareImgShape,
    rgbSquareImgShape,
    sphericalGaussainsShape,
)
from utils.common_layers import Fusion2DBlock, INReLU, apply_mask, normalize
from utils.config import ParameterNames
from utils.losses import *
from utils.rendering_layer import RenderingLayer
from utils.tb_utils import two_side_by_side
from config import get_render_config

enable_argscope_for_module(tf.layers)


class BrdfNetwork(BaseNetwork):
    def __init__(
        self,
        base_nf: int = 32,
        imgSize: int = 256,
        fov: int = 60,
        distance_to_zero: float = 0.7,
        camera_pos=np.asarray([0, 0, 0]),
        light_pos=np.asarray([0, 0, 0]),
        light_color=np.asarray([1, 1, 1]),
        light_intensity_lumen=45,
        num_sgs=24,
        no_rendering_loss: bool = False,
    ):
        self.base_nf = base_nf
        self.imgSize = imgSize
        self.fov = fov
        self.distance_to_zero = distance_to_zero

        self.camera_pos = tf.convert_to_tensor(
            camera_pos.reshape([1, 3]), dtype=tf.float32
        )
        self.light_pos = tf.convert_to_tensor(
            light_pos.reshape([1, 3]), dtype=tf.float32
        )
        intensity = light_intensity_lumen / (4.0 * np.pi)
        light_col = light_color * intensity
        self.light_col = tf.convert_to_tensor(
            light_col.reshape([1, 3]), dtype=tf.float32
        )

        self.num_sgs = num_sgs
        self.no_rendering_loss = no_rendering_loss

        self.axis_sharpness = tf.convert_to_tensor(
            sg.setup_axis_sharpness(self.num_sgs), dtype=tf.float32
        )

    def inputs(self):
        """
        proc_global,  # 0
        cam2,  # 1
        ensureSingleChannel(mask),  # 2
        normal,  # 3
        ensureSingleChannel(depth),  # 4
        """
        return [
            rgbSquareImgShape(self.imgSize, "cam1"),  # Cam 1
            rgbSquareImgShape(self.imgSize, "cam2"),  # Cam 2
            graySquareImgShape(self.imgSize, "mask"),  # Mask
            rgbSquareImgShape(self.imgSize, "normal"),  # Previous Normal Estimation
            graySquareImgShape(self.imgSize, "depth"),  # Previous Depth Estimation
            sphericalGaussainsShape(self.num_sgs, "sgs"),  # Previous SGs Estimation
            rgbSquareImgShape(self.imgSize, "diffuse_gt"),  # Diffuse
            rgbSquareImgShape(self.imgSize, "specular_gt"),  # Specular
            graySquareImgShape(self.imgSize, "roughness_gt"),  # Roughness
        ]

    def outputs(self) -> List[str]:
        return [
            "brdf_predictions/diffuse",
            "brdf_predictions/specular",
            "brdf_predictions/roughness",
        ]

    def output_image_names(self) -> List[str]:
        return [
            p.value
            for p in [
                ParameterNames.DIFFUSE_PRED,
                ParameterNames.SPECULAR_PRED,
                ParameterNames.ROUGHNESS_PRED,
            ]
        ]

    def network_architecture(self, *args):
        cam1, cam2, mask, normal, depth, sgs = args
        layers_needed = int(log2(self.imgSize) - 2)  # Do not go down to 1x1 but 4x4

        with argscope([tf.layers.conv2d, tf.layers.conv2d_transpose], padding="same"):
            with tf.variable_scope("brdf_net"):
                with tf.variable_scope("prepare"):
                    onesTensor = tf.ones_like(mask[:, :, :, 0:1])
                    sgs_expanded = tf.reshape(
                        sgs, [-1, 1, 1, sgs.shape[1] * sgs.shape[2]]
                    )
                    sgs_to_add = onesTensor * sgs_expanded

                    brdfInput = tf.concat(
                        [cam1, cam2, normal, depth, sgs_to_add, mask[:, :, :, 0:1]],
                        axis=-1,
                        name="input_stack",
                    )

                with tf.variable_scope("enc"):
                    l = brdfInput
                    skips = []
                    for i in range(layers_needed):
                        skips.append(l)
                        l = tf.layers.conv2d(
                            l,
                            min(self.base_nf * (2 ** i), 512),
                            4,
                            strides=2,
                            activation=INReLU,
                            name="conv%d" % (i + 1),
                        )

                    encoded = tf.identity(l, "encoded")

                ####==============####
                ####Start Decoding####
                ####==============####

                with tf.variable_scope("dec"):
                    for i in range(layers_needed):
                        with tf.variable_scope("up%d" % (i + 1)):
                            inv_i = layers_needed - i
                            nf = min(self.base_nf * (2 ** (inv_i - 1)), 512)

                            l = tf.layers.conv2d_transpose(
                                l,
                                nf,
                                4,
                                strides=2,
                                activation=INReLU,
                                name="tconv%d" % (i + 1),
                            )
                            l = tf.concat(
                                [l, skips[inv_i - 1]], -1, name="skip%d" % (i + 1)
                            )
                            l = tf.layers.conv2d(
                                l, nf, 3, activation=INReLU, name="conv%d" % (i + 1)
                            )

                    brdf_params = tf.layers.conv2d(
                        l, 7, 5, activation=tf.nn.sigmoid, name="output"
                    )

            with tf.variable_scope("brdf_predictions"):
                diffuse = apply_mask(
                    tf.clip_by_value(brdf_params[:, :, :, 0:3], 0.0, 1.0),
                    mask,
                    "diffuse",
                )

                specular = apply_mask(
                    tf.clip_by_value(brdf_params[:, :, :, 3:6], 40 / 255, 1.0),
                    mask,
                    "specular",
                )

                roughness = apply_mask(
                    tf.clip_by_value(brdf_params[:, :, :, 6:7], 0.004, 1.0),
                    mask[:, :, :, 0:1],
                    "roughness",
                )

        return (diffuse, specular, roughness)

    def render(self, diffuse, specular, roughness, normal, depth, sgs, mask3):
        with tf.variable_scope("prediction_post_process"):
            sdiff = apply_mask(diffuse, mask3, "safe_diffuse", undefined=0.5)
            sspec = apply_mask(specular, mask3, "safe_specular", undefined=0.04)
            srogh = apply_mask(
                roughness, mask3[:, :, :, 0:1], "safe_roughness", undefined=0.4
            )
            snormal = tf.where(
                tf.less_equal(mask3, 1e-5),
                tf.ones_like(normal) * tf.convert_to_tensor([0.5, 0.5, 1.0]),
                normal,
                name="safe_normal",
            )

        with tf.variable_scope("env_map", reuse=True):
            with tf.variable_scope("sgs_prep"):
                batch_size = tf.shape(diffuse)[0]
                axis_sharpness = tf.tile(
                    tf.expand_dims(self.axis_sharpness, 0), tf.stack([batch_size, 1, 1])
                )  # add batch dim

                sgs_joined = tf.concat([sgs, axis_sharpness], -1, name="sgs")
                print(
                    "Joined pred shape",
                    sgs_joined.shape,
                    sgs.shape,
                    axis_sharpness.shape,
                )

        with tf.variable_scope("rendering"):
            renderer = RenderingLayer(
                self.fov,
                self.distance_to_zero,
                tf.TensorShape([None, self.imgSize, self.imgSize, 3]),
            )
            rerendered = renderer.call(
                sdiff,
                sspec,
                srogh,
                snormal,
                depth,  # Depth is still in 0 - 1 range
                mask3[:, :, :, 0:1],
                self.camera_pos,
                self.light_pos,
                self.light_col,
                sgs_joined,
            )
            rerendered = apply_mask(rerendered, mask3, "masked_rerender")
            rerendered = tf.check_numerics(rerendered, "Rendering produces NaN!")
            rerendered = tf.identity(rerendered, "rendered")

        return rerendered

    def build_graph(
        self,
        cam1: tf.Tensor,
        cam2: tf.Tensor,
        mask: tf.Tensor,
        normal: tf.Tensor,
        depth: tf.Tensor,
        sgs: tf.Tensor,
        diffuse_gt: tf.Tensor,
        specular_gt: tf.Tensor,
        roughness_gt: tf.Tensor,
    ):
        with tf.variable_scope("prepare"):
            mask = mask[:, :, :, 0:1]
            repeat = [1 for _ in range(len(mask.shape))]
            repeat[-1] = 3

            mask3 = tf.tile(mask, repeat)

            two_side_by_side(cam1, cam2, "input", 10)

            in1 = cam1
            in2 = cam2
            m3 = mask3

            two_side_by_side(normal, tf.tile(depth, repeat), "geom", 10)

            batch_size = tf.shape(cam1)[0]

        diffuse, specular, roughness = self.network_architecture(
            in1, in2, m3, normal, depth, sgs
        )

        rendered = self.render(diffuse, specular, roughness, normal, depth, sgs, mask3)

        ## Rerender
        with tf.variable_scope("viz"):
            rendered_reinhard = rendered / (1.0 + rendered)
            loss_img_reinhard = cam1 / (1.0 + cam1)
            two_side_by_side(
                tf.clip_by_value(tf.pow(loss_img_reinhard, 1.0 / 2.2), 0.0, 1.0),
                tf.clip_by_value(tf.pow(rendered_reinhard, 1.0 / 2.2), 0.0, 1.0),
                "rendered",
                10,
            )

        with tf.variable_scope("loss"):
            with tf.variable_scope("rendering"):
                rerendered_log = tf.clip_by_value(
                    tf.log(1.0 + tf.nn.relu(rendered)), 0.0, 13.0
                )
                rerendered_log = tf.check_numerics(
                    rerendered_log, "Rerendered log image contains NaN or Inf"
                )
                loss_log = tf.clip_by_value(tf.log(1.0 + tf.nn.relu(cam1)), 0.0, 13.0)
                loss_log = tf.check_numerics(
                    loss_log, "The Loss log image contains NaN or Inf"
                )

                l1_err = l1_loss(loss_log, rerendered_log)
                rerendered_loss = tf.reduce_mean(
                    masked_loss(l1_err, mask3), name="rendering_loss"
                )
                add_moving_summary(rerendered_loss)
                tf.losses.add_loss(rerendered_loss, tf.GraphKeys.LOSSES)

            with tf.variable_scope("diffuse"):
                diffuse_loss = tf.reduce_mean(
                    masked_loss(l1_loss(diffuse_gt, diffuse), mask3),
                    name="diffuse_loss",
                )
                add_moving_summary(diffuse_loss)
                tf.losses.add_loss(diffuse_loss, tf.GraphKeys.LOSSES)
                two_side_by_side(diffuse_gt, diffuse, "diffuse", 10)

            with tf.variable_scope("specular"):
                specular_loss = tf.reduce_mean(
                    masked_loss(l1_loss(specular_gt, specular), mask3),
                    name="specular_loss",
                )
                add_moving_summary(specular_loss)
                tf.losses.add_loss(specular_loss, tf.GraphKeys.LOSSES)
                two_side_by_side(specular_gt, specular, "specular", 10)

            with tf.variable_scope("roughness"):
                roughness_loss = tf.reduce_mean(
                    masked_loss(l1_loss(roughness_gt, roughness), mask),
                    name="roughness_loss",
                )
                add_moving_summary(roughness_loss)
                tf.losses.add_loss(roughness_loss, tf.GraphKeys.LOSSES)
                two_side_by_side(roughness_gt, roughness, "roughness", 10)

        self.cost = tf.losses.get_total_loss(name="total_costs")

        add_moving_summary(self.cost)
        if self.training:
            add_param_summary((".*/W", ["histogram"]))  # monitor W

        return self.cost

    def optimizer(self):
        lr = tf.get_variable("learning_rate", initializer=0.0002, trainable=False)
        tf.summary.scalar("learning_rate", lr)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
        return optimizer.apply_grad_processors(
            opt, [gradproc.SummaryGradient(), gradproc.CheckGradient()]
        )
        # return opt


class InferenceModel(BrdfNetwork):
    def inputs(self):
        return [
            rgbSquareImgShape(self.imgSize, "input1"),  # Cam1
            rgbSquareImgShape(self.imgSize, "input2"),  # Cam2
            graySquareImgShape(self.imgSize, "mask"),  # Mask
            rgbSquareImgShape(self.imgSize, "normal"),  # Normal
            graySquareImgShape(self.imgSize, "depth"),  # Depth
            sphericalGaussainsShape(self.num_sgs, "sgs"),  # Previous SGs Estimation
        ]

    def build_graph(self, input1, input2, mask, normal, depth, sgs):
        with tf.variable_scope("prepare"):
            repeat = [1 for _ in range(len(mask.shape))]
            repeat[-1] = 3

            mask = tf.where(tf.less(mask, 0.9), tf.zeros_like(mask), tf.ones_like(mask))

            mask3 = tf.tile(mask, repeat)

        diffuse, specular, roughness = self.network_architecture(
            input1, input2, mask3, normal, depth, sgs
        )


class BrdfTrainer(BaseTrainer):
    def __init__(self, subparser):
        self.render_config = get_render_config()
        super().__init__(df.Dataflows.BRDF, "brdf", subparser)

    def _network_specific_callbacks(self):
        self.callbacks.append(
            ScheduledHyperParamSetter(
                "learning_rate", [(1, 0.0002), (int(self.args.epochs / 2), 0.0001)]
            )
        )

    def _network_specific_args(self, parser):
        parser.add_argument(
            "--base_nf",
            type=int,
            default=32,
            help="Number of base features to use. Number of features for each layers are based on this.",
        )
        parser.add_argument(
            "--num_sgs",
            type=int,
            default=24,
            help="The numbers of Spherical Gaussians (SG) amplitudes to use for re-rendering. Needs to match the number of previously predicted SGs.",
        )
        parser.add_argument(
            "--no_rendering_loss",
            action="store_true",
            help="Disables the re-rendering loss.",
        )

    def train_model(self, args) -> BaseNetwork:
        return BrdfNetwork(
            base_nf=args.base_nf,
            imgSize=args.image_size,
            fov=self.render_config["field_of_view"],
            distance_to_zero=self.render_config["distance_to_zero"],
            camera_pos=np.asarray(self.render_config["camera_position"]),
            light_pos=np.asarray(self.render_config["light_position"]),
            light_color=np.asarray(self.render_config["light_color"]),
            light_intensity_lumen=self.render_config["light_intensity_lumen"],
            num_sgs=args.num_sgs,
            no_rendering_loss=args.no_rendering_loss,
        )

    def inference_model(self, args) -> BaseNetwork:
        return InferenceModel(
            base_nf=args.base_nf,
            imgSize=args.image_size,
            fov=self.render_config["field_of_view"],
            distance_to_zero=self.render_config["distance_to_zero"],
            camera_pos=np.asarray(self.render_config["camera_position"]),
            light_pos=np.asarray(self.render_config["light_position"]),
            light_color=np.asarray(self.render_config["light_color"]),
            light_intensity_lumen=self.render_config["light_intensity_lumen"],
            num_sgs=args.num_sgs,
            no_rendering_loss=args.no_rendering_loss,
        )
