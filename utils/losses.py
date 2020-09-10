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

import numpy as np
import tensorflow as tf
from tensorpack.tfutils.summary import add_moving_summary

import utils.tb_utils as tbutil
from utils.common_layers import uncompressDepth


def l1_loss(gt: tf.Tensor, pred: tf.Tensor):
    return tf.abs(gt - pred)


def l2_loss(gt: tf.Tensor, pred: tf.Tensor):
    loss = gt - pred
    return loss * loss


def cosine_distance(l, p):
    ps = p
    ls = l

    lNrm = tf.nn.l2_normalize(ls, 1)
    pNrm = tf.nn.l2_normalize(ps, 1)
    return tf.reduce_sum(lNrm * pNrm, axis=-1, keepdims=True)


def angular_distance(l, p, negative_vals=False):
    cosAng = tf.clip_by_value(cosine_distance(l, p), -1.0 + EPS, 1.0 - EPS)
    mult = 2.0 if negative_vals else 1.0
    return (mult * tf.acos(cosAng)) / np.pi


def masked_loss(loss: tf.Tensor, mask: tf.Tensor):
    return tf.where(tf.less_equal(mask, 1e-5), tf.zeros_like(loss), loss)


def height_normal_consistency_loss(
    depth, normal, mask3, consistency_loss_factor: float, imgSize: int, max_outputs: int
):
    near = uncompressDepth(1)
    far = uncompressDepth(0)
    d = uncompressDepth(depth)
    h = tf.div_no_nan(d - near, far - near)

    sobel = tf.image.sobel_edges(h)  # b,h,w,1,[dy,dx] - 1 because height has 1 channel
    dx = sobel[:, :, :, :, 1]  # b,h,w,1
    dy = -sobel[:, :, :, :, 0]
    # We're using a depth map instead of a height. Which means bright
    # values are at a greater depth. Thus, we need to invert the gradient
    texelSize = 1 / imgSize
    dz = tf.ones_like(dx) * texelSize * 2

    cn = tf.nn.l2_normalize(tf.concat([dx, dy, dz], axis=-1), axis=-1)

    cn = cn * 0.5 + 0.5
    cl = masked_loss(l1_loss(cn, normal), mask3)

    clr = tf.reduce_mean(cl, name="consistency_loss") * consistency_loss_factor
    add_moving_summary(clr)
    tf.losses.add_loss(clr, tf.GraphKeys.LOSSES)

    repeat = [1 for _ in range(len(depth.shape))]
    repeat[-1] = 3
    tbutil.four_side_by_side(
        tf.tile(depth, repeat), cn, normal, cl, "consistency", max_outputs
    )
