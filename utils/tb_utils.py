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

import tensorflow as tf


def two_side_by_side(img1, img2, scope, max_outputs):
    viz = tf.concat([img1, img2], 2)
    tf.summary.image(scope, viz, max_outputs=max_outputs)


def four_side_by_side(img1, img2, img3, img4, scope, max_outputs):
    row1 = tf.concat([img1, img2], 2)
    row2 = tf.concat([img3, img4], 2)
    grid = tf.concat([row1, row2], 1)
    tf.summary.image(scope, grid, max_outputs=max_outputs)
