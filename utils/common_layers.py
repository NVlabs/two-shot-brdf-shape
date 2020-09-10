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

from typing import Callable, Optional, Tuple, Union

import tensorflow as tf
from tensorpack.models import (
    BatchNorm,
    BNReLU,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    FullyConnected,
    InstanceNorm,
    MaxPooling,
    layer_register,
)
from tensorpack.tfutils.argscope import argscope, get_arg_scope

EPS = 1e-7


def apply_mask(
    img: tf.Tensor, mask: tf.Tensor, name: str, undefined: float = 0
) -> tf.Tensor:
    return tf.where(
        tf.less_equal(mask, 1e-5), tf.ones_like(img) * undefined, img, name=name
    )


def uncompressDepth(
    d: tf.Tensor, sigma: float = 2.5, epsilon: float = 0.7
) -> tf.Tensor:
    """From 0-1 values to full depth range. The possible depth range
        is modelled by sigma and epsilon and with sigma=2.5 and epsilon=0.7
        it is between 0.17 and 1.4.
        """
    return tf.div_no_nan(1.0, 2.0 * sigma * d + epsilon)


def saturate(x, l=0.0, h=1.0):
    return tf.clip_by_value(x, l, h)


def mix(x, y, a):
    return x * (1 - a) + y * a


def srgb_to_linear(x: tf.Tensor) -> tf.Tensor:
    return tf.where(
        tf.math.greater_equal(x, 0.04045),
        tf.pow(tf.div_no_nan(x + 0.055, 1.055), 2.4),
        tf.div_no_nan(x, 12.92),
    )


def linear_to_gamma(x: tf.Tensor, gamma: float = 2.2) -> tf.Tensor:
    return tf.pow(x, 1.0 / gamma)


def gamma_to_linear(x: tf.Tensor, gamma: float = 2.2) -> tf.Tensor:
    return tf.pow(x, gamma)


def isclose(x: tf.Tensor, val: float, threshold: float = EPS) -> tf.Tensor:
    return tf.less_equal(tf.abs(x - val), threshold)


def safe_sqrt(x: tf.Tensor) -> tf.Tensor:
    sqrt_in = tf.nn.relu(tf.where(isclose(x, 0.0), tf.ones_like(x) * EPS, x))
    return tf.sqrt(sqrt_in)


def magnitude(x: tf.Tensor, data_format: str = "channels_last") -> tf.Tensor:
    assert data_format in ["channels_last", "channels_first"]
    return safe_sqrt(
        dot(x, x, data_format)
    )  # Relu seems strange but we're just clipping 0 values


def own_div_no_nan(
    x: tf.Tensor, y: tf.Tensor, data_format: str = "channels_last"
) -> tf.Tensor:
    return tf.where(tf.less(to_vec3(y, data_format), 1e-7), tf.zeros_like(x), x / y)


def normalize(x: tf.Tensor, data_format: str = "channels_last") -> tf.Tensor:
    assert data_format in ["channels_last", "channels_first"]
    return own_div_no_nan(x, magnitude(x, data_format), data_format)


def dot(x: tf.Tensor, y: tf.Tensor, data_format: str = "channels_last") -> tf.Tensor:
    assert data_format in ["channels_last", "channels_first"]
    return tf.reduce_sum(x * y, axis=get_channel_axis(data_format), keepdims=True)


def to_vec3(x: tf.Tensor, data_format: str = "channels_last") -> tf.Tensor:
    assert data_format in ["channels_last", "channels_first"]
    return repeat(x, 3, get_channel_axis(data_format))


def get_channel_axis(data_format: str = "channels_last") -> int:
    assert data_format in ["channels_last", "channels_first"]
    if data_format == "channels_first":
        channel_axis = 1
    else:
        channel_axis = -1

    return channel_axis


def repeat(x: tf.Tensor, n: int, axis: int) -> tf.Tensor:
    repeat = [1 for _ in range(len(x.shape))]
    repeat[axis] = n

    return tf.tile(x, repeat)


@layer_register(use_scope=None)
def INReLU(x, name=None):
    """
    A shorthand of InstanceNormalization + ReLU.
    Args:
        x (tf.Tensor): the input
    """
    x = InstanceNorm("in", x)
    return tf.nn.relu(x, name=name)


@layer_register(use_scope=None)
def upsample(x, factor: int = 2):
    _, h, w, _ = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(
        x, [factor * h, factor * w], align_corners=True
    )
    return x


@layer_register(use_scope=None)
def Fusion2DBlock(
    prevIn: Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]],
    filters: int,
    kernel_size: int,
    stride: int,
    downscale: bool = True,
    activation=INReLU,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    lmain = Conv2D("main_conv", prevIn[0], filters, kernel_size, activation=activation)
    laux = Conv2D("aux_conv", prevIn[1], filters, kernel_size, activation=activation)

    mixInput = [lmain, laux]
    prevMixOutput = prevIn[2]
    if prevMixOutput is not None:
        mixInput.append(prevMixOutput)

    mixIn = tf.concat(mixInput, -1, "mix_input")
    lmix = Conv2D("mix_conv", mixIn, filters, kernel_size, activation=activation)

    lmix = tf.add_n([laux, lmain, lmix], "mix_summation")

    if stride > 1:
        if downscale:
            lmain = MaxPooling("main_pool", lmain, 3, strides=stride, padding="SAME")
            laux = MaxPooling("aux_pool", laux, 3, strides=stride, padding="SAME")
            lmix = MaxPooling("mix_pool", lmix, 3, strides=stride, padding="SAME")
        else:
            lmain = upsample("main_upsample", lmain, factor=stride)
            laux = upsample("aux_upsample", laux, factor=stride)
            lmix = upsample("mix_upsample", lmix, factor=stride)

    return (lmain, laux, lmix)


def resnet_shortcut(
    l: tf.Tensor, n_out: int, stride: int, isDownsampling: bool, activation=tf.identity
):
    data_format = get_arg_scope()["Conv2D"]["data_format"]
    n_in = l.get_shape().as_list()[
        1 if data_format in ["NCHW", "channels_first"] else 3
    ]
    if n_in != n_out or stride != 1:  # change dimension when channel is not the same
        if isDownsampling:
            return Conv2D(
                "convshortcut", l, n_out, 1, strides=stride, activation=activation
            )
        else:
            return Conv2DTranspose(
                "convshortcut", l, n_out, 1, strides=stride, activation=activation
            )
    else:
        return l


@layer_register(use_scope=None)
def apply_preactivation(l: tf.Tensor, preact: str):
    if preact == "bnrelu":
        shortcut = l  # preserve identity mapping
        l = BNReLU("preact", l)
    if preact == "inrelu":
        shortcut = l
        l = INReLU("preact", l)
    else:
        shortcut = l
    return l, shortcut


def preresnet_basicblock(
    l: tf.Tensor,
    ch_out: int,
    stride: int,
    preact: str,
    isDownsampling: bool,
    dilation: int = 1,
    withDropout: bool = False,
):
    l, shortcut = apply_preactivation("p1", l, preact)

    if isDownsampling:
        l = Conv2D("conv1", l, ch_out, 3, strides=stride, dilation_rate=dilation)
    else:
        l = Conv2DTranspose("tconv1", l, ch_out, 3, stride=stride)

    if withDropout:
        l = Dropout(l)
    l, _ = apply_preactivation("p2", l, preact)

    l = Conv2D("conv2", l, ch_out, 3, dilation_rate=dilation)

    return l + resnet_shortcut(shortcut, ch_out, stride, isDownsampling)


def preresnet_group(
    name: str,
    l: tf.Tensor,
    block_func: Callable[[tf.Tensor, int, int, str, bool, int, bool], tf.Tensor],
    features: int,
    count: int,
    stride: int,
    isDownsampling: bool,
    activation_function: str = "inrelu",
    dilation: int = 1,
    withDropout: bool = False,
    addLongSkip: Optional[Tuple[int, tf.Tensor]] = None,
    getLongSkipFrom: Optional[int] = None,
) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    if addLongSkip and getLongSkipFrom:
        assert addLongSkip[0] != getLongSkipFrom

    if stride != 1:
        assert dilation == 1
    if dilation != 1:
        assert stride == 1

    with tf.variable_scope(name):
        if addLongSkip is not None:
            addSkipAt, skipConn = addLongSkip
        else:
            addSkipAt, skipConn = -1, None

        for i in range(0, count):
            with tf.variable_scope("block%d" % i):

                # first block doesn't need activation
                l = block_func(
                    l,
                    features,
                    stride if i == 0 else 1,
                    "no_preact" if i == 0 else activation_function,
                    isDownsampling if i == 0 else True,
                    dilation,
                    withDropout,
                )
                if getLongSkipFrom is not None:
                    if i == getLongSkipFrom:
                        skipConnection = l

                if i == addSkipAt:
                    with tf.variable_scope("long_shortcut"):
                        changed_shortcut = resnet_shortcut(
                            skipConn, l.shape[-1], 1, True
                        )
                        l = l + changed_shortcut

        # end of each group need an extra activation
        if activation_function == "bnrelu":
            l = BNReLU("bnlast", l)
        if activation_function == "inrelu":
            l = INReLU(l)

    if getLongSkipFrom is not None:
        return l, skipConnection
    else:
        return l
