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

import argparse
import os
import sys
import time
import traceback
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorpack.dataflow import BatchData, PrintData
from tensorpack.utils.gpu import change_gpu
from tqdm import tqdm

from config import get_render_config
from dataflow.dataflow import InferenceDataflow, InferenceStage
from models.base import BaseNetwork
from models.brdf_network import InferenceModel as BrdfNet
from models.illumination_network import InferenceModel as IllumNet
from models.joint_network import InferenceModel as JointNet
from models.shape_network import InferenceModel as ShapeNet
from utils.config import ParameterNames, Stages
from utils.dataflow_utils import uncompressDepth, save
import utils.sg_utils as sg
from utils.common_layers import apply_mask
from utils.rendering_layer import RenderingLayer


class Predictor:
    def __init__(self, compact_graph: str, model: BaseNetwork, step: InferenceStage):
        self.compact_graph = compact_graph

        self.model = model
        self.step = step
        self.inputs = [i.name for i in model.inputs()]
        # list of tuples with output names and output image names
        assert len(model.outputs()) == len(model.output_image_names())
        self.outputs: List[Tuple[str, str]] = list(
            zip(model.outputs(), model.output_image_names())
        )

    def __enter__(self):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # Note, we just load the graph and do *not* need to initialize anything.
        with tf.gfile.GFile(self.compact_graph, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

        self.network_inputs = [
            self.sess.graph.get_tensor_by_name("import/" + n + ":0")
            for n in self.inputs
        ]
        self.network_outputs = [
            self.sess.graph.get_tensor_by_name("import/" + n[0] + ":0")
            for n in self.outputs
        ]

        return self

    def predict_all(self, data_path: str, isSrgb: bool, isLdr: bool):
        t0 = time.time()

        df = InferenceDataflow(data_path, self.step, isSrgb, isLdr)
        df = PrintData(df, max_list=7, max_depth=7)
        df.reset_state()  # Needed to setup dataflow
        for dp in tqdm(df):
            path, dp = dp
            self.predict(dp, path)

        t1 = time.time()
        print("Prediction finished in: {}".format(t1 - t0))

    def predict(self, dp, save_path: str):
        dp = [np.expand_dims(e, 0) for e in dp]
        inputDict = {name: dp[i] for i, name in enumerate(self.network_inputs)}
        outputs = self.sess.run(self.network_outputs, inputDict)

        if self.step == InferenceStage.JOINT:
            runIdx = Stages.REFINEMENT.value
        else:
            runIdx = Stages.INITIAL.value

        for i, o in enumerate(self.outputs):
            _, fileName = o
            try:
                fname = fileName % runIdx
            except TypeError:  # String does not contain formatting
                fname = fileName
            filePath = os.path.join(save_path, fname)
            res = outputs[i][0]
            if fileName == ParameterNames.DEPTH_PRED.value:
                res = uncompressDepth(res)
            print(
                f"Saving {res.shape} as {filePath} - Range {res.min()} to {res.max()}"
            )
            save(res, filePath)

    def __exit__(self, exception_type, exception_value, tb):
        if tb is not None:
            print("__exit__", exception_type, exception_value)
            print(sys.exc_info()[0])
            traceback.print_tb(tb)
        print("Session closing...")
        self.sess.close()
        tf.reset_default_graph()


def stepInference(data_path, model, step, weight_path, isSrgb: bool, isLdr: bool):
    with Predictor(weight_path, model, step) as p:
        p.predict_all(data_path, isSrgb, isLdr)


class Renderer:
    def __init__(self, step: InferenceStage):
        self.render_config = get_render_config()
        assert (
            step == InferenceStage.INITIAL_RENDERING
            or step == InferenceStage.FINAL_RENDERING
        )
        self.step = step
        self.stage = (
            Stages.INITIAL
            if step == InferenceStage.INITIAL_RENDERING
            else Stages.REFINEMENT
        )

    def __enter__(self):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        return self

    def render_all(self, data_path: str):
        t0 = time.time()

        df = InferenceDataflow(data_path, self.step)
        df = PrintData(df)
        df.reset_state()  # Needed to setup dataflow
        for dp in tqdm(df):
            path, dp = dp
            self.render(dp, path)

        t1 = time.time()
        print("Rendering finished in: {}".format(t1 - t0))

    def render(self, dp, save_path: str):
        # Batch everything
        dp = [np.expand_dims(e, 0) for e in dp]
        # Extract corresponding maps
        diffuse, specular, roughness, normal, depth, mask, sgs = dp

        # Setup everything for rendering
        imgSize = diffuse.shape[1]
        num_sgs = sgs.shape[1]  # Get the number of sgs
        intensity = self.render_config["light_intensity_lumen"] / (4.0 * np.pi)
        light_color = np.asarray(self.render_config["light_color"])
        light_color_intensity = light_color * intensity

        with self.sess.as_default():
            camera_pos = tf.expand_dims(
                tf.convert_to_tensor(
                    self.render_config["camera_position"], dtype=tf.float32
                ),
                0,
            )
            light_pos = tf.expand_dims(
                tf.convert_to_tensor(
                    self.render_config["light_position"], dtype=tf.float32
                ),
                0,
            )

            repeat = [1 for _ in range(len(mask.shape))]
            repeat[-1] = 3

            mask3 = tf.tile(mask, repeat)

            axis_sharpness = tf.expand_dims(
                tf.convert_to_tensor(
                    sg.setup_axis_sharpness(num_sgs), dtype=tf.float32
                ),
                0,
            )
            # Add a batch dim
            light_col = tf.convert_to_tensor(
                light_color_intensity.reshape([1, 3]), dtype=tf.float32
            )

            sgs_joined = tf.concat([sgs, axis_sharpness], -1)

            renderer = RenderingLayer(
                self.render_config["field_of_view"],
                self.render_config["distance_to_zero"],
                tf.TensorShape([None, imgSize, imgSize, 3]),
            )
            rerendered = renderer.call(
                diffuse,
                specular,
                roughness,
                normal,
                depth,
                mask3[:, :, :, 0:1],
                camera_pos,
                light_pos,
                light_col,
                sgs_joined,
            )
            rerendered = apply_mask(rerendered, mask3, "masked_rerender")
            result = rerendered.eval()[0]

        fname = ParameterNames.RERENDER.value % self.stage.value
        filePath = os.path.join(save_path, fname)
        save(result, filePath)

    def __exit__(self, exception_type, exception_value, tb):
        if tb is not None:
            print("__exit__", exception_type, exception_value)
            print(sys.exc_info()[0])
            traceback.print_tb(tb)
        print("Session closing...")
        self.sess.close()
        tf.reset_default_graph()


def stepRender(data_path, step):
    with Renderer(step) as r:
        r.render_all(data_path)


def shapeInference(data_path, weights, isSrgb: bool, isLdr: bool):
    print("\n\tPerforming Shape Inference...\n")
    stepInference(data_path, ShapeNet(), InferenceStage.SHAPE, weights, isSrgb, isLdr)


def illuminationInference(data_path, weights, isSrgb: bool, isLdr: bool):
    print("\n\tPerforming Illumination Inference...\n")
    stepInference(
        data_path, IllumNet(), InferenceStage.ILLUMINATION, weights, isSrgb, isLdr
    )


def brdfInference(data_path, weights, isSrgb: bool, isLdr: bool):
    print("\n\tPerforming BRDF Inference...\n")
    stepInference(data_path, BrdfNet(), InferenceStage.BRDF, weights, isSrgb, isLdr)
    stepRender(data_path, InferenceStage.INITIAL_RENDERING)


def jointInference(data_path, weights, isSrgb: bool, isLdr: bool):
    print("\n\tPerforming Refinement Inference...\n")
    stepInference(data_path, JointNet(), InferenceStage.JOINT, weights, isSrgb, isLdr)
    stepRender(data_path, InferenceStage.FINAL_RENDERING)


def fullInference(
    data_path,
    shape_weights,
    illumination_weights,
    brdf_weights,
    joint_weights,
    isSrgb: bool,
    isLdr: bool,
):
    shapeInference(data_path, shape_weights, isSrgb, isLdr)
    illuminationInference(data_path, illumination_weights, isSrgb, isLdr)
    brdfInference(data_path, brdf_weights, isSrgb, isLdr)
    jointInference(data_path, joint_weights, isSrgb, isLdr)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="Path to the inference folder.")
    parser.add_argument(
        "-s",
        "--shape_weights",
        required=True,
        help="Path to the shape network weights.",
    )
    parser.add_argument(
        "-i",
        "--illumination_weights",
        required=True,
        help="Path to the illumination network weights.",
    )
    parser.add_argument(
        "-b",
        "--brdf_weights",
        required=True,
        help="Path to the brdf network weights.",
    )
    parser.add_argument(
        "-j",
        "--joint_weights",
        required=True,
        help="Path to the joint refinement network weights.",
    )
    parser.add_argument(
        "--gpu",
        help="Comma separated list of GPU(s) to use. -1 Runs training/inference on CPU.",
        default="-1",
        type=str,
    )
    parser.add_argument(
        "--linear",
        help="The images are in a linear color space. Otherwise sRGB is assumed",
        action="store_true",
    )
    parser.add_argument(
        "--hdr",
        help="The images are in a HDR format such as .exr or .hdr. Otherwise LDR PNG images are assumed",
        action="store_true",
    )

    args = parser.parse_args()

    with change_gpu(args.gpu):
        fullInference(
            args.data,
            args.shape_weights,
            args.illumination_weights,
            args.brdf_weights,
            args.joint_weights,
            not args.linear,
            not args.hdr,
        )


if __name__ == "__main__":
    main()
