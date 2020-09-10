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

# This script prepares the dataset with the current
# stage for training the next stage

import argparse
import time

from tensorpack.utils.gpu import change_gpu
from tqdm import tqdm

import infer
from dataflow.dataflow import Dataflows, InferenceDataflow, InferenceStage
from models.brdf_network import InferenceModel as BrdfNet
from models.illumination_network import InferenceModel as IllumNet
from models.joint_network import InferenceModel as JointNet
from models.shape_network import InferenceModel as ShapeNet


def translate(data_path, weight_path, model, df: Dataflows, step: InferenceStage):
    with infer.Predictor(weight_path, model, step) as p:
        t0 = time.time()

        dataset = df.value(data_path, add_path=True)
        dataset.reset_state()  # Needed to setup dataflow
        for dp in tqdm(dataset):
            path, dp = dp
            p.predict(dp, path)

        t1 = time.time()
        print("Translation finished in: {}".format(t1 - t0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["shape", "illumination", "brdf", "joint"],
        help="Step selector",
    )
    parser.add_argument(
        "--data", required=True, help="Path to the trainings data folder."
    )
    parser.add_argument(
        "-w",
        "--weights",
        required=True,
        help="Path to the corresponding network weights.",
    )
    parser.add_argument(
        "--gpu",
        help="Comma separated list of GPU(s) to use. -1 Runs training/inference on CPU.",
        default="-1",
        type=str,
    )

    args = parser.parse_args()

    with change_gpu(args.gpu):
        if args.stage == "shape":
            translate(
                args.data,
                args.weights,
                ShapeNet(),
                Dataflows.SHAPE,
                InferenceStage.SHAPE,
            )
        elif args.stage == "illumination":
            translate(
                args.data,
                args.weights,
                IllumNet(),
                Dataflows.ILLUMINATION,
                InferenceStage.ILLUMINATION,
            )
        elif args.stage == "brdf":
            translate(
                args.data, args.weights, BrdfNet(), Dataflows.BRDF, InferenceStage.BRDF,
            )
            infer.stepRender(args.data, InferenceStage.INITIAL_RENDERING)
        elif args.stage == "joint":
            translate(
                args.data,
                args.weights,
                JointNet(),
                Dataflows.JOINT,
                InferenceStage.JOINT,
            )
            infer.stepRender(args.data, InferenceStage.FINAL_RENDERING)


if __name__ == "__main__":
    main()
