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
from models.shape_network import ShapeTrainer
from models.brdf_network import BrdfTrainer
from models.illumination_network import IlluminationTrainer
from models.joint_network import JointTrainer

# This script trains the selected stage


def main():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="model", help="Step selector",)

    shape_trainer = ShapeTrainer(subparser)
    illum_trainer = IlluminationTrainer(subparser)
    brdf_trainer = BrdfTrainer(subparser)
    joint_trainer = JointTrainer(subparser)

    # Parse
    args = parser.parse_args()

    # Setup and train the corresponding network
    if args.model == "shape":
        shape_trainer.train(args)
    elif args.model == "illumination":
        illum_trainer.train(args)
    elif args.model == "brdf":
        brdf_trainer.train(args)
    elif args.model == "joint":
        joint_trainer.train(args)


if __name__ == "__main__":
    main()
