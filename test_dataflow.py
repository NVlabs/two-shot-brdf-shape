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

from dataflow.dataflow import Dataflows, get_data

if __name__ == "__main__":
    import argparse
    from tensorpack.dataflow import TestDataSpeed, PrintData
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3])

    args = parser.parse_args()
    print(args)

    if args.stage == 0:
        df = Dataflows.SHAPE
    elif args.stage == 1:
        df = Dataflows.ILLUMINATION
    elif args.stage == 2:
        df = Dataflows.BRDF
    elif args.stage == 3:
        df = Dataflows.JOINT

    ds = get_data(df, args.folder, args.batch_size)
    ds = PrintData(ds)
    TestDataSpeed(ds, 1000).start()
