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
import traceback
from abc import ABC, abstractmethod
from typing import List

import tensorflow as tf
from tensorpack import Callback, ModelDesc
from tensorpack.callbacks import (
    EstimatedTimeLeft,
    GPUUtilizationTracker,
    InferenceRunner,
    MinSaver,
    ModelSaver,
    ScalarStats,
)
from tensorpack.dataflow import DataFlow
from tensorpack.input_source import QueueInput, StagingInput
from tensorpack.predict import PredictConfig
from tensorpack.tfutils.export import ModelExporter
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.train import (
    SimpleTrainer,
    SyncMultiGPUTrainerReplicated,
    TrainConfig,
    launch_train_with_config,
)
from tensorpack.utils import logger
from tensorpack.utils.gpu import change_gpu, get_num_gpu

import dataflow.dataflow as df


def rgbSquareImgShape(imgSize: int, name: str) -> tf.TensorSpec:
    return tf.TensorSpec([None, imgSize, imgSize, 3], tf.float32, name)


def graySquareImgShape(imgSize: int, name: str) -> tf.TensorSpec:
    return tf.TensorSpec([None, imgSize, imgSize, 1], tf.float32, name)


def sphericalGaussainsShape(num_sgs: int, name: str) -> tf.TensorSpec:
    return tf.TensorSpec([None, num_sgs, 3], tf.float32, name)


class BaseNetwork(ModelDesc, ABC):
    @property
    def training(self) -> bool:
        return get_current_tower_context().is_training

    @abstractmethod
    def network_architecture(self, *args):
        pass

    @abstractmethod
    def outputs(self) -> List[str]:
        pass

    @abstractmethod
    def output_image_names(self) -> List[str]:
        pass


class BaseTrainer(ABC):
    def __init__(self, step: df.Dataflows, model_name, subparser):
        self.step = step
        self._parse_arguments(model_name, subparser)

    @property
    def total_cost_var(self) -> str:
        return "total_costs"

    @property
    def validation_total_cost_var(self) -> str:
        return "validation_total_costs"

    def _default_callbacks(self):
        self.callbacks = [
            ModelSaver(max_to_keep=self.args.max_to_keep),
            EstimatedTimeLeft(),
        ]

        if self.args.gpu and self.args.gpu != "-1":
            self.callbacks.append(GPUUtilizationTracker())

        if self.args.validation is not None:
            self.callbacks.append(
                InferenceRunner(self.dataflow(True), [ScalarStats(self.total_cost_var)])
            )

        self.callbacks.append(
            MinSaver(
                self.validation_total_cost_var
                if self.args.validation is not None
                else self.total_cost_var
            )
        )

        self._network_specific_callbacks()

    def _network_specific_callbacks(self):
        pass

    def _parse_arguments(self, model_name, subparser):
        parser = subparser.add_parser(model_name, help="Run %s training" % model_name)
        parser.add_argument("--data", help="Path to the training data", required=True)
        parser.add_argument(
            "--save", help="The training checkpoint directory.", required=True
        )
        parser.add_argument("--load", help="Restore a model for resuming a training.")
        parser.add_argument("--validation", help="Path to the validation data.")

        parser.add_argument("--batch_size", type=int, default=8, help="The batch size.")
        parser.add_argument(
            "--image_size", type=int, default=256, help="The input image size"
        )
        parser.add_argument(
            "--epochs", type=int, default=200, help="The number of epochs to run."
        )
        parser.add_argument("--steps", type=int, default=3000, help="Steps per epoch.")
        parser.add_argument(
            "--shuffle",
            type=int,
            default=50000,
            help="Locally shuffle the required number of training examples. A larger number consumes more RAM.",
        )
        parser.add_argument(
            "--nproc", type=int, default=12, help="The number of dataflow workers."
        )
        parser.add_argument(
            "--max_to_keep",
            type=int,
            default=5,
            help="The maximum number of checkpoints to keep.",
        )

        parser.add_argument(
            "--gpu",
            help="Comma separated list of GPU(s) to use. -1 Runs training/inference on CPU.",
            default="-1",
            type=str,
        )

        self._network_specific_args(parser)

    def _network_specific_args(self, parser):
        pass

    def _dataflow(self, validation: bool = False) -> DataFlow:
        assert self.step is not None
        assert isinstance(self.step, df.Dataflows)

        logger.set_logger_dir(self.args.save, action="k")
        return df.get_data(
            self.step,
            self.args.validation
            if self.args.validation is not None
            else self.args.data,
            self.args.batch_size,
            n_proc=self.args.nproc,
            n_gpus=get_num_gpu(),
        )

    def train(self, args):
        self.args = args
        # Make sure the save path exist
        if not os.path.exists(self.args.save):
            os.makedirs(self.args.save)

        with change_gpu(self.args.gpu):
            train_df = self._dataflow()
            trainer = (
                SimpleTrainer()
                if get_num_gpu() <= 1
                else SyncMultiGPUTrainerReplicated(get_num_gpu())
            )
            print("Found %d gpus. Using trainer:" % get_num_gpu(), trainer)
            # Setup callbacks
            self._default_callbacks()
            try:
                launch_train_with_config(
                    self.pred_config(self.args, train_df, self.callbacks), trainer
                )
            except Exception as error:
                traceback.print_exc()
            else:
                # If everythin worked save a compated model
                self.export(os.path.join(self.args.save, "compact.pb"))

    def export(self, path):
        ModelExporter(self.inference_config(self.args)).export_compact(path)

    def pred_config(self, args, df, callbacks) -> TrainConfig:
        return TrainConfig(
            model=self.train_model(args),
            data=StagingInput(QueueInput(df)),
            callbacks=callbacks,
            max_epoch=args.epochs,
            steps_per_epoch=args.steps,
            session_init=SaverRestore(args.load) if args.load else None,
        )

    def inference_config(self, args) -> TrainConfig:
        loss_name = (
            self.validation_total_cost_var
            if args.validation is not None
            else self.total_cost_var
        )
        min_file = os.path.join(args.save, (f"min-{loss_name}.data-00000-of-00001"))
        model = self.inference_model(args)
        return PredictConfig(
            model=model,
            input_names=[i.name for i in model.inputs()],
            output_names=model.outputs(),
            session_init=SaverRestore(min_file),
        )

    @abstractmethod
    def train_model(self, args) -> BaseNetwork:
        pass

    @abstractmethod
    def inference_model(self, args) -> BaseNetwork:
        pass
