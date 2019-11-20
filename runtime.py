from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging

import colorama
import numpy as np
import torch

import logger
from holistic_records import EpochRecorder
from utils.moving_averages import MovingAverage

# --------------------------------------------------------------------------------
# Exponential moving average smoothing factor for speed estimates
# Ranges from 0 (average speed) to 1 (current/instantaneous speed) [default: 0.3].
# --------------------------------------------------------------------------------
TQDM_SMOOTHING = 1


# -------------------------------------------------------------------------------------------
# Magic progressbar for inputs of type 'iterable'
# -------------------------------------------------------------------------------------------
def create_progressbar(iterable,
                       desc="",
                       train=False,
                       unit="it",
                       initial=0,
                       offset=0,
                       invert_iterations=False,
                       logging_on_update=False,
                       logging_on_close=True,
                       postfix=False):
    # ---------------------------------------------------------------
    # Pick colors
    # ---------------------------------------------------------------
    reset = colorama.Style.RESET_ALL
    bright = colorama.Style.BRIGHT
    cyan = colorama.Fore.CYAN
    dim = colorama.Style.DIM
    green = colorama.Fore.GREEN

    # ---------------------------------------------------------------
    # Specify progressbar layout:
    #   l_bar, bar, r_bar, n, n_fmt, total, total_fmt, percentage,
    #   rate, rate_fmt, rate_noinv, rate_noinv_fmt, rate_inv,
    #   rate_inv_fmt, elapsed, remaining, desc, postfix.
    # ---------------------------------------------------------------
    bar_format = ""
    bar_format += "%s==>%s%s {desc}:%s " % (cyan, reset, bright, reset)  # description
    bar_format += "{percentage:3.0f}%"  # percentage
    bar_format += "%s|{bar}|%s " % (dim, reset)  # bar
    bar_format += " {n_fmt}/{total_fmt}  "  # i/n counter
    bar_format += "{elapsed}<{remaining}"  # eta
    if invert_iterations:
        bar_format += " {rate_inv_fmt}  "  # iteration timings
    else:
        bar_format += " {rate_noinv_fmt}  "
    bar_format += "%s{postfix}%s" % (green, reset)  # postfix

    # ---------------------------------------------------------------
    # Specify TQDM arguments
    # ---------------------------------------------------------------
    tqdm_args = {
        "iterable": iterable,
        "desc": desc,  # Prefix for the progress bar
        "total": len(iterable),  # The number of expected iterations
        "leave": True,  # Leave progress bar when done
        "miniters": 1 if train else None,  # Minimum display update interval in iterations
        "unit": unit,  # String be used to define the unit of each iteration
        "initial": initial,  # The initial counter value.
        "dynamic_ncols": True,  # Allow window resizes
        "smoothing": TQDM_SMOOTHING,  # Moving average smoothing factor for speed estimates
        "bar_format": bar_format,  # Specify a custom bar string formatting
        "position": offset,  # Specify vertical line offset
        "ascii": True,
        "logging_on_update": logging_on_update,
        "logging_on_close": logging_on_close
    }

    return logger.tqdm_with_logging(**tqdm_args)


def tensor2float_dict(tensor_dict):
    return {key: tensor.item() for key, tensor in tensor_dict.items()}


def format_moving_averages_as_progress_dict(moving_averages_dict, moving_averages_postfix="avg"):
    value = [
        (key + moving_averages_postfix, "%1.4f" % moving_averages_dict[key].mean())
        for key in sorted(moving_averages_dict.keys())
    ]
    progress_dict = collections.OrderedDict(value)
    return progress_dict


def format_learning_rate(lr):
    if np.isscalar(lr):
        return "{}".format(lr)
    else:
        return "{}".format(str(lr[0]) if len(lr) == 1 else lr)


def configure_holistic_epoch_recorder(args, epoch, loader):
    epoch_recorder = EpochRecorder(
        args,
        epoch=epoch,
        dataset=loader.dataset.__class__.__name__,
        **args.holistic_records_kwargs)
    return epoch_recorder


class TrainingEpoch:
    def __init__(self,
                 args,
                 model_and_loss,
                 device,
                 loader,
                 optimizer,
                 augmentation=None,
                 add_progress_stats=(),
                 desc="Training Epoch"):

        self._args = args
        self._desc = desc
        self._device = device
        self._loader = loader
        self._model_and_loss = model_and_loss
        self._optimizer = optimizer
        self._augmentation = augmentation
        self._add_progress_stats = dict(add_progress_stats)

    def _step(self, example_dict):

        # -------------------------------------------------------------
        # Get input and target tensor keys
        # -------------------------------------------------------------
        input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
        target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
        tensor_keys = input_keys + target_keys

        # -------------------------------------------------------------
        # Possibly transfer to Cuda
        # -------------------------------------------------------------
        for key, value in example_dict.items():
            if key in tensor_keys:
                example_dict[key] = value.to(self._device)

        # -------------------------------------------------------------
        # Optionally perform augmentations
        # -------------------------------------------------------------
        if self._augmentation is not None:
            example_dict = self._augmentation(example_dict)

        # -------------------------------------------------------------
        # Extract batch size from first input
        # -------------------------------------------------------------
        batch_size = example_dict["input1"].size(0)

        # -------------------------------------------------------------
        # Reset gradients
        # -------------------------------------------------------------
        self._optimizer.zero_grad()

        # -------------------------------------------------------------
        # Run forward pass to get losses and outputs.
        # -------------------------------------------------------------
        loss_dict, output_dict = self._model_and_loss(example_dict)

        # -------------------------------------------------------------
        # Check total_loss for NaNs
        # -------------------------------------------------------------
        loss = loss_dict[self._args.training_key]
        assert (not np.isnan(loss.item())), "training loss is NaN"

        # -------------------------------------------------------------
        # Back propagation
        # -------------------------------------------------------------
        loss.backward()

        # -------------------------------------------------------------
        # Optimizer step
        # -------------------------------------------------------------
        self._optimizer.step()

        # -------------------------------------------------------------
        # Return success flag, loss and output dictionary
        # -------------------------------------------------------------
        return loss_dict, output_dict, batch_size

    def run(self, offset=0):
        # ---------------------------------------
        # Tell model that we want to train
        # ---------------------------------------
        self._model_and_loss.train()

        # ---------------------------------------
        # Keep track of moving averages
        # ---------------------------------------
        moving_averages_dict = None

        # ---------------------------------------
        # Progress bar arguments
        # ---------------------------------------
        progressbar_args = {
            "iterable": self._loader,
            "desc": self._desc,
            "train": True,
            "offset": offset,
            "logging_on_update": False,
            "logging_on_close": True,
            "postfix": True
        }

        # ---------------------------------------
        # Perform training steps
        # ---------------------------------------
        output_dict = {}
        with create_progressbar(**progressbar_args) as progress:
            for example_dict in progress:
                # perform step
                loss_dict_per_step, output_dict, batch_size = self._step(example_dict)
                # convert
                loss_dict_per_step = tensor2float_dict(loss_dict_per_step)

                # --------------------------------------------------------
                # Possibly initialize moving averages
                # --------------------------------------------------------
                if moving_averages_dict is None:
                    moving_averages_dict = {
                        key: MovingAverage() for key in loss_dict_per_step.keys()
                    }

                # --------------------------------------------------------
                # Add moving averages
                # --------------------------------------------------------
                for key, loss in loss_dict_per_step.items():
                    moving_averages_dict[key].add_average(loss, addcount=batch_size)

                # view statistics in progress bar
                progress_stats = format_moving_averages_as_progress_dict(
                    moving_averages_dict=moving_averages_dict,
                    moving_averages_postfix="_ema")

                progress.set_postfix(progress_stats)

        # -------------------------------------------------------------
        # Return loss and output dictionary
        # -------------------------------------------------------------
        ema_loss_dict = {key: ma.mean() for key, ma in moving_averages_dict.items()}
        return ema_loss_dict, output_dict


class EvaluationEpoch:
    def __init__(self,
                 args,
                 model_and_loss,
                 device,
                 loader,
                 recorder,
                 attack=None,
                 augmentation=None,
                 add_progress_stats=(),
                 desc="Evaluation Epoch"):

        self._args = args
        self._desc = desc
        self._loader = loader
        self._model_and_loss = model_and_loss
        self._device = device
        self._attack = attack
        self._add_progress_stats = dict(add_progress_stats)
        self._recorder = recorder
        self._augmentation = augmentation

    def _step(self, example_dict):
        # -------------------------------------------------------------
        # Get input and target tensor keys
        # -------------------------------------------------------------
        input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
        target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
        tensor_keys = input_keys + target_keys

        # -------------------------------------------------------------
        # Possibly transfer to Cuda
        # -------------------------------------------------------------
        for key, value in example_dict.items():
            if key in tensor_keys:
                example_dict[key] = value.to(self._device)

        # -------------------------------------------------------------
        # Optionally perform augmentations
        # -------------------------------------------------------------
        if self._augmentation is not None:
            example_dict = self._augmentation(example_dict)

        if self._attack is not None:
            with torch.set_grad_enabled(True):
                example_dict = self._attack(self._model_and_loss, example_dict)

        # -------------------------------------------------------------
        # Extract batch size from first input
        # -------------------------------------------------------------
        batch_size = example_dict["input1"].size(0)

        # -------------------------------------------------------------
        # Run forward pass to get losses and outputs.
        # -------------------------------------------------------------
        loss_dict, output_dict = self._model_and_loss(example_dict)

        # -------------------------------------------------------------
        # Return loss and output dictionary
        # -------------------------------------------------------------
        return loss_dict, output_dict, batch_size

    def run(self, offset=0):
        # ---------------------------------------
        # Tell model that we want to evaluate
        # ---------------------------------------
        self._model_and_loss.eval()

        # ---------------------------------------
        # Keep track of moving averages
        # ---------------------------------------
        moving_averages_dict = None

        # ---------------------------------------
        # Progress bar arguments
        # ---------------------------------------
        progressbar_args = {
            "iterable": self._loader,
            "desc": self._desc,
            "train": False,
            "offset": offset,
            "logging_on_update": False,
            "logging_on_close": True,
            "postfix": True
        }

        # ---------------------------------------
        # Perform evaluation steps
        # ---------------------------------------
        output_dict = {}
        with create_progressbar(**progressbar_args) as progress:
            for example_dict in progress:

                # ---------------------------------------
                # Perform forward evaluation step
                # ---------------------------------------
                loss_dict_per_step, output_dict, batch_size = self._step(example_dict)

                # ---------------------------------------
                # recorder
                # ---------------------------------------
                # self._recorder.add_image(
                #     example_dict["basename"],
                #     example_dict["input1"])

                # ---------------------------------------
                # Convert loss dictionary to float
                # ---------------------------------------
                loss_dict_per_step = tensor2float_dict(loss_dict_per_step)

                # --------------------------------------------------------
                # Possibly initialize moving averages
                # --------------------------------------------------------
                if moving_averages_dict is None:
                    moving_averages_dict = {
                        key: MovingAverage() for key in loss_dict_per_step.keys()
                    }

                # --------------------------------------------------------
                # Add moving averages
                # --------------------------------------------------------
                for key, loss in loss_dict_per_step.items():
                    moving_averages_dict[key].add_average(loss, addcount=batch_size)

                # view statistics in progress bar
                progress_stats = format_moving_averages_as_progress_dict(
                    moving_averages_dict=moving_averages_dict,
                    moving_averages_postfix="_avg")

                progress.set_postfix(progress_stats)

        # -------------------------------------------------------------
        # Record average losses
        # -------------------------------------------------------------
        avg_loss_dict = {key: ma.mean() for key, ma in moving_averages_dict.items()}
        self._recorder.add_scalars("evaluation_losses", avg_loss_dict)

        # -------------------------------------------------------------
        # Return average losses and output dictionary
        # -------------------------------------------------------------
        return avg_loss_dict, output_dict


def exec_runtime(args,
                 device,
                 checkpoint_saver,
                 model_and_loss,
                 optimizer,
                 attack,
                 lr_scheduler,
                 train_loader,
                 validation_loader,
                 inference_loader,
                 training_augmentation,
                 validation_augmentation):
    # ----------------------------------------------------------------------------------------------
    # Validation schedulers are a bit special:
    # They want to be called with a validation loss..
    # ----------------------------------------------------------------------------------------------
    validation_scheduler = (lr_scheduler is not None and args.lr_scheduler == "ReduceLROnPlateau")

    # --------------------------------------------------------
    # Log some runtime info
    # --------------------------------------------------------
    with logger.LoggingBlock("Runtime", emph=True):
        logging.info("start_epoch: %i" % args.start_epoch)
        logging.info("total_epochs: %i" % args.total_epochs)

    # ---------------------------------------
    # Total progress bar arguments
    # ---------------------------------------
    progressbar_args = {
        "desc": "Progress",
        "initial": args.start_epoch - 1,
        "invert_iterations": True,
        "iterable": range(1, args.total_epochs + 1),
        "logging_on_close": True,
        "logging_on_update": True,
        "postfix": False,
        "unit": "ep"
    }

    # --------------------------------------------------------
    # Total progress bar
    # --------------------------------------------------------
    print(''), logging.logbook('')
    total_progress = create_progressbar(**progressbar_args)
    print("\n")

    # --------------------------------------------------------
    # Remember validation losses
    # --------------------------------------------------------
    num_validation_losses = len(args.validation_keys)
    best_validation_losses = [
        float("inf") if args.validation_keys_minimize[i] else -float("inf")
        for i in range(num_validation_losses)
    ]
    store_as_best = [False for i in range(num_validation_losses)]

    # --------------------------------------------------------
    # Transfer model to device once before training/evaluation
    # --------------------------------------------------------
    model_and_loss = model_and_loss.to(device)

    avg_loss_dict = {}
    for epoch in range(args.start_epoch, args.total_epochs + 1):
        with logger.LoggingBlock("Epoch %i/%i" % (epoch, args.total_epochs), emph=True):

            # --------------------------------------------------------
            # Update standard learning scheduler
            # --------------------------------------------------------
            if lr_scheduler is not None and not validation_scheduler:
                lr_scheduler.step(epoch)

            # --------------------------------------------------------
            # Always report learning rate and model
            # --------------------------------------------------------
            if lr_scheduler is None:
                logging.info("model: %s  lr: %s" % (args.model, format_learning_rate(args.optimizer_lr)))
            else:
                logging.info("model: %s  lr: %s" % (args.model, format_learning_rate(lr_scheduler.get_lr())))

            # -------------------------------------------
            # Create and run a training epoch
            # -------------------------------------------
            if train_loader is not None:
                avg_loss_dict, _ = TrainingEpoch(
                    args,
                    desc="   Train",
                    device=device,
                    model_and_loss=model_and_loss,
                    optimizer=optimizer,
                    loader=train_loader,
                    augmentation=training_augmentation).run()

            # -------------------------------------------
            # Create and run a validation epoch
            # -------------------------------------------
            if validation_loader is not None:
                # ---------------------------------------------------
                # Construct holistic recorder for epoch
                # ---------------------------------------------------
                epoch_recorder = configure_holistic_epoch_recorder(
                    args, epoch=epoch, loader=validation_loader)

                with torch.no_grad():
                    avg_loss_dict, output_dict = EvaluationEpoch(
                        args,
                        desc="Validate",
                        device=device,
                        model_and_loss=model_and_loss,
                        attack=attack,
                        loader=validation_loader,
                        recorder=epoch_recorder,
                        augmentation=validation_augmentation).run()

                # ----------------------------------------------------------------
                # Evaluate valdiation losses
                # ----------------------------------------------------------------
                validation_losses = [avg_loss_dict[vkey] for vkey in args.validation_keys]
                for i, (vkey, vminimize) in enumerate(zip(args.validation_keys, args.validation_keys_minimize)):
                    if vminimize:
                        store_as_best[i] = validation_losses[i] < best_validation_losses[i]
                    else:
                        store_as_best[i] = validation_losses[i] > best_validation_losses[i]
                    if store_as_best[i]:
                        best_validation_losses[i] = validation_losses[i]

                # ----------------------------------------------------------------
                # Update validation scheduler, if one is in place
                # We use the first key in validation keys as the relevant one
                # ----------------------------------------------------------------
                if lr_scheduler is not None and validation_scheduler:
                    lr_scheduler.step(validation_losses[0], epoch=epoch)

            # ----------------------------------------------------------------
            # Also show best loss on total_progress
            # ----------------------------------------------------------------
            total_progress_stats = {
                "best_" + vkey + "_avg": "%1.4f" % best_validation_losses[i]
                for i, vkey in enumerate(args.validation_keys)
            }
            total_progress.set_postfix(total_progress_stats)

            # ----------------------------------------------------------------
            # Bump total progress
            # ----------------------------------------------------------------
            total_progress.update()
            print('')

            # ----------------------------------------------------------------
            # Store checkpoint
            # ----------------------------------------------------------------
            if checkpoint_saver is not None:
                checkpoint_saver.save_latest(
                    directory=args.save,
                    model_and_loss=model_and_loss,
                    stats_dict=dict(avg_loss_dict, epoch=epoch),
                    store_as_best=store_as_best,
                    store_prefixes=args.validation_keys)

            # ----------------------------------------------------------------
            # Vertical space between epochs
            # ----------------------------------------------------------------
            print(''), logging.logbook('')

    # ----------------------------------------------------------------
    # Finish
    # ----------------------------------------------------------------
    total_progress.close()
    logging.info("Finished.")
