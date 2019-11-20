from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import inspect
import logging
import os
import sys

import colorama
import torch

import attacks
import augmentations
import datasets
import holistic_records
import logger
import losses
import models
from utils import json
from utils import strings
from utils import type_inference as typeinf


def _add_arguments_for_module(parser,
                              module,
                              name,
                              default_class,
                              add_class_argument=True,  # whether to add class choice as argument
                              include_classes="*",
                              exclude_classes=(),
                              exclude_params=("self", "args"),
                              param_defaults=(),  # allows to overwrite any default param
                              forced_default_types=(),  # allows to set types for known arguments
                              unknown_default_types=()):  # allows to set types for unknown arguments

    # -------------------------------------------------------------------------
    # Gets around the issue of mutable default arguments
    # -------------------------------------------------------------------------
    exclude_params = list(exclude_params)
    param_defaults = dict(param_defaults)
    forced_default_types = dict(forced_default_types)
    unknown_default_types = dict(unknown_default_types)

    # -------------------------------------------------------------------------
    # Determine possible choices from class names in module, possibly apply include/exclude filters
    # -------------------------------------------------------------------------
    module_dict = typeinf.module_classes_to_dict(
        module, include_classes=include_classes, exclude_classes=exclude_classes)

    # -------------------------------------------------------------------------
    # Parse known arguments to determine choice for argument name
    # -------------------------------------------------------------------------
    if add_class_argument:
        parser.add_argument(
            "--%s" % name, type=str, default=default_class, choices=module_dict.keys())
        known_args = parser.parse_known_args(sys.argv[1:])[0]
    else:
        # build a temporary parser, and do not add the class as argument
        tmp_parser = argparse.ArgumentParser()
        tmp_parser.add_argument(
            "--%s" % name, type=str, default=default_class, choices=module_dict.keys())
        known_args = tmp_parser.parse_known_args(sys.argv[1:])[0]

    class_name = vars(known_args)[name]

    # -------------------------------------------------------------------------
    # If class is None, there is no point in trying to parse further arguments
    # -------------------------------------------------------------------------
    if class_name is None:
        return

    # -------------------------------------------------------------------------
    # Get constructor of that argument choice
    # -------------------------------------------------------------------------
    class_constructor = module_dict[class_name]

    # -------------------------------------------------------------------------
    # Determine constructor argument names and defaults
    # -------------------------------------------------------------------------
    try:
        argspec = inspect.getargspec(class_constructor.__init__)
        argspec_defaults = argspec.defaults if argspec.defaults is not None else []
        full_args = argspec.args
        default_args_dict = dict(zip(argspec.args[-len(argspec_defaults):], argspec_defaults))
    except TypeError:
        print(argspec)
        print(argspec.defaults)
        raise ValueError("unknown_default_types should be adjusted for module: '%s.py'" % name)

    def _get_type_from_arg(arg):
        if isinstance(arg, bool):
            return strings.as_bool_or_none
        else:
            return type(arg)

    # -------------------------------------------------------------------------
    # Add sub_arguments
    # -------------------------------------------------------------------------
    for argname in full_args:

        # ---------------------------------------------------------------------
        # Skip
        # ---------------------------------------------------------------------
        if argname in exclude_params:
            continue

        # ---------------------------------------------------------------------
        # Sub argument name
        # ---------------------------------------------------------------------
        sub_arg_name = "%s_%s" % (name, argname)

        # ---------------------------------------------------------------------
        # If a default argument is given, take that one
        # ---------------------------------------------------------------------
        if argname in param_defaults.keys():
            parser.add_argument(
                "--%s" % sub_arg_name,
                type=_get_type_from_arg(param_defaults[argname]),
                default=param_defaults[argname])

        # ---------------------------------------------------------------------
        # If a default parameter can be inferred from the module, pick that one
        # ---------------------------------------------------------------------
        elif argname in default_args_dict.keys():

            # -----------------------------------------------------------------
            # Check for forced default types
            # -----------------------------------------------------------------
            if argname in forced_default_types.keys():
                argtype = forced_default_types[argname]
            else:
                argtype = _get_type_from_arg(default_args_dict[argname])
            parser.add_argument(
                "--%s" % sub_arg_name, type=argtype, default=default_args_dict[argname])

        # ---------------------------------------------------------------------
        # Take from the unkowns list
        # ---------------------------------------------------------------------
        elif argname in unknown_default_types.keys():
            parser.add_argument("--%s" % sub_arg_name, type=unknown_default_types[argname])

        else:
            raise ValueError(
                "Do not know how to handle argument '%s' for class '%s'" % (argname, name))


def _add_special_arguments(parser):
    # -------------------------------------------------------------------------
    # Known arguments so far
    # -------------------------------------------------------------------------
    known_args = vars(parser.parse_known_args(sys.argv[1:])[0])

    # -------------------------------------------------------------------------
    # Add special arguments for training
    # -------------------------------------------------------------------------
    loss = known_args["loss"]
    if loss is not None:
        parser.add_argument("--training_key", type=str, default="total_loss")

    # -------------------------------------------------------------------------
    # Add special arguments for validation
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--validation_keys", type=strings.as_stringlist_or_none, default="[total_loss]")
    parser.add_argument(
        "--validation_keys_minimize", type=strings.as_booleanlist_or_none, default="[True]")

    # -------------------------------------------------------------------------
    # Add special arguments for checkpoints
    # -------------------------------------------------------------------------
    checkpoint = known_args["checkpoint"]
    if checkpoint is not None:
        parser.add_argument(
            "--checkpoint_mode", type=str, default="resume_from_latest",
            choices=["resume_from_latest", "resume_from_best"])

        parser.add_argument(
            "--checkpoint_include_params", type=strings.as_stringlist_or_none, default="[*]")
        parser.add_argument(
            "--checkpoint_exclude_params", type=strings.as_stringlist_or_none, default="[]")

    # -------------------------------------------------------------------------
    # Add special arguments for optimizer groups
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--optimizer_group", action="append", type=strings.as_dict_or_none, default=None)


def _parse_arguments():
    # -------------------------------------------------------------------------
    # Argument parser and shortcut function to add arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    # -------------------------------------------------------------------------
    # Standard arguments
    # -------------------------------------------------------------------------
    add("--batch_size", type=int, default=1)
    add("--checkpoint", type=strings.as_string_or_none, default=None)
    add("--cuda", type=strings.as_bool_or_none, default=True)
    add("--evaluation", type=strings.as_bool_or_none, default=False)
    add("--logging_loss_graph", type=strings.as_bool_or_none, default=False)
    add("--logging_model_graph", type=strings.as_bool_or_none, default=False)
    add("--name", default="run", type=str)
    add("--num_workers", type=int, default=4)
    add("--proctitle", default="./workwork", type=str)
    add("--save", "-s", default="/tmp/work", type=str)
    add("--seed", type=int, default=1)
    add("--start_epoch", type=int, default=1)
    add("--total_epochs", type=int, default=10)

    # -------------------------------------------------------------------------
    # Arguments inferred from losses
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        losses,
        name="loss",
        default_class=None,
        exclude_classes=["_*", "Variable"],
        exclude_params=["self", "args"])

    # -------------------------------------------------------------------------
    # Arguments inferred from models
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        models,
        name="model",
        default_class="FlowNet1S",
        exclude_classes=["_*", "Variable"],
        exclude_params=["self", "args"])

    # -------------------------------------------------------------------------
    # Arguments inferred from attacks
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        attacks,
        name="attack",
        default_class=None,
        exclude_classes=["_*", "Variable"],
        exclude_params=["self", "args"])

    # -------------------------------------------------------------------------
    # Arguments inferred from augmentations for training
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        augmentations,
        name="training_augmentation",
        default_class=None,
        exclude_classes=["_*"],
        exclude_params=["self", "args"],
        forced_default_types={"crop": strings.as_intlist_or_none})

    # -------------------------------------------------------------------------
    # Arguments inferred from augmentations for validation
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        augmentations,
        name="validation_augmentation",
        default_class=None,
        exclude_classes=["_*"],
        exclude_params=["self", "args"])

    # -------------------------------------------------------------------------
    # Arguments inferred from datasets for training
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        datasets,
        name="training_dataset",
        default_class=None,
        exclude_params=["self", "args", "is_cropped"],
        exclude_classes=["_*"],
        unknown_default_types={"root": str},
        forced_default_types={"photometric_augmentations": strings.as_dict_or_none,
                              "affine_augmentations": strings.as_dict_or_none})

    # -------------------------------------------------------------------------
    # Arguments inferred from datasets for validation
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        datasets,
        name="validation_dataset",
        default_class=None,
        exclude_params=["self", "args", "is_cropped"],
        exclude_classes=["_*"],
        unknown_default_types={"root": str},
        forced_default_types={"photometric_augmentations": strings.as_dict_or_none,
                              "affine_augmentations": strings.as_dict_or_none})

    # -------------------------------------------------------------------------
    # Arguments inferred from PyTorch optimizers
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        torch.optim,
        name="optimizer",
        default_class="Adam",
        exclude_classes=["_*", "Optimizer", "constructor"],
        exclude_params=["self", "args", "params"],
        forced_default_types={"lr": float,
                              "momentum": float,
                              "betas": strings.as_floatlist_or_none,
                              "dampening": float,
                              "weight_decay": float,
                              "nesterov": strings.as_bool_or_none})

    # -------------------------------------------------------------------------
    # Arguments inferred from PyTorch lr schedulers
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        torch.optim.lr_scheduler,
        name="lr_scheduler",
        default_class=None,
        exclude_classes=["_*", "constructor"],
        exclude_params=["self", "args", "optimizer"],
        unknown_default_types={"T_max": int,
                               "lr_lambda": str,
                               "step_size": int,
                               "milestones": strings.as_intlist_or_none,
                               "gamma": float})

    # -------------------------------------------------------------------------
    # Arguments inferred from holistic records
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        holistic_records,
        default_class="EpochRecorder",
        name="holistic_records",
        add_class_argument=False,
        exclude_classes=["_*"],
        exclude_params=["self", "args", "root", "epoch", "dataset"])

    # -------------------------------------------------------------------------
    # Special arguments
    # -------------------------------------------------------------------------
    _add_special_arguments(parser)

    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Parse default arguments from a dummy commandline not specifying any args
    # -------------------------------------------------------------------------
    defaults = vars(parser.parse_known_args(['--dummy'])[0])

    # -------------------------------------------------------------------------
    # Consistency checks
    # -------------------------------------------------------------------------
    args.cuda = args.cuda and torch.cuda.is_available()

    return args, defaults


def postprocess_args(args):
    # ----------------------------------------------------------------------------
    # Get appropriate class constructors from modules
    # ----------------------------------------------------------------------------
    args.model_class = typeinf.module_classes_to_dict(models)[args.model]

    if args.optimizer is not None:
        optimizer_classes = typeinf.module_classes_to_dict(torch.optim)
        args.optimizer_class = optimizer_classes[args.optimizer]

    if args.loss is not None:
        loss_classes = typeinf.module_classes_to_dict(losses)
        args.loss_class = loss_classes[args.loss]

    if args.lr_scheduler is not None:
        scheduler_classes = typeinf.module_classes_to_dict(torch.optim.lr_scheduler)
        args.lr_scheduler_class = scheduler_classes[args.lr_scheduler]

    if args.training_dataset is not None:
        dataset_classes = typeinf.module_classes_to_dict(datasets)
        args.training_dataset_class = dataset_classes[args.training_dataset]

    if args.validation_dataset is not None:
        dataset_classes = typeinf.module_classes_to_dict(datasets)
        args.validation_dataset_class = dataset_classes[args.validation_dataset]

    if args.training_augmentation is not None:
        augmentation_classes = typeinf.module_classes_to_dict(augmentations)
        args.training_augmentation_class = augmentation_classes[args.training_augmentation]

    if args.validation_augmentation is not None:
        augmentation_classes = typeinf.module_classes_to_dict(augmentations)
        args.validation_augmentation_class = augmentation_classes[args.validation_augmentation]

    if args.attack is not None:
        attack_classes = typeinf.module_classes_to_dict(attacks)
        args.attack_class = attack_classes[args.attack]

    # ----------------------------------------------------------------------------
    # holistic records
    # ----------------------------------------------------------------------------
    holistic_records_args = typeinf.kwargs_from_args(args, "holistic_records")
    for key, value in holistic_records_args.items():
        setattr(args, "holistic_records_kwargs", holistic_records_args)

    return args


def setup_logging_and_parse_arguments(blocktitle):
    # ----------------------------------------------------------------------------
    # Get parse commandline and default arguments
    # ----------------------------------------------------------------------------
    args, defaults = _parse_arguments()

    # ----------------------------------------------------------------------------
    # Setup logbook before everything else
    # ----------------------------------------------------------------------------
    logger.configure_logging(os.path.join(args.save, "logbook.txt"))

    # ----------------------------------------------------------------------------
    # Write arguments to file, as json and txt
    # ----------------------------------------------------------------------------
    json.write_dictionary_to_file(
        vars(args), filename=os.path.join(args.save, "args.json"), sortkeys=True)
    json.write_dictionary_to_file(
        vars(args), filename=os.path.join(args.save, "args.txt"), sortkeys=True)

    # ----------------------------------------------------------------------------
    # Log arguments
    # ----------------------------------------------------------------------------
    with logger.LoggingBlock(blocktitle, emph=True):
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.CYAN
            if isinstance(value, dict):
                for sub_argument, sub_value in collections.OrderedDict(value).items():
                    logging.info("{}{}_{}: {}{}".format(color, argument, sub_argument, sub_value, reset))
            else:
                logging.info("{}{}: {}{}".format(color, argument, value, reset))

    # ----------------------------------------------------------------------------
    # Postprocess
    # ----------------------------------------------------------------------------
    args = postprocess_args(args)

    return args
