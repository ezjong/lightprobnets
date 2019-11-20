from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from .strings import filter_list_of_strings


# -------------------------------------------------------------------------------------------------
# Looks for sub arguments in the argument structure.
# Retrieve sub arguments for modules such as optimizer_*
# -------------------------------------------------------------------------------------------------
def kwargs_from_args(args, name, exclude=[]):
    if isinstance(exclude, str):
        exclude = [exclude]
    exclude += ["class"]
    args_dict = vars(args)
    name += "_"
    subargs_dict = {
        key[len(name):]: value for key, value in args_dict.items()
        if name in key and all([key != name + x for x in exclude])
    }
    return subargs_dict


# -------------------------------------------------------------------------------------------------
# Create class instance from kwargs dictionary.
# Filters out keys that not in the constructor
# -------------------------------------------------------------------------------------------------
def instance_from_kwargs(class_constructor, kwargs):
    argspec = inspect.getargspec(class_constructor.__init__)
    full_args = argspec.args
    filtered_args = dict([(k, v) for k, v in kwargs.items() if k in full_args])
    instance = class_constructor(**filtered_args)
    return instance


def module_classes_to_dict(module, include_classes="*", exclude_classes=()):
    # -------------------------------------------------------------------------
    # If arguments are strings, convert them to a list
    # -------------------------------------------------------------------------
    if include_classes is not None:
        if isinstance(include_classes, str):
            include_classes = [include_classes]

    if exclude_classes is not None:
        if isinstance(exclude_classes, str):
            exclude_classes = [exclude_classes]

    # -------------------------------------------------------------------------
    # Obtain dictionary from given module
    # -------------------------------------------------------------------------
    item_dict = dict([(name, getattr(module, name)) for name in dir(module)])

    # -------------------------------------------------------------------------
    # Filter classes
    # -------------------------------------------------------------------------
    item_dict = dict([
        (name, value) for name, value in item_dict.items() if inspect.isclass(getattr(module, name))
    ])

    filtered_keys = filter_list_of_strings(
        item_dict.keys(), include=include_classes, exclude=exclude_classes)

    # -------------------------------------------------------------------------
    # Construct dictionary from matched results
    # -------------------------------------------------------------------------
    result_dict = dict([(name, value) for name, value in item_dict.items() if name in filtered_keys])

    return result_dict
