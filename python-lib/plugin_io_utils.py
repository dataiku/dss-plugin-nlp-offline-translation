# -*- coding: utf-8 -*-
"""Module with read/write utility functions which are *not* based on the Dataiku API"""

from typing import AnyStr
from typing import List

import pandas as pd

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def generate_unique(name: AnyStr, existing_names: List, prefix: AnyStr = None) -> AnyStr:
    """
    Generate a unique name among existing ones by suffixing a number. Can also add an optional prefix.
    """
    if prefix is not None:
        base_name = prefix + "_" + name
    else:
        base_name = name

    new_name = base_name
    for j in range(1, 1001):
        if new_name not in existing_names:
            return new_name
        new_name = base_name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")


def validate_column_input(column_name: AnyStr, column_list: List[AnyStr]) -> None:
    """
    Validate that user input for column parameter is valid.
    """
    if column_name is None or len(column_name) == 0:
        raise ValueError("You must specify a valid column name.")
    if column_name not in column_list:
        raise ValueError("Column '{}' is not present in the input dataset.".format(column_name))
