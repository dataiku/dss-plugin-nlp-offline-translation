# -*- coding: utf-8 -*-
import json
from typing import AnyStr
from typing import Dict

import dataiku
from dataiku.customrecipe import get_recipe_config
from dataiku.customrecipe import get_input_names_for_role
from dataiku.customrecipe import get_output_names_for_role

from dkulib.dku_io_utils import set_column_descriptions
from plugin_io_utils import validate_column_input
from translate import Translator


# ==============================================================================
# SETUP
# ==============================================================================

# Recipe parameters
text_column = get_recipe_config().get("text_column")
source_language = get_recipe_config().get("source_language", "")
# Handle Multilingual source language case
source_language_col = None
if source_language == "source_language_col":
    source_language_col = get_recipe_config().get("source_language_col", None)
target_language = get_recipe_config().get("target_language", "")
batch_size = get_recipe_config().get("batch_size", 1)
device = "gpu" if get_recipe_config().get("use_gpu", False) else "cpu"

# ==============================================================================
# DEFINITIONS
# ==============================================================================

input_dataset = dataiku.Dataset(get_input_names_for_role("input_dataset")[0])
output_dataset = dataiku.Dataset(get_output_names_for_role("output_dataset")[0])
validate_column_input(text_column, [col["name"] for col in input_dataset.read_schema()])
input_df = input_dataset.get_dataframe()

translator = Translator(
    input_df=input_df,
    input_column=text_column,
    target_language=target_language,
    source_language=source_language,
    source_language_col=source_language_col,
    device=device,
)

# ==============================================================================
# RUN
# ==============================================================================

output_df = translator.translate_df(batch_size=batch_size)

output_dataset.write_with_schema(output_df)

set_column_descriptions(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    column_descriptions=translator.column_description_dict,
)
