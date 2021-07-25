# -*- coding: utf-8 -*-
import json
from typing import Dict
from typing import AnyStr

import dataiku
from dataiku.customrecipe import get_recipe_config
from dataiku.customrecipe import get_input_names_for_role
from dataiku.customrecipe import get_output_names_for_role

from dku_io_utils import set_column_description
from plugin_io_utils import validate_column_input
from translate import Translator


# ==============================================================================
# SETUP
# ==============================================================================

# Recipe parameters
text_column = get_recipe_config().get("text_column")
target_language = get_recipe_config().get("target_language", "")
source_language = get_recipe_config().get("source_language", "")
batch_size = get_recipe_config().get("batch_size", 1)
device = get_recipe_config().get("device", "cpu")

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
    device=device,
)

# ==============================================================================
# RUN
# ==============================================================================

output_df = translator.translate_df(batch_size=batch_size)

output_dataset.write_with_schema(output_df)

set_column_description(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    column_description_dict=translator.column_description_dict,
)
