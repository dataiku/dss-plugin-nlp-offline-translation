import os

from dataiku.code_env_resources import clear_all_env_vars
from dataiku.code_env_resources import set_env_path

### CACHE DIR & ENV VARIABLES ###
clear_all_env_vars()
set_env_path("TORCH_HOME", "pytorch")
set_env_path("HF_HOME", "huggingface")

### MODELS ###
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

pretrained_model = "facebook/m2m100_418M"

# Only need to download the relevant models
AutoTokenizer.from_pretrained(pretrained_model)
AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
