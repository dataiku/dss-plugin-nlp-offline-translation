# -*- coding: utf-8 -*-
"""Module implementing offline translation."""

import logging
from typing import List

import numpy as np
import pandas as pd
import pysbd
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

from plugin_io_utils import generate_unique

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

LANGUAGE_CODE_LABELS = {
    "am": "Amharic",
    "ar": "Arabic",
    "ast": "Asturian",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan/Valencian",
    "ceb": "Cebuano",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greeek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "ff": "Fulah",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Gaelic/Scottish Gaelic",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian/Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "ilo": "Iloko",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Central Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "lb": "Luxembourgish/Letzeburgesch",
    "lg": "Ganda",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch/Flemish",
    "no": "Norwegian",
    "ns": "Northern Sotho",
    "oc": "Occitan (post 1500)",
    "or": "Oriya",
    "pa": "Panjabi/Punjabi",
    "pl": "Polish",
    "ps": "Pushto/Pashto",
    "pt": "Portuguese",
    "ro": "Romanian/Moldavian/Moldovan",
    "ru": "Russian",
    "sd": "Sindhi",
    "si": "Sinhala/Sinhalese",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "ss": "Swati",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "th": "Thai",
    "tl": "Tagalog",
    "tn": "Tswana",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "wo": "Wolof",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
}

# Map languages by their families to pysbd compatible languages
SRC_TO_PYSBD = {
    "am": "am",
    "ar": "ar",
    "bg": "bg",
    "da": "da",
    "de": "de",
    "el": "el",
    "en": "en",
    "es": "es",
    "fa": "fa",
    "fr": "fr",
    "hi": "hi",
    "hy": "hy",
    "it": "it",
    "ja": "ja",
    "kk": "kk",
    "mr": "mr",
    "my": "my",
    "nl": "nl",
    "pl": "pl",
    "ru": "ru",
    "sk": "sk",
    "ur": "ur",
    "zh": "zh",
}

# Define 300 as the max length of input tokens for generation
# 200 is the max output length for M2M as defined in its config available from the HF Model Hub
# Empirically M2M is at its best when being fed <300 tokens and generating <300 tokens
MAX_INPUT_TOKENS = 300

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def get_device(device: str = "CPU") -> torch.device:
    """
    Get torch device.

    Args:
        device: Specifies device to use

    Returns:
        device: Torch device if found

    Raises:
        ValueError: If GPU was selected, but not available
    """
    if device == "GPU":
        if torch.cuda.is_available():
            torch_device = torch.device("cuda")
        else:
            raise RuntimeError("GPU was selected, but no CUDA GPUs are available")
    else:
        torch_device = torch.device("cpu")

    logging.info(f"Running on {device}")
    return torch_device


class Translator:
    """
    Handles translation of pandas dataframe columns to other languages.

    Args:
        input_df: DataFrame on which to operate on
        input_column: Column of the input_df to translate
        target_language: Language to translate to
        source_language: Language to translate from
        source_language_col: Column name of column with source language codes.
            Used instead of source_language if specified.
        device: On which device to perform translation
        pretrained_model: Specifier for a huggingface pretrained translation model
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        input_column: str,
        target_language: str,
        source_language: str = None,
        source_language_col: str = None,
        device="CPU",
        pretrained_model="facebook/m2m100_418M",
    ) -> None:

        if not source_language and not source_language_col:
            raise ValueError(
                "Either a source language or a source language column must be specified."
            )

        self.input_df = input_df
        self.input_column = input_column
        self.source_language = source_language
        self.source_language_col = source_language_col
        self.target_language = target_language
        self.target_language_label = LANGUAGE_CODE_LABELS.get(target_language, "")

        self.device = get_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model).to(self.device)

        self.translated_text_column_name = generate_unique(
            f"{self.input_column}_{self.target_language}",
            self.input_df.columns,
            prefix=None,
        )
        self.column_description_dict = {
            self.translated_text_column_name: f"{self.target_language_label} translation of the '{self.input_column}' column."
        }

    def translate_df(
        self,
        split_sentences: bool = True,
        batch_size: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Applies translation to dataframe.
        """
        output_df = self.input_df.copy()

        # Multilingual case - Translate each language group separately
        if self.source_language_col:
            output_df[self.translated_text_column_name] = self.input_df.groupby(
                self.source_language_col
            )[self.input_column].transform(
                lambda x: self._translate_single_language_group(x, split_sentences, batch_size)
            )
        # Single source language case
        else:
            output_df[self.translated_text_column_name] = self._translate(
                self.input_df[self.input_column],
                tar_lang=self.target_language,
                src_lang=self.source_language,
                split_sentences=split_sentences,
                batch_size=batch_size,
                **kwargs,
            )

        return output_df

    def _translate_single_language_group(self, series: pd.Series, split_sentences, batch_size):
        """
        Handles multilingual translation with the source language specified in separate column.
        """
        source_language = self.input_df[self.source_language_col][series.index].iloc[0]
        return self._translate(
            series,
            tar_lang=self.target_language,
            src_lang=source_language,
            split_sentences=split_sentences,
            batch_size=batch_size,
        )

    def _translate(
        self,
        input_series: pd.Series,
        tar_lang: str,
        src_lang: str,
        split_sentences: bool = True,
        batch_size: int = 1,
        num_beams: int = 5,
        **kwargs,
    ) -> List[str]:
        """
        Generates translations of texts from source to target language.

        Args:
            input_series: Series with texts to process
            tar_lang: Language code of target language
            src_lang: Language code of source language
            split_sentences: Whether the model should split sentences before translating
            batch_size: Num texts to process at once
            num_beams: Number of beams for beam search, 1 means no beam search,
                the default of 5 is used by e.g. M2M100
        Returns:
            translated_texts: Translated texts
        """
        self.tokenizer.src_lang = src_lang
        if split_sentences:
            seg = pysbd.Segmenter(language=SRC_TO_PYSBD[src_lang], clean=False)

        logging.info(
            f"Starting translation of {len(input_series)} text row(s) with batch size of {batch_size} and source language {src_lang}."
        )

        translated_texts = []
        success_count = 0
        with torch.no_grad():
            for i in range(0, len(input_series), batch_size):
                # Subselect batch_size items
                batch = input_series[i : i + batch_size].tolist()
                # Turn into List[List[str]] with each str being one sentence
                if split_sentences:
                    batch = [seg.segment(txt) for txt in batch]
                else:
                    batch = [[txt] for txt in batch]
                # Prepare the model inputs
                batch_tokens = {}
                batch_ix = []
                for txt in batch:
                    for sentence in txt:
                        # Convert string to list of integers according to tokenizer's vocabulary
                        tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
                        # Enforce a maximum length in case of incorrect splitting or too long sentences
                        for i in range(0, len(tokens), MAX_INPUT_TOKENS):
                            input_dict = self.tokenizer.prepare_for_model(
                                tokens[i : i + MAX_INPUT_TOKENS], add_special_tokens=True
                            )
                            for input_type, input_list in input_dict.items():
                                batch_tokens.setdefault(input_type, [])
                                batch_tokens[input_type].append(input_list)
                        if len(tokens) > MAX_INPUT_TOKENS:
                            logging.warning(
                                f"Sentence is too long by ({len(tokens)} > {MAX_INPUT_TOKENS}) tokens, and will be translated in pieces, which might degrade performance. Check the source language and/or consider using the 'Split Sentences' option."
                            )
                    # Store the new length with each new sub_batch to discern what batch each text belongs to
                    batch_ix.append(len(batch_tokens[input_type]))
                # No need for truncation, as all inputs are now trimmed to less than the models seq length
                batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
                # Move to CPU/GPU
                batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}
                translated_batch = self.model.generate(
                    **batch_tokens,
                    forced_bos_token_id=self.tokenizer.get_lang_id(tar_lang),
                    num_beams=num_beams,
                    **kwargs,
                ).cpu()
                # Decode back to strings
                translated_batch = self.tokenizer.batch_decode(
                    translated_batch, skip_special_tokens=True
                )
                # Stitch back together by iterating through start & end indices, e.g. (0,1), (1,3)..
                translated_batch = [
                    " ".join(translated_batch[ix_s:ix_e])
                    for ix_s, ix_e in zip([0] + batch_ix, batch_ix)
                ]
                translated_texts.extend(translated_batch)
                success_count += 1

        logging.info(
            f"Successfully translated {success_count} text row(s) from {src_lang} to {tar_lang}."
        )
        return translated_texts
