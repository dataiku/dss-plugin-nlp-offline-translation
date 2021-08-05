# -*- coding: utf-8 -*-
"""Module implementing offline translation."""

import logging
from typing import Any, AnyStr
from typing import List

import numpy as np
import pandas as pd
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
    "fy": "Western",
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
    "km": "Central",
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
    "ns": "Northern",
    "oc": "Occitan",
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


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def get_device(device: str = "CPU") -> torch.device:
    """
    Get torch device.
    Args:
        device: Specifies device to use.
    Raises:
        ValueError: If GPU was selected, but not available.
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
        input_column: AnyStr,
        target_language: AnyStr,
        source_language: AnyStr = None,
        source_language_col: AnyStr = None,
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
        batch_size: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Applies translation to dataframe.
        """
        output_df = self.input_df.copy()

        output_df[self.translated_text_column_name] = self._translate(
            self.input_df,
            self.input_column,
            self.target_language,
            self.source_language,
            self.source_language_col,
            batch_size,
            **kwargs,
        )

        return output_df

    def _translate(
        self,
        input_df: pd.DataFrame,
        input_col: AnyStr,
        tar_lang: AnyStr,
        src_lang: AnyStr = None,
        src_lang_col: AnyStr = None,
        batch_size: int = 1,
        **kwargs,
    ) -> List[str]:
        """
        Greedily generates translations of texts from source to target language.

        Args:
            input_df: Dataframe to process
            input_col: Column name of texts to translate
            tar_lang: Language code of target language
            src_lang: Language code of source language
            src_lang_col: Column name of column with source language codes.
                Used instead of source_language if specified.
            batch_size: Num texts to process at once

        Returns:
            translated_texts: Translated texts
        """
        if src_lang:
            self.tokenizer.src_lang = src_lang

        logging.info(
            f"Starting translation of {len(input_df[input_col])} text rows with batch size of {batch_size}."
        )

        if src_lang_col and not (len(np.unique(input_df[src_lang_col])) == 1) and batch_size > 1:
            logging.warn(
                f"Using a source language column with a batch size bigger than 1 may lead to translation errors. "
                + "Make sure to either have segments of source languages spaced in the same way as the batch size or "
                + "use a batch size of 1."
            )

        translated_texts = []
        success_count = 0
        with torch.no_grad():
            for i in range(0, len(input_df[input_col]), batch_size):
                # Set source language
                if src_lang_col:
                    src_lang = input_df[src_lang_col][i]
                    if not (src_lang in LANGUAGE_CODE_LABELS):
                        logging.warn(
                            f"Skipping row number {i}, as language code '{src_lang}' is not available. Make sure it is in ISO 639-1 form and available for the model https://huggingface.co/{pretrained_model}"
                        )
                        translated_texts.append(
                            f"Language code '{src_lang}' is not available. Make sure it is in ISO 639-1 form and available for the model https://huggingface.co/{pretrained_model}"
                        )
                        continue
                    self.tokenizer.src_lang = src_lang

                # Subselect batch_size items
                batch = input_df[input_col][i : i + batch_size].tolist()
                # Truncate or pad to max sequence length of model
                batch = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                gen = self.model.generate(
                    **batch, forced_bos_token_id=self.tokenizer.get_lang_id(tar_lang), **kwargs
                ).cpu()
                translated_texts.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
                success_count += 1

        logging.info(f"Successfully translated {success_count} text rows.")
        return translated_texts
