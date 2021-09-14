import pytest
import pandas as pd

from translate import Translator


def test_m2m():
    """
    Tests the M2M model translation from English to Chinese on a tiny dataset.
    """
    toy_df = pd.DataFrame({"translate_me": ["Hello"], "dont_translate": ["Bye"]})

    # Instantiate a translator with a tiny m2m to avoid downloading >1GB
    m2m = Translator(
        toy_df,
        input_column="translate_me",
        target_language="zh",
        source_language="en",
        device="CPU",
        pretrained_model="valhalla/m2m100_tiny_random",
        revision="337a4a691b7e14ad1668f5f4e481eaea6ce59ba1",
    )

    translated_df = m2m.translate_df()

    assert translated_df["translate_me"].values[0] == toy_df["translate_me"].values[0]
    assert translated_df["dont_translate"].values[0] == toy_df["dont_translate"].values[0]
    assert translated_df["translate_me_zh"].values[0] != translated_df["translate_me"].values[0]
    assert "dont_translate_zh" not in translated_df
