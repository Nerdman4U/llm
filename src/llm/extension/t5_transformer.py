# -*- coding: utf-8 -*-
# black: skip file
"""
T5Transformer.py
"""
from typing import TypeAlias

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm.extension.base_transformer import BaseTransformer

ModelType: TypeAlias = AutoModelForSeq2SeqLM
TokenizerType: TypeAlias = AutoTokenizer


class T5Transformer(BaseTransformer):
    """
    T5Transformer class is a base class for T5 transformer models.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the T5Transformer with given arguments.

        Args:
            transformer_model_name (str): Name of the transformer model.
            model_type (TypeAlias): Type of the model, defaults to AutoModelForSeq2SeqLM.
            tokenizer_type (TypeAlias): Type of the tokenizer, defaults to AutoTokenizer.
        """
        kwargs.setdefault("transformer_model_name", "t5-base")
        kwargs.setdefault("model_type", ModelType)
        kwargs.setdefault("tokenizer_type", TokenizerType)
        super().__init__(*args, **kwargs)


if __name__ == "__main__":

    def __test_thinking():
        """
        Test the T5Transformer thinking capability.
        """
        ai = T5Transformer()
        print(ai.transformer_model_name)
        # print(ai.get_model())
        # print(ai.get_tokenizer())
        # print(ai.generate("Say hi"))
        response = ai.think(
            "Translate English to French: The house is wonderful.",
            generation_kwargs={"max_length": 50},
        ).value
        print(response)  # Output: La maison est merveilleuse.
