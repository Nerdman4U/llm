# -*- coding: utf-8 -*-
# black: skip file
"""
SalesforceTransformer.py
"""
from typing import TypeAlias

from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.extension.base_transformer import BaseTransformer

ModelType: TypeAlias = AutoModelForCausalLM
TokenizerType: TypeAlias = AutoTokenizer


class SalesforceTransformer(BaseTransformer):
    """
    SalesforceTransformer class is a base class for generated classes.

    Class using AutoModelForCausalLM and AutoTokenizer.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the SalesforceTransformer class with given arguments.
        """
        kwargs.setdefault("transformer_model_name", "Salesforce/codegen-350M-mono")
        kwargs.setdefault("model_type", ModelType)
        kwargs.setdefault("tokenizer_type", TokenizerType)
        super().__init__(*args, **kwargs)


if __name__ == "__main__":

    def __test_thinking():
        ai = SalesforceTransformer()
        print(ai.transformer_model_name)
        # print(ai.get_model())
        # print(ai.get_tokenizer())
        response = ai.think("def sum(*args):")
        print(response)
