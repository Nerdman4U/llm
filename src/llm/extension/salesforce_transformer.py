# -*- coding: utf-8 -*-
# black: skip file
"""
SalesforceTransformer.py
"""
from typing import TypeAlias, cast

from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.extension.base_transformer import BaseTransformer

from llm.generated.__core.generic_class_loader import load_and_validate_generated_class
generated, GeneratedClass = load_and_validate_generated_class(
    "llm.generated.salesforce_transformer",
    "SalesforceTransformer",
)


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

        super().__init__(*args, **kwargs)

        if not GeneratedClass:
            raise ImportError(
                f"Generated class {__class__.__name__} not found. "
                "Ensure that the generated code is available."
            )
        kwargs['extension'] = self
        self._generated = GeneratedClass(*args, **kwargs)

    # def get_model(self) -> AutoModelForCausalLM:
    #     """Retrieve the pre-trained BERT model with proper typing."""
    #     return cast(AutoModelForCausalLM, super().get_model())

    # # pylint: disable=useless-parent-delegation
    # def get_tokenizer(self) -> AutoTokenizer:
    #     """Retrieve the pre-trained BERT tokenizer with proper typing."""
    #     return cast(AutoTokenizer, super().get_tokenizer())


if __name__ == "__main__":

    def __test_thinking():
        ai = SalesforceTransformer()
        print(ai.transformers_model_name)
        # print(ai.get_model())
        # print(ai.get_tokenizer())
        response = ai.think("def sum(*args):")
        print(response)

    __test_thinking()
