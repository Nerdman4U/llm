# -*- coding: utf-8 -*-
# black: skip file
"""
T5Transformer.py
"""
from typing import TypeAlias, cast

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm.extension.base_transformer import BaseTransformer

from llm.generated.__core.generic_class_loader import load_and_validate_generated_class
generated, GeneratedClass = load_and_validate_generated_class(
    "llm.generated.t5_transformer",
    "T5Transformer",
)


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

        super().__init__(*args, **kwargs)

        if not GeneratedClass:
            raise ImportError(
                f"Generated class {__class__.__name__} not found. "
                "Ensure that the generated code is available."
            )
        kwargs['extension'] = self
        self._generated = GeneratedClass(*args, **kwargs)

    def get_model(self) -> AutoModelForSeq2SeqLM:
        """Retrieve the pre-trained BERT model with proper typing."""
        return cast(AutoModelForSeq2SeqLM, super().get_model())

    # pylint: disable=useless-parent-delegation
    def get_tokenizer(self) -> AutoTokenizer:
        """Retrieve the pre-trained BERT tokenizer with proper typing."""
        return cast(AutoTokenizer, super().get_tokenizer())


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

    __test_thinking()
