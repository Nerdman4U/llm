# -*- coding: utf-8 -*-
# black: skip file
"""
BaseTransformer.py
"""
from typing import cast, Any
from dataclasses import dataclass

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

# generated
from llm.generated.base_transformer import (
    BaseTransformer as GeneratedBaseTransformer,
)


@dataclass
class ThinkResult:
    """ThinkResult class is a data class to hold the result of the think method."""

    type: str
    value: str | list[str] | dict[str, Any]


class BaseTransformer(GeneratedBaseTransformer):
    """
    BaseTransformer class is a base class LLM classes.

    It provides easy access to the model and tokenizer, and defines methods for
    generating and decoding text given to LLM.

    Attributes:
        transformer_model_name (str) : Name of the transformer model.
        cache_dir (str)              : Directory to cache the model and tokenizer.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the T5BaseTransformer class with given arguments.

        Kwargs:
            cache_dir (str): Directory to cache the model and tokenizer.
        """
        self._model = None
        self._tokenizer = None
        self._cache_dir: str = kwargs.get("cache_dir", "./cache")
        self._transformer_model_name: str | None = kwargs.get("transformer_model_name")
        self._model_type = kwargs.get("model_type")
        self._tokenizer_type = kwargs.get("tokenizer_type")
        super().__init__(*args, **kwargs)

    @property
    def model(self):
        """
        Retrieve the pre-trained model.
        """
        if not self.cache_dir():
            raise ValueError(
                "Cache directory must be set before initializing the tokenizer."
            )
        if not self.transformer_model_name:
            raise ValueError(
                "Transformer model name must be set before initializing the model."
            )
        return self._model

    @model.setter
    def model(self, value):
        """Set the pre-trained model."""
        self._model = value

    @model.deleter
    def model(self):
        """Delete the pre-trained model."""
        self._model = None

    @property
    def tokenizer(self):
        """
        Retrieve the pre-trained tokenizer.
        """
        if not self.cache_dir():
            raise ValueError(
                "Cache directory must be set before initializing the tokenizer."
            )
        if not self.transformer_model_name:
            raise ValueError(
                "Transformer model name must be set before initializing the model."
            )
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        """Set the pre-trained tokenizer."""
        self._tokenizer = value

    @tokenizer.deleter
    def tokenizer(self):
        """Delete the pre-trained tokenizer."""
        self._tokenizer = None

    @property
    def transformer_model_name(self) -> str:
        """Retrieve the model name."""
        if not self._transformer_model_name:
            raise ValueError(
                "Transformer model name must be set before initializing the model."
            )
        return self._transformer_model_name

    @property
    def model_type(self):
        """
        Retrieve the model type.
        """
        if not self._model_type:
            raise ValueError("Model type must be set before initializing the model.")
        return self._model_type

    @property
    def tokenizer_type(self):
        """
        Retrieve the tokenizer type.
        """
        if not self._tokenizer_type:
            raise ValueError(
                "Tokenizer type must be set before initializing the tokenizer."
            )
        return self._tokenizer_type

    def cache_dir(self) -> str:
        """Retrieve the cache directory."""
        return self._cache_dir

    def get_model(self):
        """Retrieve the pre-trained model."""
        if self.model:
            return self.model

        # cache to memory - use dynamic model_type
        model_class = self.model_type
        model = model_class.from_pretrained(
            self.transformer_model_name, cache_dir=self.cache_dir()
        )
        self._model = model
        return model

    def get_tokenizer(self):
        """Retrieve the pre-trained tokenizer."""
        if self.tokenizer:
            return self.tokenizer

        # load tokenizer from network - use dynamic tokenizer_type
        tokenizer_class = self.tokenizer_type
        tok = tokenizer_class.from_pretrained(
            self.transformer_model_name, cache_dir=self.cache_dir()
        )
        self._tokenizer = tok  # cache to memory
        return tok

    def think(self, *args, **kwargs) -> ThinkResult:
        """
        This is the main method to generate text from the input.

        Args:
            input_text (str): Input text to give LLM.
            generate_type (str): Type of generation to perform. Options are:
                - "default": Default generation method.
                - "multiple": Generate multiple sequences (alternatives).
                - "with_scores": Generate with scores/probabilities.
                - "batch": Generate in batch for multiple inputs.
            generation_kwargs (dict): Additional generation parameters like:
                - max_length: Maximum length of generated sequence
                - num_return_sequences: Number of sequences to generate
                - temperature: Sampling temperature
                - do_sample: Whether to use sampling

        Returns:
            str: Decoded output from the transformer model.

        Example:
            >>> ai = T5Transformer()
            >>> ai.think("Translate English to French: The house is wonderful.")
            "La maison est merveilleuse."
        """
        input_text = args[0] if len(args) > 0 else kwargs.get("input_text")
        generate_type = (
            args[1] if len(args) > 1 else kwargs.get("generate_type", "default")
        )
        generation_kwargs = kwargs.get("generation_kwargs", {})

        if generate_type == "multiple":
            if not input_text:
                raise ValueError("Input text must be provided for generation.")
            if not isinstance(input_text, str):
                raise ValueError("Input text must be a string for default generation.")
            num_sequences = kwargs.get("num_sequences", 3)
            value = self.generate_multiple(input_text, num_sequences=num_sequences)
            return ThinkResult(type="multiple", value=value)
        elif generate_type == "with_scores":
            if not input_text:
                raise ValueError("Input text must be provided for generation.")
            if not isinstance(input_text, str):
                raise ValueError("Input text must be a string for default generation.")
            value = self.generate_with_scores(input_text, **kwargs)
            return ThinkResult(type="with_scores", value=value)
        elif generate_type == "batch":
            input_texts = kwargs.get("input_texts", [])
            if not isinstance(input_texts, list):
                raise ValueError("input_texts must be a list for batch generation.")
            value = self.batch_generate(input_texts, **kwargs)
            return ThinkResult(type="batch", value=value)
        else:
            if not input_text:
                raise ValueError("Input text must be provided for generation.")
            if not isinstance(input_text, str):
                raise ValueError("Input text must be a string for default generation.")
            # Default generation method
            if not input_text.strip():
                raise ValueError("Input text cannot be empty for generation.")

            # Set default generation parameters
            default_params = {
                "max_length": 100,
                "num_return_sequences": 1,
                "do_sample": True,
                "temperature": 0.7,
            }
            default_params.update(generation_kwargs)
            value = str(self.decode(self.generate(input_text, **default_params)))
            return ThinkResult(type="default", value=value)

    def decode_all(self, output: torch.Tensor) -> list[str]:
        tokenizer = cast(PreTrainedTokenizerBase, self.get_tokenizer())
        if not tokenizer:
            raise ValueError(
                "Tokenizer must be initialized before converting to string."
            )
        return [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]

    def decode(self, output: torch.Tensor) -> str:
        """
        Decode the output from the transformer model to a string.

        Args:
            output: The model output tensor(s)
            return_all: If True, return all sequences; if False, return only the first

        Returns:
            str or list[str]: Decoded text(s)
        """
        tokenizer = cast(PreTrainedTokenizerBase, self.get_tokenizer())
        if not tokenizer:
            raise ValueError(
                "Tokenizer must be initialized before converting to string."
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    def generate(self, input_text: str, **generation_kwargs) -> torch.Tensor:
        """
        Generate output from the transformer model based on the input text.

        Args:
            input_text (str): The input text to generate from
            **generation_kwargs: Additional generation parameters like:
                - max_length: Maximum length of generated sequence
                - num_return_sequences: Number of sequences to generate
                - temperature: Sampling temperature
                - do_sample: Whether to use sampling
                - top_k: Top-k sampling parameter
                - top_p: Top-p (nucleus) sampling parameter
        """
        tokenizer = cast(PreTrainedTokenizerBase, self.get_tokenizer())
        if not tokenizer:
            raise ValueError("Tokenizer must be initialized before generating output.")
        model = cast(PreTrainedModel, self.get_model())
        if not model:
            raise ValueError("Model must be initialized before generating output.")

        inputs = tokenizer(input_text, return_tensors="pt")

        return model.generate(**inputs, **generation_kwargs)  # type: ignore

    def generate_multiple(
        self, input_text: str, num_sequences: int = 3, **kwargs
    ) -> list[str]:
        """
        Generate multiple output sequences (alternatives) from the input
        text.

        Args:
            input_text (str): Input text to generate from
            num_sequences (int): Number of sequences to generate
            **kwargs: Additional generation parameters

        Returns:
            list[str]: List of generated text sequences
        """
        kwargs["num_return_sequences"] = num_sequences
        output = self.generate(input_text, **kwargs)
        decoded = self.decode_all(output)
        return decoded if isinstance(decoded, list) else [decoded]

    def generate_with_scores(self, input_text: str, **kwargs):
        """
        Generate output with generation scores/probabilities.

        Args:
            input_text (str): Input text to generate from
            **kwargs: Additional generation parameters

        Returns:
            dict: Dictionary containing generated text and scores
        """
        kwargs["return_dict_in_generate"] = True
        kwargs["output_scores"] = True

        tokenizer = cast(PreTrainedTokenizerBase, self.get_tokenizer())
        model = cast(PreTrainedModel, self.get_model())

        inputs = tokenizer(input_text, return_tensors="pt")

        # Set default generation parameters
        default_params = {
            "max_length": 100,
            "do_sample": True,
            "temperature": 0.7,
        }
        default_params.update(kwargs)

        output = model.generate(**inputs, **default_params)  # type: ignore

        # Decode the sequences
        sequences = output.sequences if hasattr(output, "sequences") else output
        decoded_text = self.decode(sequences)

        result = {
            "text": decoded_text,
            "sequences": sequences,
        }

        # Add scores if available
        if hasattr(output, "scores") and output.scores:
            result["scores"] = output.scores

        return result

    def batch_generate(self, input_texts: list[str], **kwargs) -> list[str]:
        """
        Generate outputs for multiple input texts in batch.

        Args:
            input_texts (list[str]): List of input texts
            **kwargs: Additional generation parameters

        Returns:
            list[str]: List of generated texts
        """
        tokenizer = cast(PreTrainedTokenizerBase, self.get_tokenizer())
        model = cast(PreTrainedModel, self.get_model())

        # Tokenize all inputs
        inputs = tokenizer(
            input_texts, return_tensors="pt", padding=True, truncation=True
        )

        # Set default generation parameters
        default_params = {
            "max_length": 100,
            "do_sample": True,
            "temperature": 0.7,
        }
        default_params.update(kwargs)

        # Generate
        outputs = model.generate(**inputs, **default_params)  # type: ignore

        # Decode all outputs
        return [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]
