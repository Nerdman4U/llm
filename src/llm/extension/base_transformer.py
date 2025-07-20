"""
BaseTransformer.py
"""
from __future__ import annotations
from typing import cast

import torch
import transformers
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

# generated
from llm.generated.__core.generic_class_loader import load_and_validate_generated_class
generated, GeneratedClass = load_and_validate_generated_class(
    "llm.generated.base_transformer",
    "BaseTransformer",
)

TYPE_CLASSES = {
    "PreTrainedModel": PreTrainedModel,
    "PreTrainedTokenizerBase": PreTrainedTokenizerBase,
    "AutoModelForSeq2SeqLM": transformers.AutoModelForSeq2SeqLM,
    "AutoTokenizer": transformers.AutoTokenizer,
    "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
    "GemmaTokenizer": transformers.GemmaTokenizer,
    "BertModel": transformers.BertModel,
    "BertTokenizer": transformers.BertTokenizer,
}


class BaseTransformer():
    """
    BaseTransformer class is a base class LLM classes.

    It provides easy access to the model and tokenizer, and defines methods for
    generating and decoding text given to LLM.

    Attributes:
        transformer_model_name (str) : Name of the transformer model.
        cache_dir (str)              : Directory to cache the model and tokenizer.
        model_type (TypeAlias)       : Type of the model, defaults to PreTrainedModel.
        tokenizer_type (TypeAlias)   : Type of the tokenizer, defaults to PreTrained

    Public methods:
        conjure
        conjure_multiple
        conjure_with_scores
        conjure_batches
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the T5BaseTransformer class with given arguments.

        Kwargs:
            cache_dir (str): Directory to cache the model and tokenizer.
        """
        self._model = None
        self._tokenizer = None
        kwargs.setdefault("cache_dir", "./cache")

        if not GeneratedClass:
            raise ImportError(
                f"Generated class {__class__.__name__} not found. "
                "Ensure that the generated code is available."
            )
        kwargs['extension'] = self
        self._generated = GeneratedClass(*args, **kwargs)

        self._cache_dir: str = self.generated().cache_dir
        self._transformers_model_name: str = self.generated().transformers_model_name
        self._model_type: str = self.generated().model_type
        self._tokenizer_type: str = self.generated().tokenizer_type

    # ---------------------------------------------
    # 1. Generated code
    # ---------------------------------------------
    def generated(self):
        """
        Retrieve the generated class instance.

        Returns:
            GeneratedClass: Instance of the generated class.
        """
        return self._generated

    def __getattr__(self, name):
        """Seamlessly delegate to generated class."""
        if self._generated and hasattr(self._generated, name):
            return getattr(self._generated, name)
        return None

    # ---------------------------------------------
    # 2. LLM properties
    #    - model
    #    - tokenizer
    # ---------------------------------------------
    @property
    def model(self):
        """
        Retrieve the pre-trained model.
        """
        if not self.cache_dir:
            raise ValueError(
                "Cache directory must be set before initializing the tokenizer."
            )
        if not self.transformers_model_name:
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
        if not self.cache_dir:
            raise ValueError(
                "Cache directory must be set before initializing the tokenizer."
            )
        if not self.transformers_model_name:
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

    # ---------------------------------------------
    # 3. Protected methods
    # ---------------------------------------------
    def _get_model(self):
        """Retrieve the pre-trained model."""
        if self.model:
            return self.model

        model_type_str = self.model_type
        if not model_type_str:
            raise ValueError("Model type must be set before initializing the model.")

        # cache to memory - use dynamic model_type
        model_class = TYPE_CLASSES.get(model_type_str)
        if not model_class:
            raise ValueError("Model type must be set before initializing the model.")
        if not self.transformers_model_name:
            raise ValueError(
                "Transformer model name must be set before initializing the model."
            )
        model = model_class.from_pretrained(
            self.transformers_model_name, cache_dir=self.cache_dir
        )

        # cache to memory
        self._model = model
        return model

    def _get_tokenizer(self):
        """Retrieve the pre-trained tokenizer."""
        if self.tokenizer:
            return self.tokenizer

        tokenizer_type_str = self.tokenizer_type
        if not tokenizer_type_str:
            raise ValueError(
                "Tokenizer type must be set before initializing the tokenizer.")

        # load tokenizer from network - use dynamic tokenizer_type
        tokenizer_class = TYPE_CLASSES.get(tokenizer_type_str)
        if not tokenizer_class:
            raise ValueError(
                "Tokenizer type must be set before initializing the tokenizer.")
        tok = tokenizer_class.from_pretrained(
            self.transformers_model_name, cache_dir=self.cache_dir
        )
        self._tokenizer = tok  # cache to memory
        return tok

    def _decode_all(self, output: torch.Tensor) -> list[str]:
        """
        Decode the output from the transformer model to a list of strings.

        Args:
            output: The model output tensor(s)

        Returns:
            list[str]: Decoded text(s)
        """
        tokenizer = cast(PreTrainedTokenizerBase, self._get_tokenizer())
        if not tokenizer:
            raise ValueError(
                "Tokenizer must be initialized before converting to string."
            )
        return [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]

    def _decode(self, output: torch.Tensor) -> str:
        """
        Decode the output from the transformer model to a string.

        Args:
            output: The model output tensor(s)
            return_all: If True, return all sequences; if False, return only the first

        Returns:
            str: Decoded text(s)
        """
        tokenizer = cast(PreTrainedTokenizerBase, self._get_tokenizer())
        if not tokenizer:
            raise ValueError(
                "Tokenizer must be initialized before converting to string."
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    def _generate(self, input_text: str, **generation_kwargs) -> torch.Tensor:
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
        tokenizer = cast(PreTrainedTokenizerBase, self._get_tokenizer())
        if not tokenizer:
            raise ValueError("Tokenizer must be initialized before generating output.")
        model = cast(PreTrainedModel, self._get_model())
        if not model:
            raise ValueError("Model must be initialized before generating output.")

        inputs = tokenizer(input_text, return_tensors="pt")

        return model.generate(**inputs, **generation_kwargs)  # type: ignore

    # ---------------------------------------------
    # 4. Public methods
    # ---------------------------------------------
    def conjure(self, prompt, **kwargs) -> str:
        """
        Generate a result from the prompt using the model.

        Args:
            prompt (str): The input text to generate from
            **kwargs: Additional generation parameters

        Returns:
            str: The generated text
        """
        return self._decode(self._generate(prompt, **kwargs))

    def conjure_multiple(
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
        decoded = self._decode_all(self._generate(input_text, **kwargs))
        return decoded if isinstance(decoded, list) else [decoded]

    def conjure_with_scores(self, input_text: str, **kwargs):
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

        tokenizer = cast(PreTrainedTokenizerBase, self._get_tokenizer())
        model = cast(PreTrainedModel, self._get_model())

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
        decoded_text = self._decode(sequences)

        result = {
            "text": decoded_text,
            "sequences": sequences,
        }

        # Add scores if available
        if hasattr(output, "scores") and output.scores:
            result["scores"] = output.scores

        return result

    def conjure_batches(self, input_texts: list[str], **kwargs) -> list[str]:
        """
        Generate outputs for multiple input texts in batch.

        What happens under the hood:

        Sequential: GPU sits mostly idle
        GPU: [▓░░░] [▓░░░] [▓░░░]  ← Only ~25% utilization per call
              call1  call2  call3

        Batch: GPU fully utilized
        GPU: [▓▓▓▓] ← ~90% utilization, processes all 3 at once!
              batch

        Args:
            input_texts (list[str]): List of input texts
            **kwargs: Additional generation parameters

        Returns:
            list[str]: List of generated texts
        """
        tokenizer = cast(PreTrainedTokenizerBase, self._get_tokenizer())
        model = cast(PreTrainedModel, self._get_model())

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
