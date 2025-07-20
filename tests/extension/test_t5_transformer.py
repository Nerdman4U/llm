"""
T5Transformer.py
"""
import unittest
from unittest.mock import MagicMock, patch

import torch

from local_conjurer.extension.t5_transformer import T5Transformer

from tests.extension.base_test_class import BaseTestClass

# pylint: disable=unused-import
from tests.generated.test_t5_transformer import (
    TestT5Transformer as GeneratedTestT5Transformer
)


class TestT5Transformer(BaseTestClass):
    """
    Test cases for T5Transformer class.
    """

    def test_transformer_has_correct_values(self):
        """
        Test the get_tokenizer method of T5Transformer class.
        """
        obj = T5Transformer()
        self.assertEqual(obj.transformers_model_name, "t5-base")
        self.assertEqual(obj.model_type, "AutoModelForSeq2SeqLM")
        self.assertEqual(obj.tokenizer_type, "AutoTokenizer")

    @patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_should_generate(
        self, mock_tokenizer_from_pretrained, mock_model_from_pretrained
    ):
        """
        Test the generate method of T5Transformer class.
        """

        # Create a mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "T5 generated this text!"  # ✅ Add decode mock
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        # Create a mock model instance
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_from_pretrained.return_value = mock_model

        obj = T5Transformer()
        result = obj.conjure("Test input")

        # Verify the result is a tensor
        self.assertIsInstance(result, str)

        # Verify the model's generate method was called
        mock_model.generate.assert_called_once()

        # Verify tokenizer was called with the input text
        mock_tokenizer.assert_called_once_with("Test input", return_tensors="pt")
        mock_tokenizer.decode.assert_called_once()

    @patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_should_conjure(
        self, mock_tokenizer_from_pretrained, mock_model_from_pretrained
    ):
        """
        Test the think method of T5Transformer class.
        """
        # Create a mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Bonjour le monde"
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        # Create a mock model instance
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[4, 5, 6, 7, 8]])
        mock_model_from_pretrained.return_value = mock_model

        obj = T5Transformer()
        result = obj.conjure(
            "Translate English to French: Hello world",
            generation_kwargs={"max_length": 50},
        )

        self.assertEqual(result, "Bonjour le monde")

        # Verify the decode method was called
        mock_tokenizer.decode.assert_called_once()

    @patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_should_conjure_multiple(
        self, mock_tokenizer_from_pretrained, mock_model_from_pretrained
    ):
        """
        Test the think method of T5Transformer class with multiple inputs.
        """
        # Create a mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Bonjour le monde"
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        # Create a mock model instance
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[4, 5, 6, 7, 8]])
        mock_model_from_pretrained.return_value = mock_model

        obj = T5Transformer()
        result = obj.conjure_multiple(
            "Translate English to French: Hello world",
            generate_type="multiple",
        )

        self.assertEqual(result, ["Bonjour le monde"])

        # Verify the decode method was called
        mock_tokenizer.decode.assert_called_once()

    @patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_should_conjure_batches(
        self, mock_tokenizer_from_pretrained, mock_model_from_pretrained
    ):
        """
        Test the think method of T5Transformer class with batch inputs.
        """
        # Create a mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Bonjour le monde"
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        # Create a mock model instance
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[4, 5, 6, 7, 8]])
        mock_model_from_pretrained.return_value = mock_model

        obj = T5Transformer()
        result = obj.conjure_batches(
            [
                "Translate English to French: Hello world",
                "Translate English to Spanish: Hello world",
            ],
            generate_type="batch",
            generation_kwargs={"max_length": 50},
        )

        self.assertEqual(result, ["Bonjour le monde"])

        # Verify the decode method was called
        mock_tokenizer.decode.assert_called_once()

    @patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_should_think_with_scores(
        self, mock_tokenizer_from_pretrained, mock_model_from_pretrained
    ):
        """
        Test the think method of T5Transformer class with scores.
        """
        # Create a mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Bonjour le monde"
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        # Create a mock model instance
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[4, 5, 6, 7, 8]])
        mock_model_from_pretrained.return_value = mock_model

        obj = T5Transformer()
        result = obj.conjure_with_scores(
            "Translate English to French: Hello world",
            generate_type="with_scores",
            generation_kwargs={"max_length": 50},
        )

        self.assertEqual(
            result,
            {
                "text": "Bonjour le monde",
                "sequences": mock_model.generate.return_value,
            },
        )

        # Verify the decode method was called
        mock_tokenizer.decode.assert_called_once()


if __name__ == "__main__":
    unittest.main()
