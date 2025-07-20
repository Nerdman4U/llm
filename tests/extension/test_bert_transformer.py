"""
BertTransformer.py
"""
import unittest

from local_conjurer.extension.bert_transformer import (
    BertTransformer
)

# pylint: disable=unused-import
from tests.extension.base_test_class import BaseTestClass
from tests.generated.test_bert_transformer import (
    TestBertTransformer as GeneratedTestBertTransformer
)


class TestBertTransformer(BaseTestClass):
    """
    Test cases for BertTransformer class.
    """

    def test_transformer_has_correct_values(self):
        """
        Test the get_tokenizer method of BertTransformer class.
        """
        obj = BertTransformer()
        self.assertEqual(obj.transformers_model_name, "bert-base-uncased")
        self.assertEqual(obj.model_type, "BertModel")
        self.assertEqual(obj.tokenizer_type, "BertTokenizer")


if __name__ == '__main__':
    unittest.main()
