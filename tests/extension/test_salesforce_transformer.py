"""
SalesforceTransformer.py
"""
import unittest

from llm.extension.salesforce_transformer import (
    SalesforceTransformer
)

# pylint: disable=unused-import
from tests.generated.test_salesforce_transformer import (
    TestSalesforceTransformer as GeneratedTestSalesforceTransformer
)
from tests.extension.base_test_class import BaseTestClass


class TestSalesforceTransformer(BaseTestClass):
    """
    Test cases for SalesforceTransformer class.
    """

    def test_transformer_has_correct_values(self):
        """
        Test the get_tokenizer method of BertTransformer class.
        """
        obj = SalesforceTransformer()
        self.assertEqual(obj.transformers_model_name, "Salesforce/codegen-350M-mono")
        self.assertEqual(obj.model_type, "AutoModelForCausalLM")
        self.assertEqual(obj.tokenizer_type, "AutoTokenizer")


if __name__ == "__main__":
    unittest.main()
