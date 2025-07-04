# -*- coding: utf-8 -*-
# black: skip file
"""
T5Transformer.py

GENERATED BY METAPROJECT 1.4.0.
"""
import unittest

# fmt: off
from llm.extension.t5_transformer import (
    T5Transformer
)
# fmt: on

from tests.extension import shared_test_methods
from tests.extension.base_test_class import BaseTestClass

class TestT5Transformer(BaseTestClass):
    """
    Test cases for T5Transformer class.
    """
    def setUp(self):
        """Set up the test environment."""
        shared_test_methods.set_up(self)

    def tearDown(self):
        """Tear down the test environment."""
        to_be_removed = []
        shared_test_methods.tear_down(self, to_be_removed)

    def test_should_initialize(self):
        """Test if the class can be initialized."""
        obj = T5Transformer()
        self.assertTrue(obj)

    def test_should_have_initialization_params(self):
        """Test if the class has initialization parameters."""
        obj = T5Transformer()
        self.assertTrue(obj.initialization_params())

    def test_should_return_itself_as_a_str(self):
        """Test if the class can return itself as a string."""
        obj = T5Transformer(a=1)
        self.assertTrue(isinstance(str(obj), str))
        self.assertTrue(len(str(obj)) > 0)

    def test_should_return_itself_as_a_repr(self):
        """Test if the class can return itself as a representation."""
        obj = T5Transformer()
        self.assertTrue(isinstance(repr(obj), str))
        self.assertTrue(len(repr(obj)) > 0)

    def test_should_return_dict_representation(self):
        """Test if the class can return a dictionary representation."""
        obj = T5Transformer(a=1)
        result = obj.to_dict()
        self.assertTrue(isinstance(result, dict))
        self.assertTrue(len(result) > 0)
        self.assertEqual(result["a"], 1)

    def test_should_create_from_dict(self):
        """Test if the class can be created from a dictionary."""
        obj = T5Transformer()
        new_obj = T5Transformer.from_dict(**obj.initialization_params().to_dict())
        self.assertTrue(new_obj)
        self.assertTrue(isinstance(new_obj, T5Transformer))
        self.assertEqual(new_obj.to_dict(), obj.to_dict())
        self.assertEqual(
            new_obj.initialization_params().to_dict(),
            obj.initialization_params().to_dict()
        )

if __name__ == '__main__':
    unittest.main()
