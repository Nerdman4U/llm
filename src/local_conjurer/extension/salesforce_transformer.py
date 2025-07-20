"""
SalesforceTransformer.py
"""
from local_conjurer.extension.base_transformer import BaseTransformer

from local_conjurer.generated.__core.generic_class_loader import load_and_validate_generated_class
generated, GeneratedClass = load_and_validate_generated_class(
    "local_conjurer.generated.salesforce_transformer",
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


if __name__ == "__main__":

    def __test_thinking():
        ai = SalesforceTransformer()
        response = ai.conjure("def sum(*args):")
        print("Result:")
        print("-------------------")
        print(response)

    __test_thinking()
