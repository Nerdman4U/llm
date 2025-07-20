"""
T5Transformer.py
"""
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


if __name__ == "__main__":

    def __test_thinking():
        """
        Test the T5Transformer thinking capability.
        """
        ai = T5Transformer()
        response = ai.conjure(
            "Translate English to French: The house is wonderful.",
            generation_kwargs={"max_length": 50},
        )
        print(response)  # Output: La maison est merveilleuse.

    __test_thinking()
