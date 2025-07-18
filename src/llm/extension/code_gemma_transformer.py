# -*- coding: utf-8 -*-
# black: skip file
"""
CodeGemmaTransformer.py
"""
from typing import TypeAlias, cast

# from huggingface_hub import login
from transformers import GemmaTokenizer, AutoModelForCausalLM

from llm.extension.base_transformer import BaseTransformer

from llm.generated.__core.generic_class_loader import load_and_validate_generated_class
generated, GeneratedClass = load_and_validate_generated_class(
    "llm.generated.code_gemma_transformer",
    "CodeGemmaTransformer",
)


class CodeGemmaTransformer(BaseTransformer):
    """
    CodeGemmaTransformer class is a base class for generated classes.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the CodeGemmaTransformer class with given arguments.
        """
        kwargs.setdefault("transformer_model_name", "google/codegemma-2b")

        super().__init__(*args, **kwargs)

        if not GeneratedClass:
            raise ImportError(
                f"Generated class {__class__.__name__} not found. "
                "Ensure that the generated code is available."
            )
        kwargs['extension'] = self
        self._generated = GeneratedClass(*args, **kwargs)

    # pylint: disable=useless-parent-delegation
    def get_model(self) -> AutoModelForCausalLM:
        """Retrieve the pre-trained CodeGemma model with proper typing."""
        return cast(AutoModelForCausalLM, super().get_model())

    # pylint: disable=useless-parent-delegation
    def get_tokenizer(self) -> GemmaTokenizer:
        """Retrieve the pre-trained CodeGemma tokenizer with proper typing."""
        return cast(GemmaTokenizer, super().get_tokenizer())

    def generate_code(self, prompt: str, **kwargs) -> str | list[str]:
        """
        Generate code completion using CodeGemma.

        Args:
            prompt (str): Code prompt with instruction
            **kwargs: Generation parameters

        Returns:
            str: Generated code completion

        Example:
            >>> cg = CodeGemmaTransformer()
            >>> code = cg.generate_code('''
            ... def fibonacci(n):
            ...     # Complete this function to return nth fibonacci number
            ... ''')
        """
        # Set code-specific generation parameters
        code_params = {
            "max_length": 200,
            "temperature": 0.2,  # Lower temperature for more consistent code
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": self.get_tokenizer().eos_token_id,
        }
        code_params.update(kwargs)

        output = self.generate(prompt, **code_params)
        return self.decode(output)

    def complete_function(
        self, function_signature: str, description: str = "", **kwargs
    ) -> str | list[str]:
        """
        Complete a function based on its signature and description.

        Args:
            function_signature (str): The function signature
            description (str): Description of what the function should do
            **kwargs: Generation parameters

        Returns:
            str: Complete function implementation
        """
        prompt = (
            f"{function_signature}\n    # {description}\n"
            if description
            else f"{function_signature}\n"
        )
        return self.generate_code(prompt, **kwargs)

    def refactor_code(self, code: str, instruction: str, **kwargs) -> str | list[str]:
        """
        Refactor existing code based on instruction.

        Args:
            code (str): Original code
            instruction (str): Refactoring instruction
            **kwargs: Generation parameters

        Returns:
            str: Refactored code
        """
        prompt = f"{code}\n\n# {instruction}\n"
        return self.generate_code(prompt, **kwargs)

    def explain_code(self, code: str, **kwargs) -> str | list[str]:
        """
        Generate explanation for given code.

        Args:
            code (str): Code to explain
            **kwargs: Generation parameters

        Returns:
            str: Code explanation
        """
        prompt = f"{code}\n\n# Explain what this code does:\n"
        return self.generate_code(prompt, **kwargs)


if __name__ == "__main__":

    def __test_thinking():
        # Example usage
        ai = CodeGemmaTransformer()
        print(ai.transformer_model_name)
        # print(ai.get_model())
        # print(ai.get_tokenizer())

        args_input_text = """
            class Person:
                def __init__(self, name):
                    self.name = name
                    # Add age attribute with getter and setter
        """

        __main_output = ai.generate(args_input_text)
        # print(__main_output)
        __main_decoded = ai.decode(__main_output)
        # print(__main_decoded)
        print(ai.think(args_input_text, generation_kwargs={"max_length": 150}).value)

    __test_thinking()
