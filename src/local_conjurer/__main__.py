"""
Local Conjurer: Your AI Code Assistant

Transform your ideas into code with the power of local AI models.
"""
# Style 1: Full descriptive names
from local_conjurer import CodeGemma, Bert, T5, Sales

# # Style 2: Short magical aliases
conjurer = CodeGemma()

# # Style 3: Multiple conjurers
text_wizard = Bert()
translation_wizard = T5()
code_wizard = Sales()

# # All use the same magical methods:
# magic = code_wizard.conjure("def fibonacci(n):")
print("Similarity: ", text_wizard.similarity(
    "What is the capital of France?",
    "Paris is the capital of France."))

print("Translation: ", translation_wizard.conjure(
    "Translate to French: 'Hello, how are you?'"
))
