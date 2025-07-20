# LLM Transformer Wrappers

A Python library providing clean, easy-to-use wrapper classes for popular Hugging Face Transformer models. Simplifies working with different types of language models through a unified interface.

## Features

-   🎯 **Unified API** - Consistent interface across different transformer types
-   🚀 **Easy Integration** - Simple setup and usage
-   🧠 **Multiple AI Models** - Support for text generation, understanding, and code completion
-   🔍 **Advanced Search** - Built-in semantic search capabilities
-   ⚡ **Efficient Processing** - Batch operations and caching support
-   🧪 **Well Tested** - Comprehensive test suite with proper mocking

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd llm

# Set up virtual environment
python -m venv .venv
source .init_workspace

# Run tests
pyt

# Dry-run
python src/local_conjurer
```

### Environment Setup

For Google CodeGemma models, set your Hugging Face access token:

```bash
export HUGGING_FACE_TOKEN="your_hf_token_here"
```

## Supported Models

### 🔤 **T5 (Text-to-Text Transfer Transformer)**

**Purpose**: General text-to-text generation (translation, summarization, question answering)

T5 treats every NLP task as a text-to-text problem. Excellent for translation, summarization, and text transformation tasks.

```python
from local_conjurer.extension.t5_transformer import T5Transformer

ai = T5Transformer()
response = ai.conjure(
    "Translate English to French: The house is wonderful.",
    generation_kwargs={"max_length": 50},
).value
print(response)  # Output: "La maison est merveilleuse."
```

### 🧠 **BERT (Bidirectional Encoder Representations)**

**Purpose**: Text understanding, similarity, and semantic search

BERT excels at understanding text context and meaning. Perfect for similarity comparison, semantic search, and text classification.

```python
from local_conjurer.extension.bert_transformer import BertTransformer

ai = BertTransformer()
documents = [
    "The cat sat on the mat",
    "Dogs are loyal animals",
    "Felines are independent creatures",
    "Python is a programming language",
]
results = ai.search("cats and kittens", documents, top_k=3)
for doc, score in results:
    print(f"{score:.3f}: {doc}")
```

**Additional BERT capabilities:**

```python
# Text similarity
score = ai.similarity("I love cats", "I adore felines")  # Returns ~0.85

# Batch similarity
scores = ai.batch_similarity("machine learning", documents)
```

### 💻 **CodeGemma (Code Generation)**

**Purpose**: Code completion, generation, and programming assistance

CodeGemma specializes in understanding and generating code. Great for code completion, refactoring suggestions, and programming help.

```python
from local_conjurer.extension.code_gemma_transformer import CodeGemmaTransformer

ai = CodeGemmaTransformer()
code_prompt = """
class Person:
    def __init__(self, name):
        self.name = name
        # Add age attribute with getter and setter
"""
response = ai.conjure(code_prompt, generation_kwargs={"max_length": 150}).value
print(response)
```

**Advanced code generation:**

```python
# Function completion
code = ai.complete_function(
    "def fibonacci(n):",
    "Calculate fibonacci recursively"
)

# Code refactoring
better_code = ai.refactor_code(
    "old_code_here",
    "Make it more efficient and add error handling"
)
```

### 🏢 **Salesforce CodeT5+ (Advanced Code Understanding)**

**Purpose**: Enterprise-grade code generation and understanding

Salesforce's CodeT5+ provides advanced code generation capabilities with better understanding of code context and structure.

```python
from local_conjurer.extension.salesforce_transformer import SalesforceTransformer

ai = SalesforceTransformer()
response = ai.conjure("def calculate_fibonacci(n):")
print(response)
```

## Advanced Usage

### Generation Options

All transformers support flexible generation parameters:

```python
# Multiple alternative outputs
result = ai.conjure(
    "Your prompt here",
    generate_type="multiple",
    num_sequences=3,
    temperature=0.8
)

# Batch processing
result = ai.conjure(
    None,
    generate_type="batch",
    input_texts=["prompt1", "prompt2", "prompt3"]
)

# Generation with confidence scores
result = ai.conjure(
    "Your prompt here",
    generate_type="with_scores",
    temperature=0.7
)
```

### Caching

Models are cached locally for faster subsequent loads:

```python
# Default cache location: ./cache
# Custom cache location:
ai = T5Transformer(cache_dir="/custom/cache/path")
```

## API Reference

### Common Methods

All transformer classes inherit these methods:

-   **`conjure(prompt, \*\*kwargs)`** - Main generation method with multiple modes
-   **`conjure_multiple(prompt, \*\*kwargs)`** - Multiple varying results
-   **`conjure_with_scores(prompt, \*\*kwargs)`** - With scores
-   **`conjure_batches(prompt, \*\*kwargs)`** - Generate with batches
-   **`decode(tokens)`** - Convert tokens back to text
-   **`get_model()`** - Access the underlying Hugging Face model
-   **`get_tokenizer()`** - Access the tokenizer

### BERT-Specific Methods

-   **`similarity(text1, text2)`** - Calculate similarity between texts
-   **`search(query, documents, top_k)`** - Semantic search through documents
-   **`get_embeddings(text)`** - Get text embeddings
-   **`batch_similarity(query, documents)`** - Efficient batch similarity

### CodeGemma-Specific Methods

-   **`generate_code(prompt)`** - Optimized code generation
-   **`complete_function(signature, description)`** - Function completion
-   **`refactor_code(code, instruction)`** - Code refactoring

## Testing

Run the comprehensive test suite:

```bash
# All tests
pyt

# Specific transformer tests
python -m pytest tests/extension/test_t5_transformer.py
python -m pytest tests/extension/test_bert_transformer.py
```

## Configuration

Models can be customized during initialization:

```python
ai = T5Transformer(
    transformers_model_name="t5-large",  # Use larger model
    cache_dir="./my_cache",               # Custom cache location
    device="cuda"                         # Use GPU if available
)
```

## Requirements

-   Python 3.8+
-   PyTorch
-   Transformers
-   Additional dependencies in `requirements.txt`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## License

[Your License Here]

---

**Need help?** Check the test files in `tests/extension/` for more usage examples!
