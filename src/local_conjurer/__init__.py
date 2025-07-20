"""
🪄 Local Conjurer - Summon Code Magic on Your Own Machine

Transform your ideas into code with the power of local AI models.
"""
import importlib.metadata

from .extension.base_transformer import BaseTransformer
from .extension.bert_transformer import BertTransformer
from .extension.t5_transformer import T5Transformer
from .extension.code_gemma_transformer import CodeGemmaTransformer
from .extension.salesforce_transformer import SalesforceTransformer

# Clean aliases for conjurers:
CodeGemmaConjurer = CodeGemmaTransformer
T5Conjurer = T5Transformer
BertConjurer = BertTransformer
BaseConjurer = BaseTransformer
Salesforce = SalesforceTransformer

# User-friendly aliases:
CodeGemma = CodeGemmaTransformer
T5 = T5Transformer
Bert = BertTransformer
Sales = SalesforceTransformer

__all__ = [
    "BaseTransformer", "BaseConjurer",
    "BertTransformer", "BertConjurer", "Bert",
    "T5Transformer", "T5Conjurer", "T5",
    "CodeGemmaTransformer", "CodeGemmaConjurer", "CodeGemma",
]

try:
    __version_number__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version_number__ = "0.0.0"
