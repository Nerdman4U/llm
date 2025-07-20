"""
BertTransformer.py
"""
import torch

from local_conjurer.extension.base_transformer import BaseTransformer


from local_conjurer.generated.__core.generic_class_loader import load_and_validate_generated_class
generated, GeneratedClass = load_and_validate_generated_class(
    "local_conjurer.generated.bert_transformer",
    "BertTransformer",
)


class BertTransformer(BaseTransformer):
    """
    BertTransformer class for BERT transformer models. BERT stands for Bidirectional Encoder
    Representations from Transformers. It's a transformer model developed by Google that's
    specifically designed for understanding text rather than generating it.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BertTransformer class with given arguments.
        """
        super().__init__(*args, **kwargs)

        if not GeneratedClass:
            raise ImportError(
                f"Generated class {__class__.__name__} not found. "
                "Ensure that the generated code is available."
            )
        kwargs['extension'] = self
        self._generated = GeneratedClass(*args, **kwargs)

        self._cache_dir: str = self.generated().cache_dir
        self._transformers_model_name: str = self.generated().transformers_model_name
        self._model_type: str = self.generated().model_type
        self._tokenizer_type: str = self.generated().tokenizer_type

    def _generate(self, *args, **kwargs):
        """
        Generate method for BertTransformer. BERT is primarily used for understanding text,
        so this method may not be applicable in the same way as in generative models.
        """
        raise NotImplementedError(
            "BERT is not a generative model, so generate is not implemented."
        )

    def _encode(self, text: str):
        """
        Encode text into BERT embeddings.

        Args:
            text (str): Input text to encode

        Returns:
            Tensor: BERT embeddings for the input text
        """
        tokenizer = self._get_tokenizer()
        model = self._get_model()

        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Get BERT embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        return outputs.last_hidden_state

    def get_embeddings(self, text: str):
        """
        Get sentence-level embeddings by pooling BERT's output.

        Args:
            text (str): Input text

        Returns:
            Tensor: Sentence-level embeddings
        """
        embeddings = self._encode(text)
        # Use [CLS] token embedding as sentence representation
        return embeddings[:, 0, :]  # First token is [CLS]

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using BERT embeddings.

        Args:
            text1 (str): First text to compare
            text2 (str): Second text to compare

        Returns:
            float: Cosine similarity score between -1 and 1
                  (1 = identical, 0 = unrelated, -1 = opposite)

        Example:
            >>> ai = BertTransformer()
            >>> score = ai.similarity("I love cats", "I adore felines")
            >>> print(score)  # Should be high, around 0.8-0.9
            >>> score = ai.similarity("I love cats", "I adore dogs")
            >>> print(score)  # Should be low, around -0.2-0.3
        """
        # Get embeddings for both texts
        emb1 = self.get_embeddings(text1)
        emb2 = self.get_embeddings(text2)

        # Calculate cosine similarity
        # pylint: disable=not-callable
        similarity_score = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)

        return float(similarity_score.item())

    def search(
        self, query: str, documents: list[str], top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Search through documents to find the most similar ones to the query.

        Args:
            query (str): The search query text
            documents (list[str]): List of documents to search through
            top_k (int): Number of top results to return (default: 5)

        Returns:
            list[tuple[str, float]]: List of (document, similarity_score) tuples,
                                   sorted by similarity score (highest first)

        Example:
            >>> bert = BertTransformer()
            >>> docs = [
            ...     "The cat sat on the mat",
            ...     "Dogs are loyal animals",
            ...     "Felines are independent creatures",
            ...     "Python is a programming language"
            ... ]
            >>> results = bert.search("cats and kittens", docs, top_k=3)
            >>> for doc, score in results:
            ...     print(f"{score:.3f}: {doc}")
        """
        if not documents:
            return []

        # Calculate similarity scores for all documents
        scores = []
        for doc in documents:
            score = self.similarity(query, doc)
            scores.append((doc, score))

        # Sort by similarity score (highest first) and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def batch_similarity(self, query: str, documents: list[str]) -> list[float]:
        """
        Calculate similarity scores between a query and multiple documents efficiently.

        Args:
            query (str): The query text
            documents (list[str]): List of documents to compare against

        Returns:
            list[float]: Similarity scores for each document
        """
        if not documents:
            return []

        tokenizer = self._get_tokenizer()
        model = self._get_model()

        # Get query embedding
        query_inputs = tokenizer(
            query, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            query_outputs = model(**query_inputs)
            query_embedding = query_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Process documents in batches for efficiency
        batch_size = 32
        all_scores = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i: i + batch_size]

            # Tokenize batch of documents
            doc_inputs = tokenizer(
                batch_docs, return_tensors="pt", padding=True, truncation=True
            )

            with torch.no_grad():
                doc_outputs = model(**doc_inputs)
                doc_embeddings = doc_outputs.last_hidden_state[:, 0, :]  # [CLS] tokens

                # Calculate similarities for this batch
                # pylint: disable=not-callable
                batch_scores = torch.nn.functional.cosine_similarity(
                    query_embedding.expand(doc_embeddings.size(0), -1),
                    doc_embeddings,
                    dim=1,
                )
                all_scores.extend(batch_scores.tolist())

        return all_scores

    def search_efficient(
        self, query: str, documents: list[str], top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Efficient search through large collections of documents using batch processing.

        Args:
            query (str): The search query text
            documents (list[str]): List of documents to search through
            top_k (int): Number of top results to return (default: 5)

        Returns:
            list[tuple[str, float]]: List of (document, similarity_score) tuples,
                                   sorted by similarity score (highest first)
        """
        if not documents:
            return []

        # Use batch processing for efficiency
        scores = self.batch_similarity(query, documents)

        # Combine documents with their scores
        doc_scores = list(zip(documents, scores))

        # Sort by similarity score (highest first) and return top_k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]


if __name__ == "__main__":

    def __test_similarity():
        ai = BertTransformer()
        print(ai.similarity("I love cats", "I adore felines"))
        # >> 0.968

    def __test_search():
        ai = BertTransformer()
        documents = [
            "The cat sat on the mat",
            "Dogs are loyal animals",
            "Felines are independent creatures",
            "Python is a programming language",
        ]
        results = ai.search("cats and kittens", documents, top_k=3)
        print("Result:")
        print("-------------------")
        for doc, score in results:
            print(f"{score:.3f}: {doc}")

    __test_search()
    # __test_similarity()
