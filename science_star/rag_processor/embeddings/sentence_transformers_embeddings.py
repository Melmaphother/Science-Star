from __future__ import annotations

from typing import Any

from numpy import ndarray

from rag_processor.embeddings.base import BaseEmbedding


class SentenceTransformerEncoder(BaseEmbedding[str]):
    r"""This class provides functionalities to generate text
    embeddings using `Sentence Transformers`.

    References:
        https://www.sbert.net/
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        **kwargs,
    ):
        r"""Initializes the: obj: `SentenceTransformerEmbedding` class
        with the specified transformer model.

        Args:
            model_name (str, optional): The name of the model to use.
                (default: :obj:`intfloat/e5-large-v2`)
            **kwargs (optional): Additional arguments of
                :class:`SentenceTransformer`, such as :obj:`prompts` etc.
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, **kwargs)

    def embed_list(
        self,
        objs: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        r"""Generates embeddings for the given texts using the model.

        Args:
            objs (list[str]): The texts for which to generate the
                embeddings.

        Returns:
            list[list[float]]: A list that represents the generated embedding
                as a list of floating-point numbers.
        """
        if not objs:
            raise ValueError("Input text list is empty")
        embeddings = self.model.encode(
            objs, normalize_embeddings=True, **kwargs
        )
        assert isinstance(embeddings, ndarray)
        return embeddings.tolist()

    def get_output_dim(self) -> int:
        r"""Returns the output dimension of the embeddings.

        Returns:
            int: The dimensionality of the embeddings.
        """
        output_dim = self.model.get_sentence_embedding_dimension()
        assert isinstance(output_dim, int)
        return output_dim
