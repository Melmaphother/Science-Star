from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from rag_processor.embeddings.base import BaseEmbedding
from rag_processor.types import NOT_GIVEN, EmbeddingModelType, NotGiven
from rag_processor.utils import api_keys_required


class OpenAIEmbedding(BaseEmbedding[str]):
    r"""Provides text embedding functionalities using OpenAI's models.

    Args:
        model_type (EmbeddingModelType, optional): The model type to be
            used for text embeddings.
            (default: :obj:`TEXT_EMBEDDING_3_SMALL`)
        api_key (str, optional): The API key for authenticating with the
            OpenAI service. (default: :obj:`None`)
        dimensions (int, optional): The text embedding output dimensions.
            (default: :obj:`NOT_GIVEN`)

    Raises:
        RuntimeError: If an unsupported model type is specified.
    """

    def __init__(
        self,
        model_type: EmbeddingModelType = (
            EmbeddingModelType.TEXT_EMBEDDING_3_SMALL
        ),
        api_key: str | None = None,
        dimensions: int | NotGiven = NOT_GIVEN,
    ) -> None:
        if not model_type.is_openai:
            raise ValueError("Invalid OpenAI embedding model type.")
        self.model_type = model_type
        if dimensions == NOT_GIVEN:
            self.output_dim = model_type.output_dim
        else:
            assert isinstance(dimensions, int)
            self.output_dim = dimensions

        self._api_key = os.environ.get("OPENAI_API_KEY")
        self._base_url = os.environ.get("OPENAI_BASE_URL")


        self.client = OpenAI(timeout=60, max_retries=3, api_key=self._api_key, base_url=self._base_url)

    @api_keys_required("OPENAI_API_KEY")
    def embed_list(
        self,
        objs: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        r"""Generates embeddings for the given texts.

        Args:
            objs (list[str]): The texts for which to generate the embeddings.
            **kwargs (Any): Extra kwargs passed to the embedding API.

        Returns:
            list[list[float]]: A list that represents the generated embedding
                as a list of floating-point numbers.
        """

        # TODO: count tokens
        if self.model_type == EmbeddingModelType.TEXT_EMBEDDING_ADA_2:
            response = self.client.embeddings.create(
                input=objs,
                model=self.model_type.value,
                **kwargs,
            )
        else:
            response = self.client.embeddings.create(
                input=objs,
                model=self.model_type.value,
                dimensions=self.output_dim,
                **kwargs,
            )
        return [data.embedding for data in response.data]

    def get_output_dim(self) -> int:
        r"""Returns the output dimension of the embeddings.

        Returns:
            int: The dimensionality of the embedding for the current model.
        """
        return self.output_dim
