from __future__ import annotations

import os
from typing import Any, Optional

from openai import OpenAI

from rag_processor.embeddings.base import BaseEmbedding
from rag_processor.utils import api_keys_required


class OpenAICompatibleEmbedding(BaseEmbedding[str]):
    r"""Provides text embedding functionalities supporting OpenAI
    compatibility.

    Args:
        model_type (str): The model type to be used for text embeddings.
        api_key (str): The API key for authenticating with the model service.
        url (str): The url to the model service.
    """

    def __init__(
        self,
        model_type: str,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        self.model_type = model_type
        self.output_dim: Optional[int] = None

        self._api_key = api_key or os.environ.get(
            "OPENAI_COMPATIBILIY_API_KEY"
        )
        self._url = url or os.environ.get("OPENAI_COMPATIBILIY_API_BASE_URL")
        self._client = OpenAI(
            timeout=60,
            max_retries=3,
            api_key=self._api_key,
            base_url=self._url,
        )

    @api_keys_required("OPENAI_COMPATIBILIY_API_KEY")
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

        response = self._client.embeddings.create(
            input=objs,
            model=self.model_type,
            **kwargs,
        )
        self.output_dim = len(response.data[0].embedding)
        return [data.embedding for data in response.data]

    def get_output_dim(self) -> int:
        r"""Returns the output dimension of the embeddings.

        Returns:
            int: The dimensionality of the embedding for the current model.
        """
        if self.output_dim is None:
            raise ValueError(
                "Output dimension is not yet determined. Call "
                "'embed_list' first."
            )
        return self.output_dim
