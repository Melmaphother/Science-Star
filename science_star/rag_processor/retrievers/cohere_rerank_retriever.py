import os
from typing import Any, Dict, List, Optional

from rag_processor.retrievers import BaseRetriever
from rag_processor.utils import dependencies_required

DEFAULT_TOP_K_RESULTS = 1


class CohereRerankRetriever(BaseRetriever):
    r"""An implementation of the `BaseRetriever` using the `Cohere Re-ranking`
    model.

    Attributes:
        model_name (str): The model name to use for re-ranking.
        api_key (Optional[str]): The API key for authenticating with the
            Cohere service.

    References:
        https://txt.cohere.com/rerank/
    """

    @dependencies_required('cohere')
    def __init__(
        self,
        model_name: str = "rerank-multilingual-v2.0",
        api_key: Optional[str] = None,
    ) -> None:
        r"""Initializes an instance of the CohereRerankRetriever. This
        constructor sets up a client for interacting with the Cohere API using
        the specified model name and API key. If the API key is not provided,
        it attempts to retrieve it from the COHERE_API_KEY environment
        variable.

        Args:
            model_name (str): The name of the model to be used for re-ranking.
                Defaults to 'rerank-multilingual-v2.0'.
            api_key (Optional[str]): The API key for authenticating requests
                to the Cohere API. If not provided, the method will attempt to
                retrieve the key from the environment variable
                'COHERE_API_KEY'.

        Raises:
            ImportError: If the 'cohere' package is not installed.
            ValueError: If the API key is neither passed as an argument nor
                set in the environment variable.
        """
        import cohere

        try:
            self.api_key = api_key or os.environ["COHERE_API_KEY"]
        except ValueError as e:
            raise ValueError(
                "Must pass in cohere api key or specify via COHERE_API_KEY"
                " environment variable."
            ) from e

        self.co = cohere.Client(self.api_key)
        self.model_name = model_name

    def query(
        self,
        query: str,
        retrieved_result: List[Dict[str, Any]],
        top_k: int = DEFAULT_TOP_K_RESULTS,
    ) -> List[Dict[str, Any]]:
        r"""Queries and compiles results using the Cohere re-ranking model.

        Args:
            query (str): Query string for information retriever.
            retrieved_result (List[Dict[str, Any]]): The content to be
                re-ranked, should be the output from `BaseRetriever` like
                `VectorRetriever`.
            top_k (int, optional): The number of top results to return during
                retriever. Must be a positive integer. Defaults to
                `DEFAULT_TOP_K_RESULTS`.

        Returns:
            List[Dict[str, Any]]: Concatenated list of the query results.
        """
        rerank_results = self.co.rerank(
            query=query,
            documents=retrieved_result,
            top_n=top_k,
            model=self.model_name,
        )
        formatted_results = []
        for result in rerank_results.results:
            selected_chunk = retrieved_result[result.index]
            selected_chunk['similarity score'] = result.relevance_score
            formatted_results.append(selected_chunk)
        return formatted_results
