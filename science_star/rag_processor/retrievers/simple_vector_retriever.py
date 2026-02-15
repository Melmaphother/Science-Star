from typing import IO, TYPE_CHECKING, Any, Dict, List, Optional, Union

from rag_processor.embeddings import BaseEmbedding, OpenAIEmbedding
from rag_processor.retrievers.base import BaseRetriever

from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

from rag_processor.storages import QdrantStorage


class SimpleVectorRetriever():
    r"""An implementation of the `BaseRetriever` by using vector storage and
    embedding model.

    This class facilitates the retriever of relevant information using a
    query-based approach, backed by vector embeddings.

    Attributes:
        embedding_model (BaseEmbedding): Embedding model used to generate
            vector embeddings.
        storage (BaseVectorStorage): Vector storage to query.
        unstructured_modules (UnstructuredIO): A module for parsing files and
            URLs and chunking content based on specified parameters.
    """

    def __init__(
        self,
        embedding_model: Optional[BaseEmbedding] = None,
        text_splitter: Optional[RecursiveCharacterTextSplitter] = None,
        chunk_size: int=500,
        chunk_overlap: int=50,
        path : str=None
    ) -> None:
        r"""Initializes the retriever class with an optional embedding model.

        Args:
            embedding_model (Optional[BaseEmbedding]): The embedding model
                instance. Defaults to `OpenAIEmbedding` if not provided.
            storage (BaseVectorStorage): Vector storage to query.
        """
        self.embedding_model = embedding_model or OpenAIEmbedding()

        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        self.storage = None if path is None else \
                        QdrantStorage(vector_dim=self.embedding_model.get_output_dim(),
                                      collection_name="rag_cache",
                                      path = path)


    def _split_text(self, contents: str | List[str]):
        if type(contents) == str:
            chunks=self.text_splitter.split_text(contents)
        elif type(contents) == List:
            chunks = [self.text_splitter.split_text(content) for content in contents]
        else:
            print(f"[Warning] from text_splitter: Input type {type(contents)} error! Expected type to be string or List[string]")
        return chunks
    
    def retrieve(
            self,
            query,
            contents,
            limit: int=10,
            threshold: float=0.3,
            embed_batch: int=50
    ):
        '''
        @inputs
            query: str
            contents: list[str] => raw contents
            limit: int => max number of contents considered
            threshold: float => similaraty threshold of retrieval
        @return
            str: content
        '''
        # embed query
        query_vector = self.embedding_model.embed(obj=query)

        # chunk contents
        chunks = self._split_text(contents)

        # embed chunk
        vectors = []
        # Process chunks in batches and store embeddings
        for i in range(0, len(chunks), embed_batch):
            batch_chunks = chunks[i : i + embed_batch]
            batch_vectors = self.embedding_model.embed_list(
                objs=[str(chunk) for chunk in batch_chunks]
            )
            vectors += batch_vectors

        # compute distance
        results = []
        for vector, chunk in zip(vectors, chunks):
            similarity_score = np.dot(np.array(query_vector), np.array(vector))
            if similarity_score > threshold:
                results.append((similarity_score, chunk))

        # no relative contents
        if len(results) == 0:
            return "Not enough relative contents! Maybe we should search more information from other source for the task"
        # rank
        results.sort(key=lambda x: x[0], reverse=True)

        # prepare output content
        output_content = ''
        for i in range(len(results[:limit])):
            output_content += f"Relative content {i+1}:\n{results[i][1]}\n"

        # action_prompt = ""
        return output_content
