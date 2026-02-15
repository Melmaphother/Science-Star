from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseGraphStorage(ABC):
    r"""An abstract base class for graph storage systems."""

    @property
    @abstractmethod
    def get_client(self) -> Any:
        r"""Get the underlying graph storage client."""
        pass

    @property
    @abstractmethod
    def get_schema(self) -> str:
        r"""Get the schema of the graph storage"""
        pass

    @property
    @abstractmethod
    def get_structured_schema(self) -> Dict[str, Any]:
        r"""Get the structured schema of the graph storage"""
        pass

    @abstractmethod
    def refresh_schema(self) -> None:
        r"""Refreshes the graph schema information."""
        pass

    @abstractmethod
    def add_triplet(self, subj: str, obj: str, rel: str) -> None:
        r"""Adds a relationship (triplet) between two entities in the database.

        Args:
            subj (str): The identifier for the subject entity.
            obj (str): The identifier for the object entity.
            rel (str): The relationship between the subject and object.
        """
        pass

    @abstractmethod
    def delete_triplet(self, subj: str, obj: str, rel: str) -> None:
        r"""Deletes a specific triplet from the graph, comprising a subject,
        object and relationship.

        Args:
            subj (str): The identifier for the subject entity.
            obj (str): The identifier for the object entity.
            rel (str): The relationship between the subject and object.
        """
        pass

    @abstractmethod
    def query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        r"""Query the graph store with statement and parameters.

        Args:
            query (str): The query to be executed.
            params (Optional[Dict[str, Any]]): A dictionary of parameters to
                be used in the query. Defaults to `None`.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each
                dictionary represents a row of results from the query.
        """
        pass
