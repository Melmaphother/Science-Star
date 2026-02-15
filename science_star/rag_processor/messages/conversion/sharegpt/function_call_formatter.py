from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from rag_processor.messages.conversion import (
    ToolCall,
    ToolResponse,
)

CallT = TypeVar('CallT', bound=ToolCall, covariant=True)
ResponseT = TypeVar('ResponseT', bound=ToolResponse, covariant=True)


class FunctionCallFormatter(ABC, Generic[CallT, ResponseT]):
    r"""Abstract base class for function calling formats"""

    @abstractmethod
    def extract_tool_calls(self, message: str) -> List[CallT]:
        r"""Extract function call info from a message string"""
        pass

    @abstractmethod
    def extract_tool_response(self, message: str) -> Optional[ResponseT]:
        r"""Extract function response info from a message string"""
        pass

    @abstractmethod
    def format_tool_call(
        self, content: str, func_name: str, args: Dict[str, Any]
    ) -> str:
        r"""Format a function call into a message string"""
        pass

    @abstractmethod
    def format_tool_response(self, func_name: str, result: Any) -> str:
        r"""Format a function response into a message string"""
        pass
