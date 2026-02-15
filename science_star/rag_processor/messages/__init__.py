from typing import Union

from rag_processor.types import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from .conversion import (
    AlpacaItem,
    HermesFunctionFormatter,
    ShareGPTMessage,
)
from .conversion.conversation_models import (
    ShareGPTConversation,
)
from .conversion.sharegpt.function_call_formatter import (
    FunctionCallFormatter,
)

OpenAISystemMessage = ChatCompletionSystemMessageParam
OpenAIAssistantMessage = Union[
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
]
OpenAIUserMessage = ChatCompletionUserMessageParam
OpenAIToolMessageParam = ChatCompletionToolMessageParam

OpenAIMessage = ChatCompletionMessageParam


from .base import BaseMessage  # noqa: E402
from .func_message import FunctionCallingMessage  # noqa: E402

__all__ = [
    'OpenAISystemMessage',
    'OpenAIAssistantMessage',
    'OpenAIUserMessage',
    'OpenAIToolMessageParam',
    'OpenAIMessage',
    'FunctionCallFormatter',
    'HermesFunctionFormatter',
    'ShareGPTConversation',
    'ShareGPTMessage',
    'BaseMessage',
    'FunctionCallingMessage',
    'AlpacaItem',
]