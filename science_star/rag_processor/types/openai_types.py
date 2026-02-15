# isort: skip_file
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.chat import ParsedChatCompletion
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessageToolCall

Choice = Choice
ChatCompletion = ChatCompletion
ChatCompletionChunk = ChatCompletionChunk
ChatCompletionMessage = ChatCompletionMessage
ChatCompletionMessageParam = ChatCompletionMessageParam
ChatCompletionSystemMessageParam = ChatCompletionSystemMessageParam
ChatCompletionUserMessageParam = ChatCompletionUserMessageParam
ChatCompletionAssistantMessageParam = ChatCompletionAssistantMessageParam
ChatCompletionToolMessageParam = ChatCompletionToolMessageParam
ChatCompletionMessageToolCall = ChatCompletionMessageToolCall
CompletionUsage = CompletionUsage
NOT_GIVEN = NOT_GIVEN
NotGiven = NotGiven
ParsedChatCompletion = ParsedChatCompletion