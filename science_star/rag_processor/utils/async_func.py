import asyncio
from copy import deepcopy

from camel.toolkits import FunctionTool


def sync_funcs_to_async(funcs: list[FunctionTool]) -> list[FunctionTool]:
    r"""Convert a list of Python synchronous functions to Python
    asynchronous functions.

    Args:
        funcs (list[FunctionTool]): List of Python synchronous
            functions in the :obj:`FunctionTool` format.

    Returns:
        list[FunctionTool]: List of Python asynchronous functions
            in the :obj:`FunctionTool` format.
    """
    async_funcs = []
    for func in funcs:
        sync_func = func.func

        def async_callable(*args, **kwargs):
            return asyncio.to_thread(sync_func, *args, **kwargs)  # noqa: B023

        async_funcs.append(
            FunctionTool(async_callable, deepcopy(func.openai_tool_schema))
        )
    return async_funcs
