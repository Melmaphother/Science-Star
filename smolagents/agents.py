#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this file are modifications by OPPO PersonalAI Team.
# Licensed under the Apache License, Version 2.0.

import importlib
import inspect
import json
import os
import copy
import re
import tempfile
import textwrap
import time
from collections import deque
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypedDict, Union
import heapq
from collections import deque
import jinja2
import yaml
from huggingface_hub import create_repo, metadata_update, snapshot_download, upload_folder
from jinja2 import StrictUndefined, Template
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from .workflow import Workflow
from .verify_function import evaluate_answer
from uuid import uuid4
from .reformulator import prepare_response
from collections import Counter 
import random
import requests
from .agent_types import AgentAudio, AgentImage, AgentType, handle_agent_output_types
from .default_tools import TOOL_MAPPING, FinalAnswerTool
from .e2b_executor import E2BExecutor
from .local_python_executor import (
    BASE_BUILTIN_MODULES,
    LocalPythonInterpreter,
    fix_final_answer_code,
)
from .memory import Message
from .memory import ActionStep, AgentMemory, PlanningStep, SystemPromptStep, TaskStep, ToolCall
from .models import (
    ChatMessage,
    MessageRole,
    Model,
)
from .monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
    Monitor,
)
from .tools import Tool
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    make_init_file,
    parse_code_blobs,
    parse_json_tool_call,
    truncate_content,
)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
logger = getLogger(__name__)

def take_a_breath():
    pass

def populate_template(template: str, variables: Dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        initial_facts (`str`): Initial facts prompt.
        initial_plan (`str`): Initial plan prompt.
        update_facts_pre_messages (`str`): Update facts pre-messages prompt.
        update_facts_post_messages (`str`): Update facts post-messages prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
    """

    initial_facts: str
    initial_plan: str
    update_facts_pre_messages: str
    update_facts_post_messages: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class ManagedAgentPromptTemplate(TypedDict):
    """
    Prompt templates for the managed agent.

    Args:
        task (`str`): Task prompt.
        report (`str`): Report prompt.
    """

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(
        initial_facts="",
        initial_plan="",
        update_facts_pre_messages="",
        update_facts_post_messages="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


class MultiStepAgent:
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        max_steps (`int`, default `6`): Maximum number of steps the agent can take to solve the task.
        tool_parser (`Callable`, *optional*): Function used to parse the tool calls from the LLM output.
        add_base_tools (`bool`, default `False`): Whether to add the base tools to the agent's tools.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list[Callable]`, *optional*): Callbacks that will be called at each step.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
        static_plan (`bool`, *optional*): Whether to use static plan.
        dynamic_update_plan (`bool`, *optional*): Whether to use dynamic update plan.
        agent_kb (`bool`, *optional*): Whether to provide knowledge retrieval from agent kb.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
        final_answer_checks (`list`, *optional*): List of Callables to run before returning a final answer for checking validity.
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[PromptTemplates] = None,
        max_steps: int = 6,
        tool_parser: Optional[Callable] = None,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        grammar: Optional[Dict[str, str]] = None,
        managed_agents: Optional[List] = None,
        step_callbacks: Optional[List[Callable]] = None,
        planning_interval: Optional[int] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        agent_type: Optional[str] = "code_agent",
        reflection: bool = False,
        reflection_threshold: int = -1,
        verify_type: str = 'list-wise',
        result_merging_type: str='list-wise',
        provide_run_summary: bool = False,
        final_answer_checks: Optional[List[Callable]] = None,
        debug: bool = False,
        subtask: bool = False,
        static_plan: bool = False,
        dynamic_update_plan: bool = False,
        n_rollouts: Optional[int] = 1,
        summary: bool = False,
        agent_kb: bool = False,
        top_k: Optional[int] = 3,
        retrieval_type: Optional[str] = "hybrid",
    ):
        if tool_parser is None:
            tool_parser = parse_json_tool_call
        self.agent_name = self.__class__.__name__
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        self.max_steps = max_steps
        self.step_number: int = 0
        self.tool_parser = tool_parser
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.state = {}
        self.name = name
        self.description = description
        self.agent_type = agent_type
        self.reflection = reflection
        self.provide_run_summary = provide_run_summary
        self.debug = debug
        self.action_trajectory=[]
        self.managed_agents = {}
        if managed_agents is not None:
            for managed_agent in managed_agents:
                assert managed_agent.name and managed_agent.description, (
                    "All managed agents need both a name and a description!"
                )
            self.managed_agents = {agent.name: agent for agent in managed_agents}

        tool_and_managed_agent_names = [tool.name for tool in tools]
        if managed_agents is not None:
            tool_and_managed_agent_names += [agent.name for agent in managed_agents]
        if len(tool_and_managed_agent_names) != len(set(tool_and_managed_agent_names)):
            raise ValueError(
                "Each tool or managed_agent should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_and_managed_agent_names if tool_and_managed_agent_names.count(name) > 1]}"
            )

        for tool in tools:
            assert isinstance(tool, Tool), f"This element is not of class Tool: {str(tool)}"
        self.tools = {tool.name: tool for tool in tools}

        if add_base_tools:
            for tool_name, tool_class in TOOL_MAPPING.items():
                if tool_name != "python_interpreter" or self.__class__.__name__ == "ToolCallingAgent":
                    self.tools[tool_name] = tool_class()
        self.tools["final_answer"] = FinalAnswerTool()

        self.system_prompt = self.initialize_system_prompt()
        self.input_messages = None
        self.task = None
        self.memory = AgentMemory(self.system_prompt)
        self.logger = AgentLogger(level=verbosity_level)
        self.monitor = Monitor(self.model, self.logger)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.step_callbacks.append(self.monitor.update_metrics)
        self.final_answer_checks = final_answer_checks
        # plan args
        self.subtask = subtask
        self.static_plan = static_plan
        self.dynamic_update_plan = dynamic_update_plan
        self.workflow = None
        # tts 
        self.n_rollouts = n_rollouts
        self.reflection_threshold = reflection_threshold
        self.verify_type = verify_type
        self.result_merging_type = result_merging_type,
        # tts prompts
        self.ORM_prompt = ""
        self.PRM_prompt = ""
        self.LIST_WISE_prompt = ""
        self.REFLECTION_prompt = ""
        self.BASE_ADDITIONAL_PROMPT = ""
        self.ORM_list_wise_prompt = ""
        self.PRM_list_wise_prompt = ""
        # memory
        self.summary=summary
        # agent kb args
        self.agent_kb = agent_kb
        self.top_k = top_k
        self.retrieval_type = retrieval_type
        self._load_prompts()

    @property
    def logs(self):
        logger.warning(
            "The 'logs' attribute is deprecated and will soon be removed. Please use 'self.memory.steps' instead."
        )
        return [self.memory.system_prompt] + self.memory.steps
    
    def _load_prompt_from_package(self, file_name: str) -> str:
        try:
            path = importlib.resources.files(f"smolagents.prompts").joinpath(file_name)
            content = path.read_text()
            data = yaml.safe_load(content)
            return data.get("prompt", "")
        except Exception as e:
            self.logger.log(f"Error loading prompt from {file_name}: {e}")
            return ""

    def _load_prompts(self):
        try:
            self.ORM_prompt = self._load_prompt_from_package("ORM.yaml")
            self.PRM_prompt = self._load_prompt_from_package("PRM.yaml")
            self.LIST_WISE_prompt = self._load_prompt_from_package("list_wise.yaml")
            self.REFLECTION_prompt = self._load_prompt_from_package("single_node_reflection.yaml")
            self.ORM_list_wise_prompt=self._load_prompt_from_package("ORM_list_wise.yaml")
            self.PRM_list_wise_prompt=self._load_prompt_from_package("PRM_list_wise.yaml")

            self.BASE_ADDITIONAL_PROMPT = (
                "You will now receive an additional prompt, which summarizes the experience of the previous step in the trajectory."
                "This summary includes successes, errors, and lessons learned from the prior attempt."
                "Carefully review this prompt to gain insights from the previous step."
                "Your goal is to build on prior knowledge, avoiding previously identified mistakes and unnecessary repetitions. "
                "Aim to explore new, more efficient strategies for solving the task."
            )
        except Exception as e:
            self.logger.log(f"Unexpected error during prompt initialization: {e}")
            self.ORM_prompt = ""
            self.PRM_prompt = ""
            self.LIST_WISE_prompt = ""
            self.REFLECTION_prompt = ""

    def initialize_system_prompt(self):
        """To be implemented in child classes"""
        pass

    def write_memory_to_messages(
        self,
        memory_steps: Optional[List[ActionStep]]=None,
        summary_mode: Optional[bool] = False,
    ) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode, summary=self.summary)
        for memory_step in memory_steps if memory_steps else self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode, summary=self.summary))
        return messages

    def visualize(self):
        """Creates a rich tree visualization of the agent's structure."""
        self.logger.visualize_agent_tree(self)

    def extract_action(self, model_output: str, split_token: str) -> Tuple[str, str]:
        """
        Parse action from the LLM output

        Args:
            model_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = model_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str, images: Optional[list[str]]) -> str:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`, *optional*): Paths to image(s).

        Returns:
            `str`: Final answer to the task.
        """
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            }
        ]
        if images:
            messages[0]["content"].append({"type": "image"})
        messages += self.write_memory_to_messages()[1:]
        messages += [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            }
        ]
        try:
            chat_message: ChatMessage = self.model(messages)
            return chat_message.content
        except Exception as e:
            return f"Error in generating final LLM output:\n{e}"

    def execute_tool_call(self, tool_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.tools).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            error_msg = f"Unknown tool {tool_name}, should be instead one of {list(available_tools.keys())}."
            raise AgentExecutionError(error_msg, self.logger)

        try:
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                    for key, value in arguments.items():
                        if isinstance(value, str) and value in self.state:
                            arguments[key] = self.state[value]
                    if tool_name in self.managed_agents:
                        observation = available_tools[tool_name].__call__(**arguments)
                    else:
                        observation = available_tools[tool_name].__call__(**arguments, sanitize_inputs_outputs=True)
                except json.JSONDecodeError:
                    if tool_name in self.managed_agents:
                        observation = available_tools[tool_name].__call__(arguments)
                    else:
                        observation = available_tools[tool_name].__call__(arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(**arguments)
                else:
                    observation = available_tools[tool_name].__call__(**arguments, sanitize_inputs_outputs=True)
            else:
                error_msg = f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                raise AgentExecutionError(error_msg, self.logger)
            return observation
        except Exception as e:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                error_msg = (
                    f"Error when executing tool {tool_name} with arguments {arguments}: {type(e).__name__}: {e}\nYou should only use this tool with a correct input.\n"
                    f"As a reminder, this tool's description is the following: '{tool.description}'.\nIt takes inputs: {tool.inputs} and returns output type {tool.output_type}"
                )
                raise AgentExecutionError(error_msg, self.logger)
            elif tool_name in self.managed_agents:
                error_msg = (
                    f"Error in calling team member: {e}\nYou should only ask this team member with a correct request.\n"
                    f"As a reminder, this team member's description is the following:\n{available_tools[tool_name]}"
                )
                raise AgentExecutionError(error_msg, self.logger)

    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """To be implemented in children classes. Should return either None if the step is not final."""
        pass

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: Optional[List[str]] = None,
        additional_args: Optional[Dict] = None,
        additional_knowledge: Optional[str] = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in a streaming way.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[str]`, *optional*): Paths to image(s).
            additional_args (`dict`): Any other variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!

        Example:
        ```py
        from smolagents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """

        self.task = task
        if additional_args is not None:
            self.state.update(additional_args)
            self.task += f"""
                You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
                {str(additional_args)}."""

        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )

        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if stream:
            return self._run(task=self.task, images=images)
        return deque(self._run(task=self.task, images=images, additional_knowledge=additional_knowledge), maxlen=1)[0]
    
    def _run(self, task: str, images: List[str] | None = None, additional_knowledge: Optional[str] = None) -> Generator[ActionStep | AgentType, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """
        pass

    def reflect_planing(self,task,answer_message,evaluate_thought):
        memory_messages = self.write_memory_to_messages()[1:]

        # Redact updated facts
        facts_update_pre_messages = {
            "role": MessageRole.SYSTEM,
            "content": [{"type": "text", "text": self.prompt_templates["planning"]["update_facts_pre_messages"]}],
        }
        facts_update_post_messages = {
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": self.prompt_templates["planning"]["update_facts_post_messages"]}],
        }
        input_messages = [facts_update_pre_messages] + memory_messages + [facts_update_post_messages]
        chat_message_facts: ChatMessage = self.model(input_messages)
        facts_update = chat_message_facts.content
        update_plan_pre_messages = {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["reflection_plan_pre_messages"], variables={
                            "task": task,
                            "tools": self.tools,
                            "managed_agents": self.managed_agents,
                            "trajectory":answer_message,
                            "analysis":evaluate_thought
                        },
                        ),
                    }
                ],
            }
        chat_message_plan: ChatMessage = self.model(
            [update_plan_pre_messages],
            stop_sequences=["<end_plan>"],
        )

       # Log final facts and plan
        final_plan_redaction = textwrap.dedent(
            f"""I still need to solve the task I was given:
            ```
            {task}
            ```

            Here is my new/updated plan of action to solve the task:
            ```
            {chat_message_plan.content}
            ```"""
        )

        final_facts_redaction = textwrap.dedent(
            f"""Here is the updated list of the facts that I know:
            ```
            {facts_update}
            ```"""
        )
        reflection_step=PlanningStep(
                model_input_messages=input_messages,
                plan=final_plan_redaction,
                facts=final_facts_redaction,
                model_output_message_plan=chat_message_plan,
                model_output_message_facts=chat_message_facts,
            )
        self.logger.log(
            Rule("[bold]Updated plan", style="orange"),
            Text(final_plan_redaction),
            level=LogLevel.INFO,
        )
        return  reflection_step

    def planning_step(self, task, is_first_step: bool, step: int, additional_knowledge: Optional[str] = None) -> None:
        """
        Used periodically by the agent to plan the next steps to reach the objective.
        Args:
            task (`str`): Task to perform.
            is_first_step (`bool`): If this step is not the first one, the plan should be an update over a previous plan.
            step (`int`): The number of the current step, used as an indication for the LLM.
        """
        if self.static_plan:
            return self._get_static_plan()
        
        if self.dynamic_update_plan:
            return self._handle_dynamic_plan(task, is_first_step, step, additional_knowledge)

        if is_first_step:
            input_messages, answer_facts = self._initial_fact_generation(task)
            answer_plan, final_plan_redaction = self._generate_initial_plan(
                task, answer_facts, additional_knowledge
            )
        else:
            input_messages, answer_facts = self._update_fact_generation()
            answer_plan, final_plan_redaction = self._generate_updated_plan(task, answer_facts, step)

        final_facts_redaction = textwrap.dedent(
            f"""Here are the facts that I know so far:
            ```
            {answer_facts}
            ```""".strip()
        )

        # Log results
        self.logger.log(
            Rule("[bold]Initial plan" if is_first_step else "[bold]Updated plan", style="orange"),
            Text(final_plan_redaction),
            level=LogLevel.INFO,
        )

        return PlanningStep(
            model_input_messages=input_messages,
            plan=final_plan_redaction,
            facts=final_facts_redaction,
            model_output_message_plan=answer_plan,
            model_output_message_facts=answer_facts,
        )

    def _dynamic_initial_plan(self, task, additional_knowledge):
        # Fact generation
        input_messages = [self._create_message(MessageRole.USER, self.prompt_templates["planning"]["initial_facts"], task)]
        chat_message_facts = self.model(input_messages)
        
        # Plan generation
        plan_template = self._prepare_plan_template(task, chat_message_facts.content, additional_knowledge)
        message_prompt_plan = self._create_message(MessageRole.USER, plan_template)
        chat_message_plan = self.model([message_prompt_plan], stop_sequences=["<end_plan>"])
        
        # Initialize workflow
        self.workflow = Workflow(chat_message_plan.content)
        
        return self._create_dynamic_output(
            input_messages, 
            chat_message_facts, 
            chat_message_plan, 
            is_first_step=True
        )

    def _dynamic_updated_plan(self, task, step):
        # Fact update
        input_messages, facts_update = self._generate_updated_facts()
        
        # Plan update
        update_messages = self._prepare_update_messages(task, step, facts_update)
        chat_message_plan = self.model(update_messages, stop_sequences=["<end_plan>"])
        
        # Apply workflow update
        self.workflow.apply_update(chat_message_plan.content)
        
        return self._create_dynamic_output(
            input_messages, 
            ChatMessage(role="assistant", content=facts_update), 
            chat_message_plan, 
            is_first_step=False
        )

    def _prepare_plan_template(self, task, facts, additional_knowledge):
        if self.agent_kb and additional_knowledge:
            return populate_template(
                self.prompt_templates["planning"]["initial_plan_with_knowledge"],
                variables={
                    "task": task,
                    "tools": self.tools,
                    "managed_agents": self.managed_agents,
                    "answer_facts": facts,
                    "knowledge_data": additional_knowledge,
                },
            )
        return populate_template(
            self.prompt_templates["planning"]["initial_plan"],
            variables={
                "task": task,
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "answer_facts": facts,
            },
        )
    
    def _generate_updated_facts(self):
        memory_messages = self.write_memory_to_messages()[1:]
        pre_message = self._create_message(MessageRole.SYSTEM, self.prompt_templates["planning"]["update_facts_pre_messages"])
        post_message = self._create_message(MessageRole.USER, self.prompt_templates["planning"]["update_facts_post_messages"])
        input_messages = [pre_message] + memory_messages + [post_message]
        return input_messages, self.model(input_messages).content

    def _prepare_update_messages(self, task, step, facts_update):
        memory_messages = self.write_memory_to_messages()[1:]
        pre_message = self._create_message(
            MessageRole.SYSTEM,
            self.prompt_templates["planning"]["update_plan_pre_messages"],
            task
        )
        post_message = self._create_message(
            MessageRole.USER,
            self.prompt_templates["planning"]["update_plan_post_messages"],
            variables={
                "task": task,
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "facts_update": facts_update,
                "remaining_steps": (self.max_steps - step),
            }
        )
        return [pre_message] + memory_messages + [post_message]

    def _create_dynamic_output(self, input_messages, facts_msg, plan_msg, is_first_step):
        plan_content = self.workflow.__str__() if not is_first_step else plan_msg.content
        final_plan = self._format_plan_output(plan_content, is_first_step)
        final_facts = self._format_facts_output(facts_msg.content, is_first_step)
        
        self.logger.log(
            Rule(f"[bold]{'Initial' if is_first_step else 'Updated'} plan", style="orange"),
            Text(final_plan),
            level=LogLevel.INFO,
        )
        
        return PlanningStep(
            model_input_messages=input_messages,
            plan=final_plan,
            facts=final_facts,
            model_output_message_plan=plan_msg,
            model_output_message_facts=facts_msg,
        )

    def _format_plan_output(self, content, is_first_step):
        template = """Here is the plan of action that I will follow to solve the task:
        ```
        {content}
        ```""" if is_first_step else """I still need to solve the task I was given:
        ```
        {task}
        ```
        
        Here is my new/updated plan of action to solve the task:
        ```
        {content}
        ```"""
        return textwrap.dedent(template.format(content=content))

    def _format_facts_output(self, content, is_first_step):
        template = "Here are the facts that I know so far:" if is_first_step else "Here is the updated list of the facts that I know:"
        return textwrap.dedent(f"""{template}
        ```
        {content}
        ```""".strip())

    def _create_message(self, role, template, variables=None):
        return {
            "role": role,
            "content": [{
                "type": "text",
                "text": populate_template(template, variables=variables if isinstance(variables, dict) else {"task": variables})
            }]
        }

    def _initial_fact_generation(self, task):
        input_messages = [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["initial_facts"], variables={"task": task}
                        ),
                    }
                ],
            },
        ]
        chat_message_facts: ChatMessage = self.model(input_messages)
        return input_messages, chat_message_facts.content

    def _get_static_plan(self):
        static_plan = textwrap.dedent(f"""Here is the plan of action that I will follow to solve the task:
        ```
        1. If needed, use the search tool to find relevant information.
        2. To inspect the information from search tool, use the proper agent tool.
        3. Execute domain-specific processing with all information you have, such as mathematical calculations, statistical analysis or logical reasoning.    
        4. Format final output, remember to follow the commanded format.
        ```""")

        memory_messages = self.write_memory_to_messages()

        model_output_message_plan = ChatMessage(
            role="assistant",
            content="",
            tool_calls=None,
        )

        model_output_message_facts = ChatMessage(
            role="assistant",
            content="",
            tool_calls=None,
        )

        return PlanningStep(
            model_input_messages=memory_messages,
            plan=static_plan,
            facts="",
            model_output_message_plan=model_output_message_plan,
            model_output_message_facts=model_output_message_facts,
        )

    def _handle_dynamic_plan(self, task, is_first_step, step, additional_knowledge):
        if is_first_step:
            return self._dynamic_initial_plan(task, additional_knowledge)
        return self._dynamic_updated_plan(task, step)
    
    
    def _generate_initial_plan(self, task, answer_facts, additional_knowledge):

        if self.agent_kb and additional_knowledge:
            knowledge_data_all = "Please strictly follow the suggestions below:\n" + additional_knowledge
            final_facts_knowledge = textwrap.dedent(
                f"""Here are the similar tasks, plans and relevant experience that I should follow:
                ```
                {knowledge_data_all}
                ```""".strip()
            )
            initial_plan_template = populate_template(
                self.prompt_templates["planning"]["initial_plan_with_knowledge"],
                variables={
                    "task": task,
                    "tools": self.tools,
                    "managed_agents": self.managed_agents,
                    "answer_facts": answer_facts,
                    "knowledge_data": additional_knowledge,
                },
            )
            self.logger.log(
                Rule("[bold]retrieved task and plan", style="orange"),
                Text(final_facts_knowledge),
                level=LogLevel.INFO,
            )
        elif self.subtask:
            initial_plan_template = populate_template(
                self.prompt_templates["planning"]["initial_plan_with_subtask"],
                variables={
                    "task": task,
                    "tools": self.tools,
                    "managed_agents": self.managed_agents,
                    "answer_facts": answer_facts,
                },
            )
        else:
            initial_plan_template = populate_template(
                self.prompt_templates["planning"]["initial_plan"],
                variables={
                    "task": task,
                    "tools": self.tools,
                    "managed_agents": self.managed_agents,
                    "answer_facts": answer_facts,
                },
            )

        message_prompt_plan = {
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": initial_plan_template}],
        }

        chat_message_plan: ChatMessage = self.model([message_prompt_plan], stop_sequences=["<end_plan>"])
        final_plan_redaction = textwrap.dedent(
            f"""Here is the plan of action that I will follow to solve the task:
            ```
            {chat_message_plan.content}
            ```"""
        )
        return chat_message_plan, final_plan_redaction


    def _update_fact_generation(self):
        memory_messages = self.write_memory_to_messages()[1:]

        facts_update_pre_messages = {
            "role": MessageRole.SYSTEM,
            "content": [{"type": "text", "text": self.prompt_templates["planning"]["update_facts_pre_messages"]}],
        }
        facts_update_post_messages = {
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": self.prompt_templates["planning"]["update_facts_post_messages"]}],
        }

        input_messages = [facts_update_pre_messages] + memory_messages + [facts_update_post_messages]
        chat_message_facts: ChatMessage = self.model(input_messages)
        return input_messages, chat_message_facts.content


    def _generate_updated_plan(self, task, answer_facts, step):
        memory_messages = self.write_memory_to_messages()[1:]

        update_plan_pre_messages = {
            "role": MessageRole.SYSTEM,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                    ),
                }
            ],
        }

        if self.subtask:
            update_plan_post_messages = {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_post_messages_with_subtask"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "facts_update": answer_facts,
                                "remaining_steps": (self.max_steps - step),
                            },
                        ),
                    }
                ],
            }
        else:
            update_plan_post_messages = {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_post_messages"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "facts_update": answer_facts,
                                "remaining_steps": (self.max_steps - step),
                            },
                        ),
                    }
                ],
            }

        chat_message_plan: ChatMessage = self.model(
            [update_plan_pre_messages] + memory_messages + [update_plan_post_messages],
            stop_sequences=["<end_plan>"],
        )

        final_plan_redaction = textwrap.dedent(
            f"""I still need to solve the task I was given:
            ```
            {task}
            ```

            Here is my new/updated plan of action to solve the task:
            ```
            {chat_message_plan.content}
            ```"""
        )

        return chat_message_plan, final_plan_redaction

    def replay(self, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        self.memory.replay(self.logger, detailed=detailed)

    def __call__(self, task: str, **kwargs):
        """Adds additional prompting for the managed agent, runs it, and wraps the output.

        This method is called only by a managed agent.
        """
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=task),
        )
        report = self.run(full_task, **kwargs)
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"], variables=dict(name=self.name, final_answer=report)
        )
        if self.provide_run_summary:
            answer += "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n"
            for message in self.write_memory_to_messages(summary_mode=True):
                content = message["content"]
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += "\n</summary_of_work>"
        return answer

    def save(self, output_dir: str, relative_path: Optional[str] = None):
        """
        Saves the relevant code files for your agent. This will copy the code of your agent in `output_dir` as well as autogenerate:

        - a `tools` folder containing the logic for each of the tools under `tools/{tool_name}.py`.
        - a `managed_agents` folder containing the logic for each of the managed agents.
        - an `agent.json` file containing a dictionary representing your agent.
        - a `prompt.yaml` file containing the prompt templates used by your agent.
        - an `app.py` file providing a UI for your agent when it is exported to a Space with `agent.push_to_hub()`
        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str`): The folder in which you want to save your tool.
        """
        make_init_file(output_dir)

        if self.managed_agents:
            make_init_file(os.path.join(output_dir, "managed_agents"))
            for agent_name, agent in self.managed_agents.items():
                agent_suffix = f"managed_agents.{agent_name}"
                if relative_path:
                    agent_suffix = relative_path + "." + agent_suffix
                agent.save(os.path.join(output_dir, "managed_agents", agent_name), relative_path=agent_suffix)

        class_name = self.__class__.__name__

        for tool in self.tools.values():
            make_init_file(os.path.join(output_dir, "tools"))
            tool.save(os.path.join(output_dir, "tools"), tool_file_name=tool.name, make_gradio_app=False)

        yaml_prompts = yaml.safe_dump(
            self.prompt_templates,
            default_style="|",
            default_flow_style=False,
            width=float("inf"),
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

        with open(os.path.join(output_dir, "prompts.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_prompts)

        agent_dict = self.to_dict()
        agent_dict["tools"] = [tool.name for tool in self.tools.values()]
        with open(os.path.join(output_dir, "agent.json"), "w", encoding="utf-8") as f:
            json.dump(agent_dict, f, indent=4)

        with open(os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{r}\n" for r in agent_dict["requirements"])

        agent_name = f"agent_{self.name}" if getattr(self, "name", None) else "agent"
        managed_agent_relative_path = relative_path + "." if relative_path is not None else ""
        app_template = textwrap.dedent("""
            import yaml
            import os
            from smolagents import GradioUI, {{ class_name }}, {{ agent_dict['model']['class'] }}

            # Get current directory path
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

            {% for tool in tools.values() -%}
            from {{managed_agent_relative_path}}tools.{{ tool.name }} import {{ tool.__class__.__name__ }} as {{ tool.name | camelcase }}
            {% endfor %}
            {% for managed_agent in managed_agents.values() -%}
            from {{managed_agent_relative_path}}managed_agents.{{ managed_agent.name }}.app import agent_{{ managed_agent.name }}
            {% endfor %}

            model = {{ agent_dict['model']['class'] }}(
            {% for key in agent_dict['model']['data'] if key not in ['class', 'last_input_token_count', 'last_output_token_count'] -%}
                {{ key }}={{ agent_dict['model']['data'][key]|repr }},
            {% endfor %})

            {% for tool in tools.values() -%}
            {{ tool.name }} = {{ tool.name | camelcase }}()
            {% endfor %}

            with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
                prompt_templates = yaml.safe_load(stream)

            {{ agent_name }} = {{ class_name }}(
                model=model,
                tools=[{% for tool_name in tools.keys() if tool_name != "final_answer" %}{{ tool_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                managed_agents=[{% for subagent_name in managed_agents.keys() %}agent_{{ subagent_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                {% for attribute_name, value in agent_dict.items() if attribute_name not in ["model", "tools", "prompt_templates", "authorized_imports", "managed_agents", "requirements"] -%}
                {{ attribute_name }}={{ value|repr }},
                {% endfor %}prompt_templates=prompt_templates
            )
            if __name__ == "__main__":
                GradioUI({{ agent_name }}).launch()
            """).strip()
        template_env = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined)
        template_env.filters["repr"] = repr
        template_env.filters["camelcase"] = lambda value: "".join(word.capitalize() for word in value.split("_"))
        template = template_env.from_string(app_template)

        app_text = template.render(
            {
                "agent_name": agent_name,
                "class_name": class_name,
                "agent_dict": agent_dict,
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "managed_agent_relative_path": managed_agent_relative_path,
            }
        )

        with open(os.path.join(output_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write(app_text + "\n")

    def to_dict(self) -> Dict[str, Any]:
        """Converts agent into a dictionary."""
        for attr in ["final_answer_checks", "step_callbacks"]:
            if getattr(self, attr, None):
                self.logger.log(f"This agent has {attr}: they will be ignored by this method.", LogLevel.INFO)

        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        tool_requirements = {req for tool in self.tools.values() for req in tool.to_dict()["requirements"]}
        managed_agents_requirements = {
            req for managed_agent in self.managed_agents.values() for req in managed_agent.to_dict()["requirements"]
        }
        requirements = tool_requirements | managed_agents_requirements
        if hasattr(self, "authorized_imports"):
            requirements.update(
                {package.split(".")[0] for package in self.authorized_imports if package not in BASE_BUILTIN_MODULES}
            )

        agent_dict = {
            "tools": tool_dicts,
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": {
                managed_agent.name: managed_agent.__class__.__name__ for managed_agent in self.managed_agents.values()
            },
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "verbosity_level": int(self.logger.level),
            "grammar": self.grammar,
            "planning_interval": self.planning_interval,
            "name": self.name,
            "description": self.description,
            "requirements": list(requirements),
        }
        if hasattr(self, "authorized_imports"):
            agent_dict["authorized_imports"] = self.authorized_imports
        if hasattr(self, "use_e2b_executor"):
            agent_dict["use_e2b_executor"] = self.use_e2b_executor
        if hasattr(self, "max_print_outputs_length"):
            agent_dict["max_print_outputs_length"] = self.max_print_outputs_length
        return agent_dict

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Loads an agent defined on the Hub.

        <Tip warning={true}>

        Loading a tool from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        </Tip>

        Args:
            repo_id (`str`):
                The name of the repo on the Hub where your tool is defined.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            trust_remote_code(`bool`, *optional*, defaults to False):
                This flags marks that you understand the risk of running remote code and that you trust this tool.
                If not setting this to True, loading the tool from Hub will fail.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your agent, and the
                others will be passed along to its init.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading an agent from Hub requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        download_kwargs = {"token": token, "repo_type": "space"} | {
            key: kwargs.pop(key)
            for key in [
                "cache_dir",
                "force_download",
                "proxies",
                "revision",
                "local_files_only",
            ]
            if key in kwargs
        }

        download_folder = Path(snapshot_download(repo_id=repo_id, **download_kwargs))
        return cls.from_folder(download_folder, **kwargs)

    @classmethod
    def from_folder(cls, folder: Union[str, Path], **kwargs):
        """Loads an agent from a local folder.

        Args:
            folder (`str` or `Path`): The folder where the agent is saved.
            **kwargs: Additional keyword arguments that will be passed to the agent's init.
        """
        folder = Path(folder)
        agent_dict = json.loads((folder / "agent.json").read_text())

        managed_agents = []
        for managed_agent_name, managed_agent_class in agent_dict["managed_agents"].items():
            agent_cls = getattr(importlib.import_module("smolagents.agents"), managed_agent_class)
            managed_agents.append(agent_cls.from_folder(folder / "managed_agents" / managed_agent_name))

        tools = []
        for tool_name in agent_dict["tools"]:
            tool_code = (folder / "tools" / f"{tool_name}.py").read_text()
            tools.append(Tool.from_code(tool_code))

        model_class: Model = getattr(importlib.import_module("smolagents.models"), agent_dict["model"]["class"])
        model = model_class.from_dict(agent_dict["model"]["data"])

        args = dict(
            model=model,
            tools=tools,
            managed_agents=managed_agents,
            name=agent_dict["name"],
            description=agent_dict["description"],
            max_steps=agent_dict["max_steps"],
            planning_interval=agent_dict["planning_interval"],
            grammar=agent_dict["grammar"],
            verbosity_level=agent_dict["verbosity_level"],
        )
        if cls.__name__ == "CodeAgent":
            args["additional_authorized_imports"] = agent_dict["authorized_imports"]
            args["use_e2b_executor"] = agent_dict["use_e2b_executor"]
            args["max_print_outputs_length"] = agent_dict["max_print_outputs_length"]
        args.update(kwargs)
        return cls(**args)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload agent",
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the agent to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload agent"`):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `None`):
                Whether to make the repo private. If `None`, the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.
        """
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        repo_id = repo_url.repo_id
        metadata_update(
            repo_id,
            {"tags": ["smolagents", "agent"]},
            repo_type="space",
            token=token,
            overwrite=True,
        )

        with tempfile.TemporaryDirectory() as work_dir:
            self.save(work_dir)
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            return upload_folder(
                repo_id=repo_id,
                commit_message=commit_message,
                folder_path=work_dir,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )


class ToolCallingAgent(MultiStepAgent):
    """
    This agent uses JSON-like tool calls, using method `model.get_tool_call` to leverage the LLM engine's tool calling capabilities.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[PromptTemplates] = None,
        planning_interval: Optional[int] = None,
        agent_kb: bool = False,
        agent_type: Optional[str] = "tool_agent",
        top_k:Optional[int] = 1,
        retrieval_type:Optional[str] = "hybrid",
        **kwargs,
    ):
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files(f"smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            agent_kb=agent_kb,
            agent_type= agent_type,
            top_k = top_k,
            retrieval_type = retrieval_type,
            **kwargs,
        )

        self.task_records = {}
        self.tool_call_records = []

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"tools": self.tools, "managed_agents": self.managed_agents},
        )
        return system_prompt
    
    def _run(self, task: str, images: List[str] | None = None, additional_knowledge: Optional[str] = None) -> Generator[ActionStep | AgentType, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """
        final_answer = None
        self.step_number = 1
        while final_answer is None and self.step_number <= self.max_steps:
            step_start_time = time.time()
            memory_step = ActionStep(
                step_number=self.step_number,
                start_time=step_start_time,
                observations_images=images,
            )
            try:
                if (self.planning_interval is not None and self.step_number % self.planning_interval == 0 and self.planning_interval != 1) or self.step_number == 1:
                    planning_step = self.planning_step(
                        task,
                        is_first_step=(self.step_number == 1),
                        step=self.step_number,
                    )
                    self.memory.steps.append(planning_step)
                self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)

                final_answer = self.step(memory_step)
                if final_answer is not None and self.final_answer_checks is not None:
                    for check_function in self.final_answer_checks:
                        try:
                            assert check_function(final_answer, self.memory)
                        except Exception as e:
                            final_answer = None
                            raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)
            except AgentError as e:
                memory_step.error = e
                raise
            finally:
                memory_step.end_time = time.time()
                memory_step.duration = memory_step.end_time - step_start_time
                self.memory.steps.append(memory_step)
                for callback in self.step_callbacks:
                    if len(inspect.signature(callback).parameters) == 1:
                        callback(memory_step)
                    else:
                        callback(memory_step, agent=self)
                self.step_number += 1
                yield memory_step

        if final_answer is None and self.step_number == self.max_steps + 1:
            error_message = "Reached max steps."
            step_start_time = time.time()
            final_answer = self.provide_final_answer(task, images)
            final_memory_step = ActionStep(
                step_number=self.step_number, error=AgentMaxStepsError(error_message, self.logger)
            )
            final_memory_step.action_output = final_answer
            final_memory_step.end_time = time.time()
            final_memory_step.duration = memory_step.end_time - step_start_time
            self.memory.steps.append(final_memory_step)

            _task_info = {
                'answer': final_answer,
                'tool_calls': self.tool_call_records
            }
            self.task_records[self.task] = _task_info

            for callback in self.step_callbacks:
                if len(inspect.signature(callback).parameters) == 1:
                    callback(final_memory_step)
                else:
                    callback(final_memory_step, agent=self)
            yield final_memory_step

        yield handle_agent_output_types(final_answer)


    def step(self, memory_step: ActionStep, memory_messages=None) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages() if memory_messages is None else memory_messages

        self.input_messages = memory_messages

        memory_step.model_input_messages = memory_messages.copy()

        try:
            model_message: ChatMessage = self.model(
                memory_messages,
                tools_to_call_from=list(self.tools.values()),
                stop_sequences=["Observation:"],
            )
            memory_step.model_output_message = model_message
            if model_message.tool_calls is None or len(model_message.tool_calls) == 0:
                raise Exception("Model did not call any tools. Call `final_answer` tool to return a final answer.")
            tool_call = model_message.tool_calls[0]
            tool_name, tool_call_id = tool_call.function.name, tool_call.id
            tool_arguments = tool_call.function.arguments

        except Exception as e:
            raise AgentGenerationError(f"Error in generating tool call with model:\n{e}", self.logger) from e

        memory_step.tool_calls = [ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)]

        self.logger.log(
            Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
            level=LogLevel.INFO,
        )
        if tool_name == "final_answer":
            if isinstance(tool_arguments, dict):
                if "answer" in tool_arguments:
                    answer = tool_arguments["answer"]
                else:
                    answer = tool_arguments
            else:
                answer = tool_arguments
            if (
                isinstance(answer, str) and answer in self.state.keys()
            ):
                final_answer = self.state[answer]
                self.logger.log(
                    f"[bold {YELLOW_HEX}]Final answer:[/bold {YELLOW_HEX}] Extracting key '{answer}' from state to return value '{final_answer}'.",
                    level=LogLevel.INFO,
                )
            else:
                final_answer = answer
                self.logger.log(
                    Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                    level=LogLevel.INFO,
                )
            if self.debug:
                take_a_breath()
            memory_step.action_output = final_answer

            _task_info = {
                'answer': final_answer,
                'tool_calls': self.tool_call_records
            }
            self.task_records[self.task] = _task_info
            self.tool_call_records=[]

            return final_answer
        else:
            if tool_arguments is None:
                tool_arguments = {}
            observation = self.execute_tool_call(tool_name, tool_arguments)

            _tool_info = {
                'name': tool_name,
                'args': tool_arguments,
                'observation': observation
            }
            self.tool_call_records.append(_tool_info)
            
            observation_type = type(observation)
            if observation_type in [AgentImage, AgentAudio]:
                if observation_type == AgentImage:
                    observation_name = "image.png"
                elif observation_type == AgentAudio:
                    observation_name = "audio.mp3"
                # TODO: observation naming could allow for different names of same type

                self.state[observation_name] = observation
                updated_information = f"Stored '{observation_name}' in memory."
            else:
                updated_information = str(observation).strip()
            self.logger.log(
                f"Observations: {updated_information.replace('[', '|')}",  # escape potential rich-tag-like components
                level=LogLevel.INFO,
            )
            if self.debug:
                take_a_breath()
            memory_step.observations = updated_information
            return None


class CodeAgent(MultiStepAgent):
    """
    In this agent, the tool calls will be formulated by the LLM in code format, then parsed and executed.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        use_e2b_executor (`bool`, default `False`): Whether to use the E2B executor for remote code execution.
        max_print_outputs_length (`int`, *optional*): Maximum length of the print outputs.
        search_type (`str`, default `none`): Types of tts method to apply.
        summary (`bool`, default `False`): Whether to use memory summary for reasoning.
        use_long_term_memory (`bool`, default `False`): Whether to use long-term memory for reasoning.
        **kwargs: Additional keyword arguments.

    """
    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[PromptTemplates] = None,
        grammar: Optional[Dict[str, str]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        agent_type: Optional[str] = "code_agent",
        planning_interval: Optional[int] = None,
        use_e2b_executor: bool = False,
        max_print_outputs_length: Optional[int] = None,
        search_type: str='default',
        summary: bool = False,
        use_long_term_memory: bool = False,
        retrieve_key_memory: bool = False,
        **kwargs,
    ):
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.use_e2b_executor = use_e2b_executor
        self.max_print_outputs_length = max_print_outputs_length
        self.search_type=search_type
        self.summary=summary
        self.use_long_term_memory=use_long_term_memory
        self.retrieve_key_memory=retrieve_key_memory
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.texts = []  # 存储原始文本
        self.embeddings = []  # 存储对应的向量表示
        self.long_term_memory=[]
        self.Most_Similar=None
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files(f"smolagents.prompts").joinpath("code_agent.yaml").read_text()
        )

        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            grammar=grammar,
            planning_interval=planning_interval,
            agent_type=agent_type,
            **kwargs,
        )
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                0,
            )

        if use_e2b_executor and len(self.managed_agents) > 0:
            raise Exception(
                f"You passed both {use_e2b_executor} and some managed agents. Managed agents is not yet supported with remote code execution."
            )

        all_tools = {**self.tools, **self.managed_agents}
        if use_e2b_executor:
            self.python_executor = E2BExecutor(
                self.additional_authorized_imports,
                list(all_tools.values()),
                self.logger,
            )
        else:
            self.python_executor = LocalPythonInterpreter(
                self.additional_authorized_imports,
                all_tools,
                max_print_outputs_length=max_print_outputs_length,
            )

    def embed_text(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Vectorization failed: {e}")
            raise
    
    def process_and_store_text(self, text: str, top_n: int = 1) -> List[Dict[str, float]]:
        similar_texts = self.find_most_similar(text, top_n)
        self.add_text(text)
        return similar_texts
    
    def add_text(self, text: str) -> None:
        embedding = self.embed_text(text)
        self.texts.append(text)
        self.embeddings.append(embedding)
    
    def find_most_similar(self, query_text: str, top_n: int = 1) -> List[Dict[str, float]]:
        if not self.texts:
            return []
            
        query_embedding = np.array(self.embed_text(query_text)).reshape(1, -1)
        embeddings_array = np.array(self.embeddings)
        similarities = cosine_similarity(query_embedding, embeddings_array)[0]
        top_indices = similarities.argsort()[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.texts[idx],
                "similarity": similarities[idx]
            })
        return results

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "authorized_imports": (
                    "You can import from any package you want."
                    if "*" in self.authorized_imports
                    else str(self.authorized_imports)
                ),
            },
        )
        return system_prompt
    
    def edit_code_by_user(self, failed_code: str):
        import subprocess

        def y_n_prompt(prompt: str) -> bool:
            while True:
                user_input = input(prompt).strip().lower()
                if user_input in ['y', 'n']:
                    return user_input == 'y'
                else:
                    print("Please input 'Y' or 'n'.")

        new_code = ""
        os.makedirs('tmp', exist_ok=True)
        code_file = os.path.join('tmp', f"{hash(failed_code)}.py")
        with open(code_file, 'w') as f:
            f.write(failed_code)

        if y_n_prompt("Open Code Editor? (Y/n): "):
            result = subprocess.run(["vim", code_file], check=True)
            if result.returncode == 0:
                with open(code_file, 'r') as f:
                    new_code = f.read()
        return new_code

    def execute_code(self, memory_step: ActionStep, code_action: str):
        self.logger.log_code(title="Executing code:", content=code_action, level=LogLevel.INFO)
        try:
            output, execution_logs, is_final_answer = self.python_executor(
                code_action,
                self.state,
            )
            execution_outputs_console = []
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            observation = "Execution logs:\n" + execution_logs
            return (
                observation,
                output,
                memory_step,
                is_final_answer,
                execution_outputs_console,
            )
        except Exception as e:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            if self.debug:
                new_code = self.edit_code_by_user(failed_code=code_action)
                if new_code:
                    self.execute_code(memory_step, new_code)
            raise AgentExecutionError(error_msg, self.logger)

    def _parse_plan(self, raw_plan):
        parallel_section = re.search(r'##PARALLEL_LIST\n([ST\d, ]+)', raw_plan)
        parallel_list = [x.strip() for x in parallel_section.group(1).split(',')] if parallel_section else []
        
        subtask_dict = {}
        pattern = r'##(ST\d+)([\s\S]*?)(?=\n##ST|\Z)'
        
        for match in re.finditer(pattern, raw_plan):
            st_code, content = match.groups()
            content_stripped = content.strip()
            title_line, _, steps_content = content_stripped.partition('\n')
            title = title_line.split(':', 1)[-1].strip() if ':' in title_line else title_line.strip()
            steps = re.findall(r'^\d+\..*$', content_stripped, flags=re.MULTILINE)
            steps_str = '\n'.join(steps)
            subtask_dict[st_code] = {
                "title": title,
                "steps": steps_str
            }
        return parallel_list, subtask_dict
    
    def _retrieve_key_memory(self, memory_messages: List[Message]):
        if not self.retrieve_key_memory or len(memory_messages) <= 4:
            return
        prompt = f"Summarize the following text, i.e. what the agent did at the current step. Highlight key points: {memory_messages[-1]} \n Note that you are only responsible for summarizing, not providing optimization suggestions for the next step."
        chat_message: ChatMessage = self.model(
            [
                Message(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])
            ]
        )
        if chat_message is None or chat_message.content is None:
            raise ValueError("Model returned empty or invalid chat message.")
        summary_content =chat_message.content
        similar_texts = self.process_and_store_text(summary_content, top_n=1)
        print('similar_texts',similar_texts)
        if similar_texts:
            Most_Similar=similar_texts[0]['text']
            return Most_Similar
        else:
            print("This is the first step, there is no similar step already")
            return

    def _update_long_term_memory(self, memory_messages: List[Message]):
        if not self.use_long_term_memory or len(memory_messages) <= 1:
            return

        prompt = f"""Here is the agent's execution content from the previous step: {memory_messages[-1]}. 
            Here is the long-term memory formed by summarizing the agent's historical execution content: {self.long_term_memory}. 
            Please combine the **agent's previous execution content** and the **existing long-term memory**, summarize them while highlighting the key points, and form a new long-term memory to help the agent reason better in subsequent steps.
            Note:
            1. You must state the main objective of the task at the very beginning of the long-term memory.
            2. You must provide optimization suggestions for the next step."""

        chat_message: ChatMessage = self.model(
            [
                Message(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])
            ]
        )

        if chat_message is None or chat_message.content is None:
            raise ValueError("Model returned empty or invalid chat message.")

        summary_content = chat_message.content
        self.long_term_memory = summary_content

    def step(self, memory_step: ActionStep, memory_messages=None, additional_prompt: str = "", memory_steps: List[ActionStep | PlanningStep | TaskStep]=None) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        This function executes a step by sending messages to a model, processing the response, 
        and interacting with external tools (e.g., code execution). It updates the memory step with
        relevant information and returns the final answer if the step is complete.

        Args:
            memory_step (ActionStep): The current memory step which holds the state and information of the step.
            memory_messages (list[Message], optional): List of previous memory messages. If None, memory is fetched from `write_memory_to_messages()`.
            additional_prompt (str, optional): An optional prompt to add to the model input to guide the model's behavior.

        Returns:
            Union[None, Any]: The model's output if the step is final; otherwise, None.
        """
        memory_messages = self.write_memory_to_messages() if memory_messages is None else memory_messages
        if self.use_long_term_memory and len(memory_messages)>1:
            self.long_term_memory = self._update_long_term_memory(memory_messages)
        if self.retrieve_key_memory and len(memory_messages)>4:
            self.Most_Similar = self._retrieve_key_memory(memory_messages)

        current_step = memory_steps[-1]
        current_message = memory_messages[-1]['content'][0]['text']
        plan_list = []
        message_list = []

        if current_message.startswith("[PLAN]"):
            plan_content = current_step.model_output_message_plan.content
            parallel_list, subtask_dict = self._parse_plan(plan_content)

            for plan_name in parallel_list:
                subtask_title = subtask_dict[plan_name]["title"]
                subtask_steps = subtask_dict[plan_name]["steps"]

                subtask_plan = textwrap.dedent(
                                f"""ANSWER THE SUBTASK:
                                ```
                                subtask: {subtask_title}
                                steps: {subtask_steps}
                                ```"""
                            )
                plan_list.append(subtask_plan)

        if len(plan_list) > 0:
            for plan in plan_list:
                message = Message(role=MessageRole.USER, content=[{"type": "text", "text": f"[SUB TASK AND steps]:\n{plan.strip()}"}])
                message_list.append(message)
            
            model_input_messages_list = []
            model_output_message_list = []
            model_output_list = []
            observation_list = []
            action_output_list = []
            tool_call_list = []

            for message in message_list:
                input_messages = memory_messages.copy()[:-1]
                input_messages.append(message)
                if additional_prompt and self.reflection:
                    input_messages.append(Message(role=MessageRole.SYSTEM, content=[{"type": "text", "text": additional_prompt}]))

                model_input_messages_list.append(input_messages)
                
                try:
                    additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
                    chat_message: ChatMessage = self.model(
                        input_messages,
                        stop_sequences=["<end_code>", "Observation:"],
                        **additional_args,
                    )

                    if chat_message is None or chat_message.content is None:
                        raise ValueError("Model returned empty or invalid chat message.")
                    if isinstance(chat_message.content, list):
                        # If it's a list, join its elements into a single string
                        model_output = ''.join([str(item) for item in chat_message.content])  # Join list items into a string
                    else:
                        # If it's already a string, use it directly
                        model_output = chat_message.content
                    model_output_message_list.append(chat_message)
                    model_output = chat_message.content
                    model_output_list.append(model_output)
                except Exception as e:
                    raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

                self.logger.log_markdown(
                    content=model_output,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

                # Parse
                try:
                    code_action = fix_final_answer_code(parse_code_blobs(model_output))
                except Exception as e:
                    error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
                    if self.debug:
                        print(error_msg)
                        new_code = self.edit_code_by_user(failed_code=model_output)
                        if new_code:
                            code_action = fix_final_answer_code(parse_code_blobs(new_code))
                    else:
                        raise AgentParsingError(error_msg, self.logger)

                tool_call_list.append(
                    ToolCall(
                        name="python_interpreter",
                        arguments=code_action,
                        id=f"call_{len(memory_steps)}",
                    )
                )

                observation, output, memory_step, is_final_answer, execution_outputs_console = self.execute_code(
                    memory_step, code_action
                )

                truncated_output = truncate_content(str(output))
                observation += "Last output from code snippet:\n" + truncated_output
                observation_list.append(observation)

                execution_outputs_console += [
                    Text(
                        f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                        style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
                    ),
                ]
                self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
                if self.debug:
                    take_a_breath()
                action_output_list.append(output)

            memory_step.model_input_messages = f"model_input_message_list: {model_input_messages_list}"
            memory_step.model_output_message = f"model_output_message_list: {model_output_message_list}"
            memory_step.model_output = f"model_output_list: {model_output_list}"
            memory_step.tool_calls = tool_call_list
            memory_step.observations = f"observation_list: {observation_list}"
            memory_step.action_output = action_output_list

            return action_output_list if is_final_answer else None
        
        else:

            memory_messages = self.write_memory_to_messages(summary_mode=True) if memory_messages is None else memory_messages

            self.input_messages = memory_messages.copy()

            if additional_prompt and self.reflection:
                self.input_messages.append(Message(role=MessageRole.SYSTEM, content=[{"type": "text", "text": additional_prompt}]))

            memory_step.model_input_messages = self.input_messages.copy()

            try:
                
                additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
                if self.use_long_term_memory and self.retrieve_key_memory:
                    if self.Most_Similar is not None:
                        self.input_messages.append(
                        Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"This is the most similar historical steps: {self.Most_Similar.strip()}"}])
                    )
                    if self.long_term_memory is not None:
                        self.long_term_memory.append(
                            Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"This is the long-term memory of the agent of this task: {self.long_term_memory.strip()}"}])
                        )
                    chat_message: ChatMessage = self.model(
                        self.input_messages,
                        stop_sequences=["<end_code>", "Observation:"],
                        **additional_args,
                    )
                elif self.use_long_term_memory:
                    if self.long_term_memory is not None:
                        self.input_messages.append(
                            Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"this is the long-term memory of the agent of this task: {self.long_term_memory.strip()}"}])
                        )
                    chat_message: ChatMessage = self.model(
                        self.input_messages,
                        stop_sequences=["<end_code>", "Observation:"],
                        **additional_args,
                    )
                elif self.retrieve_key_memory:
                    if self.Most_Similar is not None:
                        self.input_messages.append(
                        Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"This is the most similar historical steps: {self.Most_Similar.strip()}"}])
                    )
                    chat_message: ChatMessage = self.model(
                        self.input_messages,
                        stop_sequences=["<end_code>", "Observation:"],  # Define stopping conditions
                        **additional_args,
                    )
                else:
                    chat_message: ChatMessage = self.model(
                        self.input_messages,
                        stop_sequences=["<end_code>", "Observation:"],
                        **additional_args,
                    )
                if chat_message is None or chat_message.content is None:
                    raise ValueError("Model returned empty or invalid chat message.")

                if isinstance(chat_message.content, list):
                    model_output = ''.join([str(item) for item in chat_message.content])
                else:
                    model_output = chat_message.content
                memory_step.model_output_message = chat_message
                memory_step.model_output = model_output

            except Exception as e:
                raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

            self.logger.log_markdown(
                content=model_output,
                title="Output message of the LLM:",
                level=LogLevel.DEBUG,
            )

            try:
                code_action = fix_final_answer_code(parse_code_blobs(model_output))
            except Exception as e:
                error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
                if self.debug:
                    new_code = self.edit_code_by_user(failed_code=model_output)
                    if new_code:
                        code_action = fix_final_answer_code(parse_code_blobs(new_code))
                else:
                    raise AgentParsingError(error_msg, self.logger)

            memory_step.tool_calls = [
                ToolCall(
                    name="python_interpreter",
                    arguments=code_action,
                    id=f"call_{len(self.memory.steps)}",
                )
            ]

            observation, output, memory_step, is_final_answer, execution_outputs_console = self.execute_code(
                memory_step, code_action
            )

            truncated_output = truncate_content(str(output))
            observation += "Last output from code snippet:\n" + truncated_output
            memory_step.observations = observation

            execution_outputs_console += [
                Text(
                    f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                    style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
                ),
            ]
            self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)

            if self.debug:
                take_a_breath()

            memory_step.action_output = output

            return output if is_final_answer else None
        
    def track_action_state(self, current_step, search_count, new_search_id, answer_message):
        error_message = current_step.error.message if current_step.error else None
        return {
            'search_id': new_search_id,
            'search_count': search_count,
            'current_depth': current_step.step_number,
            'model_output': current_step.model_output,
            'action_output': current_step.action_output,
            'error_message': error_message,
            'observations': current_step.observations,
            'score': current_step.score,
            'evaluate_thought': current_step.evaluate_thought,
            'answer_message': answer_message,
        }

    def _run(self, task: str, images: List[str] | None = None, additional_knowledge: Optional[str] = None) -> Generator[ActionStep | AgentType, None, None]:
        """
        Run the agent to execute a given task using specified search strategies.

        Args:
            task (str): The task description.
            images (List[str], optional): Image inputs for the task. Defaults to None.
            additional_knowledge (str, optional): Extra context/information for solving the task. Defaults to None.

        Yields:
            ActionStep | AgentType: Steps taken during execution or final result.
        """
        strategy_map = {
            'BON': self._run_bon_strategy,
            'BON-wise': self._run_bon_wise_strategy,
            'Beam-Search': self._run_beam_search_strategy,
            'Tree-Search': self._run_tree_search_strategy,
            'default': self._run_baseline_strategy,
        }
        self.step_number = 1
        if self.search_type in strategy_map:
            yield from strategy_map[self.search_type](task, images, additional_knowledge)
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")

    def _create_planning_step(self, task: str, step_number: int, additional_knowledge: Optional[str] = None):
        planning_step = self.planning_step(
            task,
            is_first_step=(step_number == 1),
            step=step_number,
            additional_knowledge=additional_knowledge
        )
        self.logger.log_rule(f"Step {step_number}", level=LogLevel.INFO)
        return planning_step

    def _record_action(self, memory_step, step_number, answer_message):
        new_search_id = str(uuid4())[:6]
        self.action_trajectory.append(self.track_action_state(memory_step, step_number, new_search_id, answer_message))

    def _finalize_with_max_steps_check(self, task, images, memory_steps):
        assert memory_steps is not None, "memory_steps cannot be None"
        # assert len(memory_steps) > 0, f"memory_steps cannot be empty, current length: {len(memory_steps)}"
        self.memory.steps = memory_steps

        if self.step_number == self.max_steps + 1:
            error_message = "Reached max steps."
            final_answer = self.provide_final_answer(task, images)

            final_memory_step = ActionStep(
                step_number=self.step_number,
                error=AgentMaxStepsError(error_message, self.logger)
            )
            final_memory_step.action_output = final_answer
            final_memory_step.end_time = time.time()

            self.memory.steps.append(final_memory_step)

            for callback in self.step_callbacks:
                if len(inspect.signature(callback).parameters) == 1:
                    callback(final_memory_step)
                else:
                    callback(final_memory_step, agent=self)

            yield final_memory_step
    def _final_result_merge(self,final_answer_candidates,mode='list-wise'):
        if mode=='list-wise':
            answer_message_picklist = [candidate[2] for candidate in final_answer_candidates]  # 提取所有answer_message
            text=''
            for idx, trajectory in enumerate(answer_message_picklist):
                text += f'---Trajectory - {idx}---\n'
                text += trajectory + '\n'  
            text += "you can start!"
            _, ind = evaluate_answer(text,system_prompt = self.ORM_list_wise_prompt,mode = 'ORM-list-wise' )
            if 0 <= int(ind) < len(final_answer_candidates):
                final_answer = final_answer_candidates[int(ind)][0]  # 获取对应位置的final_answer
        elif mode=='scoring':
            final_answer=max(final_answer_candidates, key=lambda x: x[1])[0]
        elif mode=='voting':
            answers = [candidate[0] for candidate in final_answer_candidates]  # 提取所有answer_message
            answer_counts = Counter(answers)
            most_common_answer, count = answer_counts.most_common(1)[0]
            # 如果有多数答案，使用多数答案；否则使用模型判断
            if count > 1:
                final_answer = most_common_answer
            else:
                final_answer = random.choice(answers)
        else:
            raise ValueError
        return final_answer
        

    def _verify_process(self,process_candidates,mode='list-wise',select_num=1):
        best_memory_step=[]
        if mode=='list-wise':
            for _ in range(select_num):
                answer_message_picklist = [candidate[2] for candidate in process_candidates]  # 提取所有answer_message
                text = ""
                for idx, trajectory in enumerate(answer_message_picklist):
                    text += f'---Trajectory - {idx}---\n'
                    text += trajectory + '\n'  
                text += "you can start!"
                _, ind = evaluate_answer(text,system_prompt = self.PRM_list_wise_prompt,mode = 'PRM-list-wise' )
                if 0 <= int(ind) < len(process_candidates):
                    best_memory_step.append(process_candidates[int(ind)][0])  # 获取对应位置的final_answer
                    process_candidates.pop(int(ind))
        elif mode=='scoring':
            best_memory_step=max(process_candidates, key=lambda x: x[1])[:select_num]
        else:
            raise ValueError
        if len(best_memory_step)==1:
            return best_memory_step[0]
        elif len(best_memory_step)>1:
            return best_memory_step
        else:
            return []



    def _run_bon_strategy(self, task, images, additional_knowledge):
        evaluate = False if self.n_rollouts == 1 else True
        final_answer_candidates = []
        Task_steps = self.memory.steps.copy()

        for _ in range(self.n_rollouts):
            memory_steps = Task_steps.copy()
            task_success = False
            self.step_number = 1
            planning_step = self._create_planning_step(task, self.step_number, additional_knowledge)
            memory_steps.append(planning_step)

            memory_messages = self.write_memory_to_messages(memory_steps=memory_steps)
            step_number = self.step_number

            while not task_success and step_number <= self.max_steps:
                try:
                    final_answer, evaluation_score, _ = self.process_step(
                        step_number, images, memory_messages, memory_steps, evaluate=evaluate)
                except Exception as e:
                    self.logger.log(f"Error in BON step {step_number}: {str(e)}")
                    final_answer = None
                
                answer_message = self.get_memory_step_message(memory_steps, None)
                if final_answer:
                    final_answer=prepare_response(task,memory_messages,self.model)
                    answer_message += '\n' + 'Final_Answer: ' + final_answer
                    final_answer_candidates.append((final_answer, evaluation_score,answer_message))
                    task_success = True
                else:
                    step_number += 1

        final_answer = self._final_result_merge(final_answer_candidates,mode='list-wise')

        yield from self._finalize_with_max_steps_check(task, images, memory_steps)

        yield handle_agent_output_types(final_answer)

    def _run_bon_wise_strategy(self, task, images, additional_knowledge):
        memory_steps = self.memory.steps.copy()
        planning_step = self._create_planning_step(task, self.step_number, additional_knowledge)
        base_memory_steps = memory_steps.copy()
        base_memory_steps.append(planning_step)
        task_success = False
        evaluate = True
        final_answer_candidates = []
        while not task_success and self.step_number <= self.max_steps:
            process_candidates = []
            for i in range(self.n_rollouts):
                current_memory_steps = copy.deepcopy(base_memory_steps)
                current_memory_messages = self.write_memory_to_messages(current_memory_steps)
                try:
                    final_answer, evaluation_score, reflection = self.process_step(
                        self.step_number, images, current_memory_messages, current_memory_steps, '', evaluate)
                except Exception as e:
                    self.logger.log(f"Error in BON-wise step {self.step_number}: {str(e)}")
                    continue
                answer_message = self.get_memory_step_message(current_memory_steps, None)
                process_candidates.append((current_memory_steps,evaluation_score,answer_message))
                self._record_action(current_memory_steps[-1], self.step_number, answer_message)
                if final_answer:
                    final_answer=prepare_response(task,current_memory_messages,self.model)
                    task_success=True
                    break
            base_memory_steps=self._verify_process(process_candidates,mode='list-wise')
            self.step_number += 1
        self.memory.steps=base_memory_steps

        yield from self._finalize_with_max_steps_check(task, images, base_memory_steps)

        yield handle_agent_output_types(final_answer)

    def _run_beam_search_strategy(self, task, images, additional_knowledge):
        memory_steps = self.memory.steps.copy()
        planning_step = self._create_planning_step(task, self.step_number, additional_knowledge)
        memory_steps.append(planning_step)
        base_memory_steps = memory_steps.copy()
        beam_size=2
        nodes_list = [base_memory_steps, copy.deepcopy(base_memory_steps)] #beamsize default2

        task_success = False
        evaluate = True
        final_answer_candidates = []
        while not task_success and self.step_number <= self.max_steps:
            process_candidates = []
            for beam_idx in range(beam_size):
                for i in range(self.n_rollouts//beam_size):  # 对每个节点进行 Branch Size 次扩展
                    # 获取当前节点的 memory_steps 和 memory_messages 副本
                    current_memory_steps = copy.deepcopy(nodes_list[beam_idx])  # 当前节点
                    current_memory_messages = self.write_memory_to_messages(current_memory_steps)
                    try:
                        final_answer, evaluation_score, reflection = self.process_step(
                            self.step_number, images, current_memory_messages, current_memory_steps, '', evaluate)
                    except Exception as e:
                        self.logger.log(f"Error in Beam-Search step {self.step_number}: {str(e)}")
                        continue
                    answer_message = self.get_memory_step_message(current_memory_steps, None)
                    if final_answer is not None:
                        final_answer=prepare_response(task,current_memory_messages,self.model)
                        answer_message += '\n' + 'Final_Answer: ' + final_answer
                        final_answer_candidates.append((final_answer, evaluation_score,answer_message))
                        if len(final_answer_candidates)>=self.n_rollouts:
                            task_success = True
                    else:
                        process_candidates.append((current_memory_steps,evaluation_score,answer_message))
                        self._record_action(current_memory_steps[-1], self.step_number, answer_message)
                    
            base_memory_steps=self._verify_process(process_candidates,mode='list-wise',select_num=beam_size)
            nodes_list=base_memory_steps
            self.step_number += 1
        self.memory.steps=current_memory_steps
        final_answer = self._final_result_merge(final_answer_candidates,mode='list-wise')
        yield from self._finalize_with_max_steps_check(task, images, current_memory_steps)

        yield handle_agent_output_types(final_answer)


    def _run_tree_search_strategy(self, task, images, additional_knowledge):
        memory_steps = self.memory.steps.copy()
        beam_size=2
        nodes_list=[]
        for tree_idx in range(self.n_rollouts//beam_size):
            planning_step = self._create_planning_step(task, self.step_number, additional_knowledge)
            nodes_list.append(memory_steps+[planning_step]) #beamsize default2
        task_success = False
        evaluate = True
        final_answer_candidates = []
        while not task_success and self.step_number <= self.max_steps:
            for beam_idx in range(beam_size):
                process_candidates = []
                for i in range(self.n_rollouts//beam_size):  # 对每个节点进行 Branch Size 次扩展
                    current_memory_steps = copy.deepcopy(nodes_list[beam_idx]) 
                    if  not current_memory_steps: 
                        self.logger.log(f"this branch have finish its task {beam_idx}")
                        continue
                    current_memory_messages = self.write_memory_to_messages(current_memory_steps)
                    try:
                        final_answer, evaluation_score, reflection = self.process_step(
                            self.step_number, images, current_memory_messages, current_memory_steps, '', evaluate)
                    except Exception as e:
                        self.logger.log(f"Error in Tree-Search step {self.step_number}: {str(e)}")
                        continue
                    answer_message = self.get_memory_step_message(current_memory_steps, None)
                    if final_answer is not None:
                        final_answer=prepare_response(task,current_memory_messages,self.model)
                        answer_message += '\n' + 'Final_Answer: ' + final_answer
                        final_answer_candidates.append((final_answer, evaluation_score,answer_message))
                        if len(final_answer_candidates)>=self.n_rollouts:
                            task_success = True
                    else:
                        process_candidates.append((current_memory_steps,evaluation_score,answer_message))
                        self._record_action(current_memory_steps[-1], self.step_number, answer_message)
                base_memory_steps=self._verify_process(process_candidates,mode='list-wise')
                nodes_list[beam_idx]=base_memory_steps
                self.step_number += 1
        self.memory.steps=current_memory_steps
        final_answer = self._final_result_merge(final_answer_candidates,mode='list-wise')
        yield from self._finalize_with_max_steps_check(task, images, current_memory_steps)
        yield handle_agent_output_types(final_answer)


    def _run_baseline_strategy(self, task, images, additional_knowledge):

        memory_steps = self.memory.steps.copy()

        planning_step = self._create_planning_step(task, self.step_number, additional_knowledge)
        memory_steps.append(planning_step)

        current_memory_messages = self.write_memory_to_messages(memory_steps=memory_steps)
        task_success, reflection = False, ''
        evaluate = True if self.reflection else False
        while not task_success and self.step_number <= self.max_steps:
            if self.planning_interval is not None and self.step_number % self.planning_interval == 0 and self.planning_interval != 1:
                planning_step = self._create_planning_step(task, self.step_number, additional_knowledge)
                memory_steps.append(planning_step)
            try:
                if not reflection:  # 如果node_dict是空字典
                    additional_prompt = ""
                else:
                    node_exp = "\n".join(f"{k}: {v}" for k, v in reflection.items())
                    additional_prompt = self.BASE_ADDITIONAL_PROMPT + f"{node_exp}\n\n"
                final_answer, _, reflection = self.process_step(
                    self.step_number, images, current_memory_messages, memory_steps,additional_prompt, evaluate=evaluate)
                answer_message = self.get_memory_step_message(memory_steps, None)

            except Exception as e:
                self.logger.log(f"Error in none strategy step {self.step_number}: {str(e)}")
                final_answer = None
                answer_message=''

            if final_answer:
                task_success = True

            self._record_action(memory_steps[-1], self.step_number, answer_message)
            self.step_number += 1
        final_answer=prepare_response(task,current_memory_messages,self.model)
        yield from self._finalize_with_max_steps_check(task, images, memory_steps)
        yield handle_agent_output_types(final_answer)

    def get_memory_step_message(self, memory_steps: List[ActionStep | PlanningStep | TaskStep],
                           current_step: ActionStep | PlanningStep | TaskStep) -> str:
        def step_to_string(step: ActionStep | PlanningStep | TaskStep) -> str:
            if isinstance(step, ActionStep):
                step_info = step.dict()
                return (
                    f"[ActionStep]\n"
                    f"  step_number: {step_info['step']}\n"
                    f"  observations: {step.observations}\n"
                    f"  action_output: {step_info['action_output']}\n"
                    f"  model_output: {step_info['model_output']}\n"
                    f"  error: {step_info['error']}\n"                )
            elif isinstance(step, TaskStep):
                return (
                    f"[TaskStep]\n"
                    f"  task: {step.task}\n"
                    f"  description: {step.description if hasattr(step, 'description') else None}"
                )
            elif isinstance(step, PlanningStep):
                return (
                    f"[PlanningStep]\n"
                    f"  facts that we knows: {step.facts}\n"
                    f"  current plan: {step.plan}\n"
                    f"  facts that we knows: {step.facts}\n"
                    f"  current plan: {step.plan}\n"
                    f"  reason: {step.reason if hasattr(step, 'reason') else None}"
                )
            else:
                step_attrs = []
                for attr_name, attr_value in step.__dict__.items():
                    step_attrs.append(f"  {attr_name} = {attr_value}")
                step_attrs_str = "\n".join(step_attrs)
                return f"[Unknown Step Type: {type(step)}]\n{step_attrs_str}"   

        if not isinstance(memory_steps, list):
            memory_steps = [memory_steps]
        all_steps = list(memory_steps)
        if current_step is not None:
            all_steps.append(current_step)

        lines = []
        for idx, step in enumerate(all_steps, start=1):
            step_str = step_to_string(step)
            lines.append(f"===== Step {idx} =====\n{step_str}")
        final_merge_message = "\n".join(lines)
        return final_merge_message

    def get_memory_message(self, memory_messages_next: List[Dict[str, Any]]):
        texts = []
        for item in memory_messages_next:
            content = item.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                first_content = content[0]
                text = first_content.get("text", "")
                texts.append(text)
        return "\n".join(texts)

        
    def process_step(self, step_number, images, memory_messages, memory_steps, additional_prompt: str = "", evaluate=True):

        reflection = None
        self.step_number = step_number
        step_start_time = time.time()

        memory_step = ActionStep(
            step_number=self.step_number,
            start_time=step_start_time,
            observations_images=images,
            score=0.0
        )

        final_answer = None
        evaluation_score = float('-inf')

        try:
            evaluation_content = ""
            answer_message = ""

            final_answer = self.step(memory_step, memory_messages, additional_prompt, memory_steps)

            if final_answer is not None and self.final_answer_checks:
                for check_function in self.final_answer_checks:
                    try:
                        assert check_function(final_answer, self.memory)
                    except Exception as e:
                        final_answer = None
                        raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)

            if final_answer is not None:
                if not isinstance(final_answer, str):
                    final_answer = str(final_answer)
                mode = 'ORM'
                memory_message = self.get_memory_step_message(memory_steps, memory_step)
                answer_message = f"{memory_message}\nFinal_Answer: {final_answer}"
                system_prompt = self.ORM_prompt
            else:
                mode = 'PRM'
                answer_message = self.get_memory_step_message(memory_steps, memory_step)
                system_prompt = self.PRM_prompt

            if evaluate:
                evaluation_score, evaluation_content = evaluate_answer(answer_message, system_prompt, mode)
            else:
                evaluation_score, evaluation_content = 0.0, ""

            if self.reflection and evaluation_score<=self.reflection_threshold:
                reflection = evaluate_answer(answer_message, self.REFLECTION_prompt, mode='reflection')

        except AgentError as e:
            memory_step.error = e
            evaluation_score = 0.0
            evaluation_content = ""

        finally:
            setattr(memory_step, 'score', evaluation_score)
            setattr(memory_step, 'evaluate_thought', evaluation_content)

            memory_step.end_time = time.time()
            memory_step.duration = memory_step.end_time - step_start_time

            memory_messages.extend(memory_step.to_messages(summary=self.summary))
            memory_steps.append(memory_step)

            for callback in self.step_callbacks:
                sig_params = inspect.signature(callback).parameters
                if len(sig_params) == 1:
                    callback(memory_step)
                else:
                    callback(memory_step, agent=self)

        return final_answer, evaluation_score, reflection