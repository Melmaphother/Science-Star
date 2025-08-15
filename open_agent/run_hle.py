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


# SYSTEM
import json
import os
import threading
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from typing import Dict, List
from omegaconf import OmegaConf, DictConfig

# Load ENVIRONMENT VARIABLES
repo_root = Path(__file__).resolve().parents[1]
env_path = repo_root / ".env"
load_dotenv(dotenv_path=env_path, override=True)
login(os.getenv("HF_TOKEN"))

# DATA
from data_utils.hle_loader import load_hle_dataset

# TOOLS
from tools.scorer import question_scorer, score_answer
from tools.reformulator import prepare_response
from tools.searcher import SearchTool
from tools.run_agents import (
    get_single_file_description,
    get_zip_description,
)
from tools.text_inspector_tool import TextInspectorTool
from tools.audio_inspector_tool import AudioInspectorTool
from tools.visual_inspector_tool import VisualInspectorTool
from tools.async_web_crawler import (
    CrawlerReadTool,
    CrawlerArchiveSearchTool,
    SimpleCrawler,
)
from tools.automodel import (
    get_api_model,
    process_selected_tasks_param,
    prepare_model_kwargs,
)


# MEMORY
from agent_kb.agent_kb_utils import AKBClient, call_model
from smolagents.memory import ActionStep, PlanningStep, TaskStep
from tqdm import tqdm
from smolagents import (
    CodeAgent,
    Model,
    ToolCallingAgent,
)

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yfinance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
    "random",
    "re",
    "sys",
    "shutil",
]


logger = logging.getLogger(__name__)

jsonl_lock = threading.Lock()

logger.warning(
    "Make sure you deactivated Tailscale VPN, else some URLs will be blocked!"
)
custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}


def load_config() -> DictConfig:
    """Load base YAML config and merge with CLI overrides (key=value).

    If a CLI key `config` is provided, it is treated as the path to the YAML base.
    Otherwise defaults to `<repo_root>/configs/hle.yaml`.
    """
    default_cfg_path = repo_root / "configs" / "hle.yaml"
    cli_cfg = OmegaConf.from_cli()
    cfg_path = Path(str(cli_cfg.get("config", default_cfg_path)))
    base_cfg = OmegaConf.load(str(cfg_path))
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    # Normalize null-like values for lists
    if cfg.get("selected_tasks") in (None, "null", "None"):
        cfg.selected_tasks = []
    return cfg


def append_answer(entry: dict, jsonl_file: str, file_lock) -> None:
    jsonl_path = Path(jsonl_file)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(entry) + "\n"
    with file_lock:
        with open(jsonl_path, "a", encoding="utf-8") as fp:
            fp.write(data)
    assert os.path.exists(jsonl_path), "File not found!"
    logger.info("Answer exported to file: {}".format(jsonl_path.resolve()))

    # Write to Markdown file
    md_file = jsonl_path.with_suffix(".md")
    with file_lock:
        with open(md_file, "a", encoding="utf-8") as md:
            md.write(f"## Question: {entry.get('question', '')}\n")
            md.write(f"- Predicted Answer: {entry.get('prediction', '')}\n")
            md.write(f"- True Answer: {entry.get('true_answer', '')}\n")
            md.write(f"- Task ID: {entry.get('task_id', '')}\n")
            md.write(f"- Task Level: {entry.get('task', '')}\n")
            md.write(f"- Start Time: {entry.get('start_time', '')}\n")
            md.write(f"- End Time: {entry.get('end_time', '')}\n")
            md.write(f"- Parsing Error: {entry.get('parsing_error', '')}\n")
            md.write(
                f"- Iteration Limit Exceeded: {entry.get('iteration_limit_exceeded', '')}\n"
            )
            if entry.get("agent_error"):
                md.write(f"- Error: {entry['agent_error']}\n")
            
            # Add judgment details if available
            judgment = entry.get("judgment_result")
            if judgment:
                md.write("### LLM Judge Evaluation\n")
                md.write(f"- **Correctness**: {judgment.get('correct', 'N/A')}\n")
                md.write(f"- **Confidence**: {judgment.get('confidence', 'N/A')}\n")
                md.write(f"- **Extracted Answer**: {judgment.get('model_answer', 'N/A')}\n")
                if judgment.get('reasoning'):
                    md.write(f"- **Judge Reasoning**: {judgment['reasoning']}\n")
                md.write(f"- **Judge Model**: {judgment.get('config_used', {}).get('model', 'N/A')}\n")
            
            md.write("### Reasoning Steps\n")
            for step in entry.get("intermediate_steps", []):
                md.write(f"- {step}\n")
            md.write("\n---\n\n")


def create_agent_hierarchy(model: Model, model_search: Model, args, debug=False):
    crawler = SimpleCrawler(serpapi_key=os.getenv("SERP_API_KEY"))
    text_limit = 100000

    search_types = ["wiki", "google", "baidu", "bing", "duckduckgo"]
    # search_types = ['wiki','duckduckgo']

    search_tools = [
        SearchTool(search_type=st, reflection=args.search_tool_reflection)
        for st in search_types
    ]

    WEB_TOOLS = [
        CrawlerReadTool(crawler),
        CrawlerArchiveSearchTool(crawler),
        TextInspectorTool(model, text_limit),
    ]
    WEB_TOOLS += search_tools

    text_webbrowser_agent = ToolCallingAgent(
        model=model_search,
        tools=WEB_TOOLS,
        max_steps=args.max_steps,
        verbosity_level=2,
        planning_interval=args.planning_interval,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"][
        "task"
    ] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""
    manager_agent = CodeAgent(
        model=model,
        tools=[
            VisualInspectorTool(model, text_limit),
            AudioInspectorTool(model, text_limit),
            TextInspectorTool(model, text_limit),
        ],
        max_steps=args.max_steps,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=args.planning_interval,
        managed_agents=[text_webbrowser_agent],
    )
    return manager_agent


def extract_intermediate_steps(agent):

    intermediate_steps = []
    for memory_step in agent.memory.steps:
        memory_step.model_input_messages = None
        step_dict = memory_step.dict()
        if isinstance(memory_step, ActionStep):
            step_dict["step_type"] = "action"
            step_dict.pop("model_output_message", None)
        elif isinstance(memory_step, TaskStep):
            step_dict["step_type"] = "task"
        elif isinstance(memory_step, PlanningStep):
            step_dict["step_type"] = "planning"
            step_dict.pop("model_output_message_facts", None)
            step_dict.pop("model_output_message_plan", None)
        else:
            step_dict["step_type"] = "unknown"
        intermediate_steps.append(step_dict)
    return intermediate_steps


def student_retrieval_process(example, args, model_id_retrieval, key, url):

    akb_client = AKBClient()

    with open("./agent_kb/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    student_agent_reason_template = prompts["student_agent_reason"]
    student_agent_refine_template = prompts["student_agent_refine"]

    student_reason = student_agent_reason_template.format(
        user_query=example["question"]
    )

    retrieval_method = {
        "hybrid": akb_client.hybrid_search,
        "text": akb_client.text_search,
        "semantic": akb_client.semantic_search,
    }[args.retrieval_type]

    student_retrieval_results = retrieval_method(student_reason, top_k=args.top_k)

    student_retrieval = ""
    for result in student_retrieval_results:
        student_retrieval += "\nSimilar task:\n"
        student_retrieval += result["query"]
        student_retrieval += "\nSuggestions:\n"
        student_retrieval += result["agent_experience"]
    student_refine = student_agent_refine_template.format(knowledge=student_retrieval)

    student_suggestions = call_model(
        query=student_refine, model_name=model_id_retrieval, key=key, url=url
    )

    return student_suggestions, retrieval_method, prompts


def teacher_retrieval_process(
    example,
    agent,
    args,
    retrieval_method,
    prompts,
    model_id_retrieval,
    key,
    key_search,
    url,
    url_search,
    output,
):

    intermediate_steps = extract_intermediate_steps(agent)

    annotated_example = {
        "question": example["question"],
        "prediction": output,
        "intermediate_steps": intermediate_steps,
    }

    teacher_agent_reason_template = prompts["teacher_agent_reason"]
    teacher_agent_refine_template = prompts["teacher_agent_refine"]

    teacher_reason = teacher_agent_reason_template.format(
        agent_log=str(annotated_example)
    )
    summary = call_model(
        query=teacher_reason,
        model_name=model_id_retrieval,
        key=key_search,
        url=url_search,
    )

    log_plan = None
    for memory_step in agent.memory.steps:
        if isinstance(memory_step, PlanningStep):
            step_dict = memory_step.dict()
            log_plan = step_dict.get("plan", "")
            break

    teacher_retrieval_results = retrieval_method(
        example["question"] + (log_plan or "") + summary, top_k=args.top_k
    )

    teacher_retrieval = ""
    for result in teacher_retrieval_results:
        teacher_retrieval += "\nSimilar task:\n"
        teacher_retrieval += result["query"]
        teacher_retrieval += "\nSuggestions:\n"
        teacher_retrieval += result["agent_experience"]

    teacher_refine = teacher_agent_refine_template.format(
        knowledge=teacher_retrieval, log_summary=summary
    )

    teacher_suggestions = call_model(
        query=teacher_refine, model_id=model_id_retrieval, key=key, url=url
    )

    return teacher_suggestions


def answer_single_question(
    example, args, model_id, model_id_search, answers_file, debug=False, retrieval=False
):

    text_limit = 100000
    model_name, key, url, model_wrapper = get_api_model(model_id)
    model_name_search, key_search, url_search, model_wrapper_search = get_api_model(
        model_id_search
    )

    kwargs = prepare_model_kwargs(model_id, args)
    kwargs_search = prepare_model_kwargs(model_id_search, args)

    model = model_wrapper(
        model_name,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
        api_key=key,
        api_base=url,
        **kwargs,
    )

    model_search = model_wrapper_search(
        model_name_search,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
        api_key=key_search,
        api_base=url_search,
        **kwargs_search,
    )

    document_inspection_tool = TextInspectorTool(model, text_limit)
    audio_inspection_tool = AudioInspectorTool(model, text_limit)
    visual_inspection_tool = VisualInspectorTool(model, text_limit)

    agent = create_agent_hierarchy(model, model_search, args, debug)

    augmented_question = (
        """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). 
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer!
Here is the task:
"""
        + example["question"]
    )

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"],
                example["question"],
                visual_inspection_tool,
                document_inspection_tool,
                audio_inspection_tool,
            )
        else:
            prompt_use_files = (
                "\n\nTo solve the task above, you will have to use this attached file:"
            )
            prompt_use_files += get_single_file_description(
                example["file_name"],
                example["question"],
                visual_inspection_tool,
                document_inspection_tool,
                audio_inspection_tool,
            )
        augmented_question += prompt_use_files

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if retrieval:
            model_name_retrieval = args.model_id_retrieval

            student_suggestions, retrieval_method, prompts = student_retrieval_process(
                example, args, model_name_retrieval, key, url
            )

            final_result = agent.run(
                augmented_question, additional_knowledge=student_suggestions
            )
            agent_memory = agent.write_memory_to_messages(summary_mode=True)
            final_result = prepare_response(
                augmented_question, agent_memory, reformulation_model=model
            )
            output = str(final_result)

            semantic_match_template = prompts["semantic_match_prompt"]

            output_query = semantic_match_template.format(
                question=example["question"],
                prediction=output,
                true_answer=example["true_answer"],
            )

            semantic_check = call_model(
                query=output_query,
                model_name=model_name_retrieval,
                key=key_search,
                url=url_search,
            )

            if (not question_scorer(output, example["true_answer"])) and (
                semantic_check == "false"
            ):
                teacher_suggestions = teacher_retrieval_process(
                    example,
                    agent,
                    args,
                    retrieval_method,
                    prompts,
                    model_name_retrieval,
                    key,
                    key_search,
                    url,
                    url_search,
                    output,
                )

                final_result = agent.run(
                    augmented_question, additional_knowledge=teacher_suggestions
                )
                agent_memory = agent.write_memory_to_messages(summary_mode=True)
                final_result = prepare_response(
                    augmented_question, agent_memory, reformulation_model=model
                )
                output = str(final_result)
        else:
            final_result = agent.run(augmented_question)
            agent_memory = agent.write_memory_to_messages(summary_mode=True)
            final_result = prepare_response(
                augmented_question, agent_memory, reformulation_model=model
            )
            output = str(final_result)

        intermediate_steps = extract_intermediate_steps(agent)

        intermediate_steps_check = [str(step) for step in agent.memory.steps]
        parsing_error = (
            True
            if any(["AgentParsingError" in step for step in intermediate_steps_check])
            else False
        )

        iteration_limit_exceeded = (
            True
            if "Agent stopped due to iteration limit or time limit." in output
            else False
        )
        raised_exception = False

    except Exception as e:
        logger.error(f"Error on task {example['task_id']}\n{e}")
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate detailed judgment using LLM judge
    judgment_result = None
    if output and not raised_exception:
        try:
            judgment_result = score_answer(
                model_answer=output,
                ground_truth=example["true_answer"],
                question=example["question"],
                method="llm_judge"
            )
        except Exception as e:
            logger.warning(f"LLM judge failed for task {example['task_id']}: {e}")
            judgment_result = None
    
    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "true_answer": example["true_answer"],
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "task": example["task"],
        "task_id": example["task_id"],
        "search_agent_actions": getattr(agent.managed_agents["search_agent"], 'task_records', []) if agent.managed_agents and "search_agent" in agent.managed_agents else [],
        "judgment_result": judgment_result,
    }
    append_answer(annotated_example, answers_file, jsonl_lock)


def get_examples_to_answer(
    answers_file, eval_df, selected_tasks=None, debug=False
) -> List[dict]:
    logger.info(f"Loading answers from {answers_file}...")
    done_questions = []
    try:
        path = Path(answers_file)
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            answer_df = pd.read_json(path, lines=True)
            done_questions = answer_df.get("task_id", []).tolist()
            logger.info(f"Found {len(done_questions)} previous results!")
        else:
            logger.info(
                "No usable records found (file missing or empty). ▶️ Starting new."
            )
    except Exception as e:
        logger.warning(
            f"Error when loading records from {answers_file}: {e}. ▶️ Starting new."
        )

    # Default to the full dataset; optionally filter by selected tasks
    filtered_df = eval_df
    if selected_tasks:
        if isinstance(selected_tasks[0], int):
            filtered_df = eval_df.iloc[selected_tasks]
        else:
            filtered_df = eval_df[eval_df["task_id"].isin(selected_tasks)]

    if debug:
        # In debug mode, rerun all tasks
        done_questions = []

    return [
        row.to_dict()
        for _, row in filtered_df.iterrows()
        if row["task_id"] not in done_questions
    ]


def main():
    args = load_config()
    logger.info(
        f"Starting run with config: {OmegaConf.to_container(args, resolve=True)}"
    )

    # Root directory
    args.repo_root = repo_root

    # Output directory
    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run name directory
    if " " in args.run_name or "/" in args.run_name:
        args.run_name = args.run_name.replace(" ", "_").replace("/", "_")
        logger.warning(
            f"Run name contains spaces or slashes, replacing with underscores {args.run_name} "
        )
    run_dir = output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # run time directory
    if args.date_time_load_from:
        # load from experiment start at date_time_load_from
        date_time = args.date_time_load_from
        if not os.path.exists(run_dir / date_time):
            raise FileNotFoundError(
                f"Previous experiment directory {run_dir / date_time} not found"
            )
        run_time_dir = run_dir / date_time
        answers_file = run_time_dir / "answers.jsonl"
    else:
        # new experiment
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_time_dir = run_dir / date_time
        run_time_dir.mkdir(parents=True, exist_ok=True)
        answers_file = run_time_dir / "answers.jsonl"

    logger.info(f"Answers file: {answers_file}")

    # Save config to run time directory
    with open(run_time_dir / "config.yaml", "w") as f:
        OmegaConf.save(config=args, f=f)

    # A specific save path: output_dir / args.run_name / date_time / "answers.jsonl"

    # Load dataset
    eval_df = load_hle_dataset(args)

    # prepare examples to run, maybe skip tasks already done
    selected_tasks = process_selected_tasks_param(args.selected_tasks)
    tasks_to_run = get_examples_to_answer(
        answers_file, eval_df, selected_tasks, args.debug
    )
    logger.info(f"Running {len(tasks_to_run)} tasks")

    # run tasks,
    # single thread if debug or concurrency is 1
    if args.debug or args.concurrency == 1:
        for example in tasks_to_run:
            answer_single_question(
                example,
                args,
                args.model_id,
                args.model_id_search,
                answers_file,
                args.debug,
                args.agent_kb,
            )
    # multi-thread if concurrency > 1
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
            futures = [
                exe.submit(
                    answer_single_question,
                    example,
                    args,
                    args.model_id,
                    args.model_id_search,
                    answers_file,
                    args.debug,
                    args.agent_kb,
                )
                for example in tasks_to_run
            ]
            for f in tqdm(
                as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"
            ):
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")

    logger.info("All tasks processed.")


if __name__ == "__main__":
    main()
