#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. Personal AI team. All rights reserved.
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

import yaml
import re
from smolagents.models import OpenAIServerModel, ChatMessage
import json
import os
import importlib

class SearchReflector:
    def __init__(self,
                 model:OpenAIServerModel=None):
        
        self.model = model if model is not None else \
                    OpenAIServerModel(
                        model_id="gpt-4.1",
                        api_base=os.getenv("OPENAI_BASE_URL"),
                        api_key=os.getenv("OPENAI_API_KEY")
                    )

        try:
            prompts=yaml.safe_load(
            importlib.resources.files("reflectors.prompts").joinpath("search_prompts.yaml").read_text())

            self.query_rollout_prompt = prompts['query_rollout']
            self.query_reflect_prompt = prompts['query_reflection']
            self.result_reflect_prompt = prompts['result_reflection']
        except:
            self.query_rollout_prompt = ""
            self.query_reflect_prompt = ""
            self.result_reflect_prompt = ""


    def _pack_message(self, role: str, content: str) -> list[dict]:
        packed_message = [
                {
                    "role": role,
                    "content": content,
                }
            ]
        return packed_message
    

    def query_rollout(self, query:str, n_rollout:int=1) -> list[str]:

        prompted_query = self.query_rollout_prompt.format(query=query, roll_out=n_rollout)
        input_messages = self._pack_message(role="user", content=prompted_query)

        chat_message :ChatMessage = self.model(
            messages = input_messages,
            stop_sequences=["<end>"],
        )
        model_output = chat_message.content

        # extract querys
        try:
            queries = model_output.split('<begin>')[1].strip()
            queries = queries.split("\n")[:n_rollout]
        except:
            queries = []

        queries.append(query)
        return queries
    

    def query_reflect(self, origin_query: str):
        messages = []
        # Add System Prompt
        if self.query_reflect_prompt != "":
            messages += self._pack_message(role='system', content=self.query_reflect_prompt)
        # Prepare Query message
        query_message = "Now you will receive a search query, please help me revise it and output strictly with Output Format." \
        f"The original search query is {origin_query}"

        messages += self._pack_message(role="user", content=query_message)
        # Response
        chat_message :ChatMessage = self.model(
                messages = messages
            )
        model_output = chat_message.content

        try:
            result = json.loads(model_output)
            analysis_info = result['Analysis']
            augmented_query = result['Augmented Query']
        except Exception as e:
            print(e)
            return "", origin_query

        return analysis_info, augmented_query
