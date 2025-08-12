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

import os
import json
from smolagents.models import (
    OpenAIServerModel,
    FakeToolCallOpenAIServerModel,
)

def process_selected_tasks_param(tasks_param):

    if not tasks_param:
        return []
    if len(tasks_param) == 1:
        item = tasks_param[0]
        if os.path.isfile(item):
            if item.endswith('.jsonl'):
                with open(item, 'r', encoding='utf-8') as f:
                    return [json.loads(line)['id'] for line in f if 'id' in json.loads(line)]
            else:
                with open(item, 'r') as f:
                    return [line.strip() for line in f if line.strip()]

        try:
            return [int(item)]
        except ValueError:
            return [item]

    result = []
    for item in tasks_param:
        try:
            result.append(int(item))
        except ValueError:
            result.append(item)

    return result

def prepare_model_kwargs(model_id, args):
    kwargs = {}
    if 'o1' in model_id or 'o3' in model_id or 'o4' in model_id:
        kwargs["reasoning_effort"] = "high"
    if args.temperature:
        kwargs["temperature"] = args.temperature
    if args.top_p:
        kwargs["top_p"] = args.top_p
    return kwargs

def get_api_model(model_id):

    model_wrapper = OpenAIServerModel
    key = os.getenv("OPENAI_API_KEY")
    url = os.getenv("OPENAI_BASE_URL")
    if "deepseek" in model_id.lower() or 'claude' in model_id.lower():
        model_wrapper = FakeToolCallOpenAIServerModel

    return model_id, key, url, model_wrapper