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

import re
from typing import Union, List
import os
import json

save_dir = 'workflow'
os.makedirs(save_dir, exist_ok=True)

class Step:

    __slots__ = ('index', 'description')

    def __init__(self, index: int, description: str):
        self.index = index
        self.description = description.strip()

    def __repr__(self):
        return f"Step({self.index}, '{self.description}')"

    def __str__(self):
        return f"{self.index}. {self.description}"


class Workflow:
    def __init__(self, steps: Union[str, List[Step]] = None, wf_name='gaia_validation'):
        self._steps = []
        self.load(steps)
        self.wf_name = wf_name

    def load(self, steps: Union[str, List[Step]] = None):
        if isinstance(steps, str):
            self._steps = self.load_from_str(steps)
        elif steps:
            self._steps = list(steps)

    def load_from_str(self, s: str):

        return self._parse_initial_str(s)

    def apply_update(self, s: str):

        new_steps = self._parse_update_str(s)
        if not new_steps:
            return

        start_num = new_steps[0].index
        max_allowed_start = len(self._steps) + 1

        if not (1 <= start_num <= max_allowed_start):
            raise ValueError(
                f"Initial step: {start_num}, out of range (1-{max_allowed_start})"
            )

        overlap_end = start_num + len(new_steps) - 1
        original_end = len(self._steps)

        if start_num <= original_end:
            self._steps = self._steps[:start_num-1] + new_steps
        else:
            self._steps += new_steps

        if start_num <= original_end and (overlap_end < original_end):
            print(f"Warning: step {overlap_end+1}-{original_end} trashed")

    @staticmethod
    def _parse_update_str(s: str) -> List[Step]:

        steps = []
        current_base = None
        for line in s.splitlines():
            if step := Workflow._parse_line(line):
                if not steps:
                    current_base = step.index
                adjusted_index = current_base + len(steps)
                if step.index != adjusted_index:
                    raise ValueError(
                        f"Update steps are not consecutive. Expected {adjusted_index}, got {step.index}"
                        "\nTip: Update blocks must form a continuous sequence"
                    )
                steps.append(Step(adjusted_index, step.description))
        return steps

    @staticmethod
    def _parse_initial_str(s: str) -> List[Step]:

        steps = []
        expected = 1
        for line in s.splitlines():
            if step := Workflow._parse_line(line):
                if step.index != expected:
                    raise ValueError(f"Steps are not consecutive. Expected {expected}, got {step.index}")
                expected += 1
                steps.append(step)
        return steps

    @staticmethod
    def _parse_line(line: str) -> Union[Step, None]:

        pattern = re.compile(
            r'^\s*[([{]?(\d+)[.)\]、}]\s*(.*)$',
            flags=re.UNICODE
        )
        line = line.strip()
        if match := pattern.match(line):
            return Step(int(match.group(1)), match.group(2))
        return None
    
    def load_from_file(self):
        wf_path = os.path.join(save_dir, f"{self.wf_name}.json")
        if os.path.exists(wf_path):
            with open(wf_path, 'r') as f:
                data = json.load(f)
            if self.task_id in data:
                self._steps = self.load_from_str(data[self.task_id]['workflow'])
            else:
                raise ValueError(f"task_id {self.task_id} not found in {wf_path}")
        else:
           raise FileNotFoundError

    def save_to_file(self, data_dict):
        wf_path = os.path.join(save_dir, f"{self.wf_name}.jsonl")
        with open(wf_path, 'a+', encoding='utf-8') as f:
            f.write(json.dumps(data_dict) + "\n")

    def __getitem__(self, index: int):
        return self._steps[index - 1]

    def __len__(self):
        return len(self._steps)

    def __repr__(self):
        return f"Workflow({self._steps})"

    def __str__(self):
        return "\n".join(
            f"{str(step)}"
            for step in self._steps
        )

