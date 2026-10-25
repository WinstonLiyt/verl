# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import pytest
import torch
from transformers import ProcessorMixin

from verl.utils.tokenizer.tokenizer import _check_special_mm_tokens_on_lists

IMAGE_TOKEN_ID = 7


def _processor():
    return SimpleNamespace(image_token="<image>", image_token_id=IMAGE_TOKEN_ID)


def test_counts_match_on_tensor_input_ids():
    input_ids = torch.tensor([[1, IMAGE_TOKEN_ID, 2, IMAGE_TOKEN_ID], [1, 2, 3, 4]])
    text = ["a <image> b <image>", "text only"]
    _check_special_mm_tokens_on_lists(_processor(), text, {"input_ids": input_ids}, ["image"])


def test_count_mismatch_still_raises():
    # Truncation dropped one image token from the first sample.
    input_ids = torch.tensor([[1, IMAGE_TOKEN_ID, 2]])
    with pytest.raises(ValueError, match="Mismatch in `image` token count"):
        _check_special_mm_tokens_on_lists(_processor(), ["<image> <image>"], {"input_ids": input_ids}, ["image"])


def test_upstream_check_receives_python_lists(monkeypatch):
    seen = []

    def record(self, text, text_inputs, modalities):
        seen.append(text_inputs["input_ids"])

    monkeypatch.setattr(ProcessorMixin, "_check_special_mm_tokens", record)
    input_ids = torch.tensor([[1, IMAGE_TOKEN_ID]])
    _check_special_mm_tokens_on_lists(_processor(), ["<image>"], {"input_ids": input_ids}, ["image"])
    assert seen == [[[1, IMAGE_TOKEN_ID]]]
