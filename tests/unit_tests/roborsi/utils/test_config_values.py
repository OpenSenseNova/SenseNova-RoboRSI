# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from roborsi.agents import composition
from roborsi.utils.config_values import SECRET_KEYS, json_value, reject_secrets


def test_helpers_keep_legacy_identity():
    assert composition.json_value is json_value
    assert composition.reject_secrets is reject_secrets
    assert composition.SECRET_KEYS is SECRET_KEYS
    assert (
        json_value({"b": [1, True, None], "a": "ok"})
        == '{"a": "ok", "b": [1, true, null]}'
    )


@pytest.mark.parametrize("value", [float("nan"), object(), (1, 2), {1: "bad"}])
def test_invalid_json_keeps_error(value):
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        json_value(value)


def test_rejects_nested_secrets():
    with pytest.raises(ValueError, match="environment"):
        reject_secrets({"nested": [{"authorization": "private"}]})
