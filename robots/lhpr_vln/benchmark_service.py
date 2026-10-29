# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""No-argument provider factory for the common Benchmark service host."""

from pathlib import Path

from roborsi.utils.config import get_project_env
from robots.vln_shared.benchmark import NavigationBenchmark, portable_provider


def make_benchmark() -> NavigationBenchmark:
    """Load an environment-owned catalog and its fixed physical action budget."""
    provider = portable_provider(
        Path(
            get_project_env(
                "ROBORSI_VLN_CATALOG", "/roborsi/assets/lhpr_vln/catalog.json"
            )
        ),
        int(get_project_env("ROBORSI_VLN_BUDGET", "500")),
    )
    if provider.spec["dataset"] != "lhpr_vln":
        raise ValueError("LHPR service requires an LHPR catalog")
    return provider
