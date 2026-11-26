"""Provide the original Simate policy's three imports in its isolated process."""

import importlib
import sys
from types import ModuleType


def load_model():
    """Load byte-pinned Simate code without installing an XPolicyLab checkout."""
    prefix = "robots.robodojo.agents.submission"
    root = ModuleType("XPolicyLab")
    root.__path__ = []
    utils = ModuleType("XPolicyLab.utils")
    utils.__path__ = []
    process = ModuleType("XPolicyLab.utils.process_data")
    from robots.robodojo.contracts import get_robot_action_dim_info

    process.get_robot_action_dim_info = get_robot_action_dim_info
    modules = {"XPolicyLab": root, "XPolicyLab.utils": utils,
               "XPolicyLab.model_template": importlib.import_module(prefix + ".model_template"),
               "XPolicyLab.utils.checkpoint_resolver": importlib.import_module(prefix + ".checkpoint_resolver"),
               "XPolicyLab.utils.process_data": process}
    for name, module in modules.items():
        if name in sys.modules and sys.modules[name] is not module:
            raise RuntimeError("Simate compatibility requires an isolated process")
        sys.modules[name] = module
    return importlib.import_module("robots.robodojo.agents.Simate_beta.model").Model
