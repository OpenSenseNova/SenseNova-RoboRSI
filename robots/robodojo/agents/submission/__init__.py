"""Submitted policy with narrow import aliases for its unchanged robot tools."""


def install_tool_imports():
    """Resolve frozen module names only when the Linux robot tools are loaded."""
    import sys

    from robots.robodojo import contracts

    from . import runtime_records

    # Register only the two leaves named by frozen tools, retaining the current
    # Runner/framework for all other imports and the Linux-only journal owner.
    for name, module in (
        ("XPolicyLab.policy.SenseNova-RoboRSI.runtime_records", runtime_records),
        ("XPolicyLab.utils.process_data", contracts),
    ):
        if name in sys.modules and sys.modules[name] is not module:
            raise ImportError("Another policy already owns the legacy tool import: " + name)
        sys.modules[name] = module
