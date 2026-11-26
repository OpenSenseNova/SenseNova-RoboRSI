"""Merge accepted geometry into the current task's compact spatial memory."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile


def load(path):
    memory = json.loads(Path(path).read_text())
    if (memory.get("schema_version") != 1 or not memory.get("task")
            or memory.get("frame") != "environment_origin"
            or memory.get("units") != "meters" or not isinstance(memory.get("objects"), dict)):
        raise ValueError("Expected the current task's environment_origin spatial memory")
    return memory


def save(path, memory):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Spatial memory must be a regular workspace file")
    payload = json.dumps(memory, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=".space-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def import_result(memory, result, *, evidence, step):
    if result.get("frame") != memory["frame"] or result.get("units") != memory["units"]:
        raise ValueError("Localization frame/units must match memory; transform first")
    if type(step) is not int or step < 0:
        raise ValueError("Native control step must be a nonnegative integer")
    for row in result["points"]:
        obj, point = row["object_id"], row["id"]
        if not isinstance(obj, str) or not obj or not isinstance(point, str) or not point:
            raise ValueError("Stable object and point IDs are required")
        points = memory["objects"].setdefault(obj, {})
        estimate = row["estimate"]
        if estimate.get("accepted") is not True:
            if isinstance(points.get(point), dict):
                points[point]["status"] = "stale"
            continue
        position = estimate.get("position_base_m")
        if (not isinstance(position, list) or len(position) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in position)):
            raise ValueError("Accepted estimate requires finite XYZ coordinates")
        points[point] = {"description": row["description"], "position_m": position,
                         "status": "localized", "control_step": step, "evidence": evidence}
        # Keep the solver's uncertainty, when available, without storing its full report.
        for key in ("uncertainty", "uncertainty_m", "position_sigma_m", "conditional_position_sigma_m", "largest_axis_sigma_m"):
            if key in estimate:
                points[point][key] = estimate[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("show", "import", "stale"))
    parser.add_argument("--memory", type=Path, default=Path("space.json"))
    parser.add_argument("--result", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--object")
    args = parser.parse_args()
    memory = load(args.memory)
    if args.operation == "import":
        if args.result is None or args.step is None:
            parser.error("import requires --result and --step")
        import_result(memory, json.loads(args.result.read_text()), evidence=str(args.result), step=args.step)
    elif args.operation == "stale":
        if args.object is None:
            parser.error("stale requires --object")
        points = memory["objects"].get(args.object, {})
        if not isinstance(points, dict):
            raise ValueError("Spatial object must contain named point records")
        for point in points.values():
            # Historical model-written notes may coexist with point records.
            # They are not coordinates and must not break stale marking.
            if isinstance(point, dict) and isinstance(point.get("position_m"), list):
                point["status"] = "stale"
    if args.operation != "show":
        save(args.memory, memory)
    print(json.dumps(memory, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
