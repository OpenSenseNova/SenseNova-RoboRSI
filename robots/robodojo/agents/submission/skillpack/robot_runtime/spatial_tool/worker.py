"""Private subagent entry point. Stdin request, one compact stdout result, exit."""
import json
import sys
from .contracts import LocalizationError, failure, public_result, tool_spec
from .localizer import pixel_localize
from .snapshot import Snapshot
from .validation import Limits


def run(payload):
    request = payload["request"]
    try:
        from jsonschema import Draft202012Validator, ValidationError
        Draft202012Validator(tool_spec(payload["mode"])["input_schema"]).validate(request)
        config = payload["config"]
        limits = Limits.parse(config.get("limits"))
        snapshot = Snapshot.load(payload["descriptor"], payload["image_root"], limits)
        value = pixel_localize(request, snapshot, limits)
        return public_result(value, request)
    except LocalizationError as error:
        return failure(request, error.status, error.reason)
    except ValidationError:
        return failure(request, "invalid_geometry", "invalid_request_parameters")
    except (KeyError, TypeError, ValueError, OSError):
        return failure(request, "invalid_geometry", "invalid_localization_input")
    except Exception:
        # Solver implementation errors must not expose internal context.
        return failure(request, "low_confidence", "localization_processing_failed")


def main():
    request = {}
    try:
        raw = sys.stdin.buffer.read(2*1024*1024+1)
        if len(raw) > 2*1024*1024:
            raise ValueError("oversized request")
        payload = json.loads(raw)
        request = payload.get("request", {})
        result = run(payload)
    except Exception:
        result = failure(request if isinstance(request, dict) else {}, "invalid_geometry", "invalid_worker_request")
    sys.stdout.write(json.dumps(result, allow_nan=False, separators=(",", ":"))+"\n")


if __name__ == "__main__":
    main()
