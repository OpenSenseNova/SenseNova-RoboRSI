#!/usr/bin/env bash
# Deployment placeholder for the model API used by evaluate_model.py.
# Replace this file with the team's verified deployment command when available.
# Keep model serving dependencies in a separate environment from the evaluator.
set -euo pipefail

# TODO: Start the model server in the foreground with an exec command.
# The deployment must define:
#   - the checkpoint / model path and required serving dependencies;
#   - the served model name (MODEL_NAME in the evaluation environment);
#   - the API origin and prefix (OPENAI_BASE_URL in the evaluation environment);
#   - authentication (OPENAI_API_KEY in the evaluation environment).
# It must accept images and tool calls using OpenAI-compatible Chat Completions
# or Responses. Hardware settings and launch flags depend on the serving stack.
# Do not put real credentials in this file.

cat >&2 <<'MESSAGE'
Model deployment is not configured yet. No API server has been started.
Replace scripts/deploy/deploy_model.sh with your verified serving command.
See scripts/deploy/README.md for the interface required by the evaluator.
If the model API is already running, skip this placeholder and run
scripts/evaluation/evaluate_model.py with its URL, API key and served model name.
MESSAGE
exit 1
