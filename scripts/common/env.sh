#!/usr/bin/env bash
# Load project-level environment variables from .env
# Usage: source "$PROJECT_ROOT/scripts/common/env.sh"

: "${PROJECT_ROOT:?PROJECT_ROOT must be set before sourcing env.sh}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

export VERL_ROOT="${VERL_ROOT:-$PROJECT_ROOT/verl}"
export DATASETS_ROOT="${DATASETS_ROOT:-$PROJECT_ROOT/datasets}"
export REWARD_ROOT="${REWARD_ROOT:-$PROJECT_ROOT/reward}"
export TEMPLATES_ROOT="${TEMPLATES_ROOT:-$PROJECT_ROOT/templates}"
