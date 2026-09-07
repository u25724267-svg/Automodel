#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-automodel@sha256:7213eab8055a2029ce1ef9022384a780f9095b9577f32edca4c48d303907421f}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/ext_data/casper_neo/Casper/kseries-next-run}"
GPU="${GPU:-0}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"

ACTION="${1:-status}"
CONTAINER_NAME="gemma4-e2b-k10-p2-nested-50m-r16-2ep-nvidia-lr-v1"
RECIPE_REL="examples/vlm_finetune/gemma4/gemma4_e2b_k10_p2_nested_50m_r16_2ep_nvidia_lr.yaml"
RECIPE_SHA="bcde66563ef4f8ac8e92e14d9b90bde41ee30c2c884fdbda154dabbe77f747c5"
DATA_AUDIT_SHA="12ebd9e7ce63f7c14da8db473dae49fc9f338b06f83e617cc61b6fe62c681acd"
EXPECTED_STEPS=3374
EXPECTED_WARMUP=337

DATA_HOST="$EXPERIMENT_ROOT/data/k10-sft"
DATA_CONTAINER="/data/gemma4-k10"
MIXTURE_NAME="mixture-p2-nested-50m-v1"
CHECKPOINT_ROOT="$EXPERIMENT_ROOT/checkpoints"
CHECKPOINT_REL="gemma4-e2b-k10/p2-nested-50m-r16-2ep-nvidia-lr-v1"
CHECKPOINT_HOST="$CHECKPOINT_ROOT/$CHECKPOINT_REL"
HF_CACHE="$EXPERIMENT_ROOT/hf-cache"
WANDB_ROOT="$EXPERIMENT_ROOT/wandb"
ENV_FILE="$REPO_ROOT/.env"
RECIPE_HOST="$REPO_ROOT/$RECIPE_REL"
DATA_AUDIT_HOST="$DATA_HOST/$MIXTURE_NAME/final_audit.json"
SNAPSHOT_DIR="$CHECKPOINT_HOST/launch-artifacts"

usage() {
    echo "Usage: launch_k10_p2_nested_50m_c2.sh <create|snapshot|start|launch|status|logs>" >&2
}

case "$ACTION" in
    create|snapshot|start|launch|status|logs) ;;
    *) usage; exit 2 ;;
esac

require_path() {
    if [[ ! -e "$1" ]]; then
        echo "Required path does not exist: $1" >&2
        exit 1
    fi
}

verify_environment() {
    require_path "$ENV_FILE"
    require_path "$HF_CACHE"
    require_path "$RECIPE_HOST"
    require_path "$DATA_HOST/$MIXTURE_NAME/train_meta.json"
    require_path "$DATA_HOST/$MIXTURE_NAME/validation_meta.json"
    require_path "$DATA_AUDIT_HOST"

    local actual_recipe_sha actual_audit_sha
    actual_recipe_sha="$(sha256sum "$RECIPE_HOST" | cut -d' ' -f1)"
    actual_audit_sha="$(sha256sum "$DATA_AUDIT_HOST" | cut -d' ' -f1)"
    if [[ "$actual_recipe_sha" != "$RECIPE_SHA" ]]; then
        echo "Recipe hash mismatch: $actual_recipe_sha" >&2
        exit 1
    fi
    if [[ "$actual_audit_sha" != "$DATA_AUDIT_SHA" ]]; then
        echo "Data audit hash mismatch: $actual_audit_sha" >&2
        exit 1
    fi

    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    if [[ -z "${WANDB_API_KEY:-}" || -z "${HF_TOKEN:-}" ]]; then
        echo "HF/W&B environment is incomplete" >&2
        exit 1
    fi
    if [[ "${WANDB_ENTITY:-}" != "dsfsi" ]]; then
        echo "WANDB_ENTITY must be dsfsi" >&2
        exit 1
    fi
    if [[ "${WANDB_PROJECT:-}" != "gemma4-african-instruction" ]]; then
        echo "WANDB_PROJECT must be gemma4-african-instruction" >&2
        exit 1
    fi
    if [[ "${WANDB_MODE:-}" != "online" ]]; then
        echo "WANDB_MODE must be online" >&2
        exit 1
    fi
    if [[ "${WANDB_DIR:-}" != "/logs/wandb" ]]; then
        echo "WANDB_DIR must be /logs/wandb" >&2
        exit 1
    fi
    ENV_FILE_SHA="$(sha256sum "$ENV_FILE" | cut -d' ' -f1)"
}

snapshot_launch_artifacts() {
    mkdir -p "$SNAPSHOT_DIR"
    cp "$RECIPE_HOST" "$SNAPSHOT_DIR/recipe.yaml"
    cp "$REPO_ROOT/examples/vlm_finetune/gemma4/launch_k10_p2_nested_50m_c2.sh" "$SNAPSHOT_DIR/"
    cp "$REPO_ROOT/examples/vlm_finetune/gemma4/K10_P2_NESTED_50M_DATA_PREREGISTRATION.md" "$SNAPSHOT_DIR/"
    cp "$REPO_ROOT/examples/vlm_finetune/gemma4/K10_P2_NESTED_50M_DATA_REPORT.md" "$SNAPSHOT_DIR/"
    cp "$DATA_AUDIT_HOST" "$SNAPSHOT_DIR/data_final_audit.json"
    git -C "$REPO_ROOT" --no-pager diff --binary >"$SNAPSHOT_DIR/source.patch"
    {
        echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "image=$IMAGE"
        echo "container=$CONTAINER_NAME"
        echo "gpu=$GPU"
        echo "allow_busy_gpu=$ALLOW_BUSY_GPU"
        echo "expected_optimizer_steps=$EXPECTED_STEPS"
        echo "expected_warmup_steps=$EXPECTED_WARMUP"
        echo "git_head=$(git -C "$REPO_ROOT" rev-parse HEAD)"
        echo "git_status_begin"
        git -C "$REPO_ROOT" status --short
        echo "git_status_end"
        sha256sum \
            "$SNAPSHOT_DIR/recipe.yaml" \
            "$SNAPSHOT_DIR/launch_k10_p2_nested_50m_c2.sh" \
            "$SNAPSHOT_DIR/K10_P2_NESTED_50M_DATA_PREREGISTRATION.md" \
            "$SNAPSHOT_DIR/K10_P2_NESTED_50M_DATA_REPORT.md" \
            "$SNAPSHOT_DIR/data_final_audit.json" \
            "$SNAPSHOT_DIR/source.patch"
    } >"$SNAPSHOT_DIR/provenance.txt"
}

prepare_checkpoint_directory() {
    if mkdir -p "$CHECKPOINT_HOST" 2>/dev/null; then
        return
    fi
    docker run --rm \
        -v "$EXPERIMENT_ROOT:/experiment" \
        "$IMAGE" \
        /bin/bash -c 'install -d -o "$1" -g "$2" "/experiment/checkpoints/$3"' \
        _ "$(id -u)" "$(id -g)" "$CHECKPOINT_REL"
}

create_run() {
    verify_environment
    if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        echo "Container already exists: $CONTAINER_NAME" >&2
        exit 1
    fi
    if [[ -d "$CHECKPOINT_HOST" ]] && find "$CHECKPOINT_HOST" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        echo "Checkpoint root is not empty: $CHECKPOINT_HOST" >&2
        exit 1
    fi
    prepare_checkpoint_directory
    mkdir -p "$WANDB_ROOT"

    docker create \
        --name "$CONTAINER_NAME" \
        --label "automodel.env-sha=$ENV_FILE_SHA" \
        --label "automodel.expected-steps=$EXPECTED_STEPS" \
        --label "automodel.expected-warmup=$EXPECTED_WARMUP" \
        --gpus "device=$GPU" \
        --network host \
        --ipc host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --log-opt max-size=100m \
        --log-opt max-file=5 \
        --env-file "$ENV_FILE" \
        -v "$REPO_ROOT:/opt/Automodel:ro" \
        -v "$DATA_HOST:$DATA_CONTAINER:ro" \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$CHECKPOINT_ROOT:/checkpoints" \
        -v "$WANDB_ROOT:/logs/wandb" \
        -w /opt/Automodel \
        "$IMAGE" \
        /opt/venv/bin/automodel "$RECIPE_REL" --nproc-per-node 1 --wandb.enable true
    snapshot_launch_artifacts
    echo "Created $CONTAINER_NAME on GPU $GPU; expected steps=$EXPECTED_STEPS warmup=$EXPECTED_WARMUP"
}

start_run() {
    verify_environment
    if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        echo "Container does not exist: $CONTAINER_NAME" >&2
        exit 1
    fi
    local state container_env_sha
    state="$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}')"
    if [[ "$state" != "created" ]]; then
        echo "Container state must be created, got $state" >&2
        exit 1
    fi
    container_env_sha="$(docker inspect "$CONTAINER_NAME" --format '{{index .Config.Labels "automodel.env-sha"}}')"
    if [[ "$container_env_sha" != "$ENV_FILE_SHA" ]]; then
        echo "Container environment hash differs from current .env" >&2
        exit 1
    fi
    mapfile -t gpu_pids < <(
        nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
    )
    if ((${#gpu_pids[@]})) && [[ "$ALLOW_BUSY_GPU" != "1" ]]; then
        echo "GPU $GPU is busy with PIDs: ${gpu_pids[*]}" >&2
        exit 1
    fi
    if ((${#gpu_pids[@]})); then
        echo "WARNING: starting on busy GPU $GPU by explicit override; PIDs: ${gpu_pids[*]}" >&2
    fi
    docker start "$CONTAINER_NAME"
    echo "Started $CONTAINER_NAME detached on GPU $GPU"
}

snapshot_run() {
    verify_environment
    if [[ "$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}')" != "created" ]]; then
        echo "Snapshot refresh requires created container state" >&2
        exit 1
    fi
    snapshot_launch_artifacts
    echo "Refreshed launch artifacts for $CONTAINER_NAME"
}

status_run() {
    if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        echo "container=absent gpu=$GPU"
        return
    fi
    docker inspect "$CONTAINER_NAME" --format \
        'container={{.Name}} status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}'
    docker logs --tail 12 "$CONTAINER_NAME" 2>&1 || true
}

case "$ACTION" in
    create) create_run ;;
    snapshot) snapshot_run ;;
    start) start_run ;;
    launch) create_run; start_run ;;
    status) status_run ;;
    logs) docker logs --tail 100 --follow "$CONTAINER_NAME" ;;
esac