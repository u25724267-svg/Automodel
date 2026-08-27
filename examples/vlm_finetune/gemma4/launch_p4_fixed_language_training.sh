#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-automodel@sha256:7213eab8055a2029ce1ef9022384a780f9095b9577f32edca4c48d303907421f}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/ext_data/casper_neo/Casper/kseries-next-run}"
K10_GPU="${K10_GPU:-0}"
K14_GPU="${K14_GPU:-1}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
EPOCHS="${EPOCHS:-1}"

ACTION="${1:-status}"
ARM="${2:-all}"

usage() {
    cat <<'EOF'
Usage: launch_p4_fixed_language_training.sh <create|snapshot|start|launch|status|logs> [k10|k14|all]

Actions:
  create  Create detached Docker container(s) and snapshot launch artifacts.
    snapshot  Refresh launch artifacts for container(s) that are still stopped.
  start   Start already-created container(s); refuses GPUs with active processes.
  launch  Create then start container(s).
  status  Show container state, OOM status, GPU assignment, and latest logs.
  logs    Follow one arm's logs (k10 or k14).

Environment overrides:
    REPO_ROOT, EXPERIMENT_ROOT, IMAGE, K10_GPU, K14_GPU, ALLOW_BUSY_GPU,
    EPOCHS (1 or 2; default 1)

For an exact reproduction, do not override IMAGE or recipe paths. W&B is
enabled by the fixed CLI override recorded in this script because repository
example YAMLs must keep W&B opt-in (`enable: false`).
EOF
}

case "$ACTION" in
    create|snapshot|start|launch|status|logs) ;;
    *) usage; exit 2 ;;
esac
case "$ARM" in
    k10|k14|all) ;;
    *) usage; exit 2 ;;
esac
if [[ "$EPOCHS" != "1" && "$EPOCHS" != "2" ]]; then
    echo "EPOCHS must be 1 or 2" >&2
    exit 2
fi
if [[ "$ACTION" == "logs" && "$ARM" == "all" ]]; then
    echo "logs requires one arm: k10 or k14" >&2
    exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required" >&2
    exit 1
fi

DATA_ROOT="$EXPERIMENT_ROOT/data"
CHECKPOINT_ROOT="$EXPERIMENT_ROOT/checkpoints"
HF_CACHE="$EXPERIMENT_ROOT/hf-cache"
WANDB_ROOT="$EXPERIMENT_ROOT/wandb"
ENV_FILE="$REPO_ROOT/.env"

require_path() {
    if [[ ! -e "$1" ]]; then
        echo "Required path does not exist: $1" >&2
        exit 1
    fi
}

arm_values() {
    local arm="$1"
    if [[ "$arm" == "k10" ]]; then
        GPU="$K10_GPU"
        DATA_HOST="$DATA_ROOT/k10-sft"
        DATA_CONTAINER="/data/gemma4-k10"
        MIXTURE_NAME="mixture-p4-5m-per-language-v2"
        EXPECTED_AUDIT_SHA="27e81495c51619920bbbd1ac58219d90949543e7e0aadb788cce8d56dca1cb3b"
        if [[ "$EPOCHS" == "1" ]]; then
            CONTAINER_NAME="gemma4-e2b-k10-p4-5m-per-language-r16-1ep-v1"
            RECIPE_REL="examples/vlm_finetune/gemma4/gemma4_e2b_k10_p4_5m_r16_1ep_nvidia_lr.yaml"
            CHECKPOINT_REL="gemma4-e2b-k10/p4-5m-per-language-r16-1ep-v1"
            EXPECTED_RECIPE_SHA="10988900d83a81bfaee6dd5753caf6a74f968ada26a396f0223ed036564b895c"
            EXPECTED_STEPS=1681
            EXPECTED_WARMUP=168
        else
            CONTAINER_NAME="gemma4-e2b-k10-p4-5m-per-language-r16-2ep-v1"
            RECIPE_REL="examples/vlm_finetune/gemma4/gemma4_e2b_k10_p4_5m_r16_2ep_nvidia_lr.yaml"
            CHECKPOINT_REL="gemma4-e2b-k10/p4-5m-per-language-r16-2ep-v1"
            EXPECTED_RECIPE_SHA="f7b491be7649866ba8eb92f9fcc07d53aa9c630ea12f8bbc7f736270680a7a0e"
            EXPECTED_STEPS=3362
            EXPECTED_WARMUP=336
        fi
    else
        GPU="$K14_GPU"
        DATA_HOST="$DATA_ROOT/k14-sft"
        DATA_CONTAINER="/data/gemma4-k14"
        MIXTURE_NAME="mixture-p4-5m-per-language-v1"
        EXPECTED_AUDIT_SHA="a7a70645b094560f530f48b3db8f4bfce105fac3bea357109324b95fb16e7594"
        if [[ "$EPOCHS" == "1" ]]; then
            CONTAINER_NAME="gemma4-e2b-k14-p4-5m-per-language-r16-1ep-v1"
            RECIPE_REL="examples/vlm_finetune/gemma4/gemma4_e2b_k14_p4_5m_r16_1ep_nvidia_lr.yaml"
            CHECKPOINT_REL="gemma4-e2b-k14/p4-5m-per-language-r16-1ep-v1"
            EXPECTED_RECIPE_SHA="71769f694a0dc3edc54094503a297858de6dac79697f0c380358c05c746eddf7"
            EXPECTED_STEPS=2354
            EXPECTED_WARMUP=235
        else
            CONTAINER_NAME="gemma4-e2b-k14-p4-5m-per-language-r16-2ep-v1"
            RECIPE_REL="examples/vlm_finetune/gemma4/gemma4_e2b_k14_p4_5m_r16_2ep_nvidia_lr.yaml"
            CHECKPOINT_REL="gemma4-e2b-k14/p4-5m-per-language-r16-2ep-v1"
            EXPECTED_RECIPE_SHA="8d0e2b8ef2e4211445e4f486d5ec1a7af9426409661c2b90b92c612e750f4c36"
            EXPECTED_STEPS=4708
            EXPECTED_WARMUP=470
        fi
    fi
    RECIPE_HOST="$REPO_ROOT/$RECIPE_REL"
    CHECKPOINT_HOST="$CHECKPOINT_ROOT/$CHECKPOINT_REL"
    DATA_AUDIT_HOST="$DATA_HOST/$MIXTURE_NAME/final_audit.json"
}

verify_environment() {
    require_path "$ENV_FILE"
    require_path "$HF_CACHE"
    require_path "$RECIPE_HOST"
    require_path "$DATA_HOST/$MIXTURE_NAME/train_meta.json"
    require_path "$DATA_HOST/$MIXTURE_NAME/validation_meta.json"
    require_path "$DATA_AUDIT_HOST"
    "$REPO_ROOT/examples/vlm_finetune/gemma4/materialize_p4_fixed_language_mixtures.sh" audit "$arm"
    local actual_audit_sha
    actual_audit_sha="$(sha256sum "$DATA_AUDIT_HOST" | cut -d' ' -f1)"
    if [[ "$actual_audit_sha" != "$EXPECTED_AUDIT_SHA" ]]; then
        echo "Data audit hash mismatch for $arm: $actual_audit_sha" >&2
        exit 1
    fi
    local actual_recipe_sha
    actual_recipe_sha="$(sha256sum "$RECIPE_HOST" | cut -d' ' -f1)"
    if [[ "$actual_recipe_sha" != "$EXPECTED_RECIPE_SHA" ]]; then
        echo "Recipe hash mismatch for $arm: $actual_recipe_sha" >&2
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
    local snapshot_dir="$CHECKPOINT_HOST/launch-artifacts"
    mkdir -p "$snapshot_dir"
    cp "$RECIPE_HOST" "$snapshot_dir/recipe.yaml"
    cp "$REPO_ROOT/examples/vlm_finetune/gemma4/launch_p4_fixed_language_training.sh" "$snapshot_dir/"
    cp "$REPO_ROOT/examples/vlm_finetune/gemma4/KSERIES_FIXED_LANGUAGE_DATA_REPRODUCTION.md" "$snapshot_dir/"
    cp "$REPO_ROOT/examples/vlm_finetune/gemma4/KSERIES_FIXED_LANGUAGE_TRAINING.md" "$snapshot_dir/"
    cp "$DATA_AUDIT_HOST" "$snapshot_dir/data_final_audit.json"
    git -C "$REPO_ROOT" --no-pager diff --binary >"$snapshot_dir/source.patch"
    {
        echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "image=$IMAGE"
        echo "container=$CONTAINER_NAME"
        echo "gpu=$GPU"
        echo "epochs=$EPOCHS"
        echo "allow_busy_gpu=$ALLOW_BUSY_GPU"
        echo "expected_optimizer_steps=$EXPECTED_STEPS"
        echo "expected_warmup_steps=$EXPECTED_WARMUP"
        echo "git_head=$(git -C "$REPO_ROOT" rev-parse HEAD)"
        echo "git_status_begin"
        git -C "$REPO_ROOT" status --short
        echo "git_status_end"
        sha256sum \
            "$snapshot_dir/recipe.yaml" \
            "$snapshot_dir/launch_p4_fixed_language_training.sh" \
            "$snapshot_dir/KSERIES_FIXED_LANGUAGE_DATA_REPRODUCTION.md" \
            "$snapshot_dir/KSERIES_FIXED_LANGUAGE_TRAINING.md" \
            "$snapshot_dir/data_final_audit.json" \
            "$snapshot_dir/source.patch"
    } >"$snapshot_dir/provenance.txt"
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

create_arm() {
    local arm="$1"
    arm_values "$arm"
    verify_environment
    if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        echo "Container already exists: $CONTAINER_NAME" >&2
        exit 1
    fi
    if [[ -e "$CHECKPOINT_HOST" ]]; then
        local unexpected_checkpoint_entry
        unexpected_checkpoint_entry="$(find "$CHECKPOINT_HOST" -mindepth 1 -maxdepth 1 ! -name launch-artifacts -print -quit)"
        if [[ -n "$unexpected_checkpoint_entry" ]]; then
            echo "Checkpoint root contains training output: $unexpected_checkpoint_entry" >&2
            exit 1
        fi
    fi
    prepare_checkpoint_directory
    mkdir -p "$WANDB_ROOT"

    docker create \
        --name "$CONTAINER_NAME" \
        --label "automodel.env-sha=$ENV_FILE_SHA" \
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

start_arm() {
    local arm="$1"
    arm_values "$arm"
    verify_environment
    if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        echo "Container does not exist: $CONTAINER_NAME" >&2
        exit 1
    fi
    local state
    state="$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}')"
    if [[ "$state" != "created" ]]; then
        echo "Container state must be created, got $state: $CONTAINER_NAME" >&2
        exit 1
    fi
    local container_env_sha
    container_env_sha="$(docker inspect "$CONTAINER_NAME" --format '{{index .Config.Labels "automodel.env-sha"}}')"
    if [[ "$container_env_sha" != "$ENV_FILE_SHA" ]]; then
        echo "Container environment hash differs from current .env: $CONTAINER_NAME" >&2
        exit 1
    fi
    mapfile -t gpu_pids < <(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
    if ((${#gpu_pids[@]})) && [[ "$ALLOW_BUSY_GPU" != "1" ]]; then
        echo "GPU $GPU is busy with PIDs: ${gpu_pids[*]}; refusing to start $CONTAINER_NAME" >&2
        exit 1
    fi
    if ((${#gpu_pids[@]})); then
        echo "WARNING: starting $CONTAINER_NAME on busy GPU $GPU by explicit override; existing PIDs: ${gpu_pids[*]}" >&2
    fi
    docker start "$CONTAINER_NAME"
    echo "Started $CONTAINER_NAME detached on GPU $GPU"
}

snapshot_arm() {
    local arm="$1"
    arm_values "$arm"
    verify_environment
    if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        echo "Container does not exist: $CONTAINER_NAME" >&2
        exit 1
    fi
    local state
    state="$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}')"
    if [[ "$state" != "created" ]]; then
        echo "Snapshot refresh requires created state, got $state: $CONTAINER_NAME" >&2
        exit 1
    fi
    snapshot_launch_artifacts
    echo "Refreshed launch artifacts for $CONTAINER_NAME"
}

status_arm() {
    local arm="$1"
    arm_values "$arm"
    if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        echo "$arm container=absent gpu=$GPU"
        return
    fi
    docker inspect "$CONTAINER_NAME" --format \
        "$arm container={{.Name}} status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}} finished={{.State.FinishedAt}} gpu=$GPU"
    docker logs --tail 8 "$CONTAINER_NAME" 2>&1 || true
}

logs_arm() {
    local arm="$1"
    arm_values "$arm"
    docker logs --tail 100 --follow "$CONTAINER_NAME"
}

for_each_arm() {
    local function_name="$1"
    if [[ "$ARM" == "all" ]]; then
        "$function_name" k10
        "$function_name" k14
    else
        "$function_name" "$ARM"
    fi
}

case "$ACTION" in
    create) for_each_arm create_arm ;;
    snapshot) for_each_arm snapshot_arm ;;
    start) for_each_arm start_arm ;;
    launch)
        for_each_arm create_arm
        for_each_arm start_arm
        ;;
    status) for_each_arm status_arm ;;
    logs) logs_arm "$ARM" ;;
esac