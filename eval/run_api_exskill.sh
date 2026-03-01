#!/bin/bash

API_KEY_1=""

API_KEY_2=""

# ============================================================================
# Reasoning Model
# ============================================================================

export REASONING_MODEL_NAME=""

export REASONING_API_KEY=$API_KEY_1
export REASONING_END_POINT=""

# Optional: Fallback API for round-robin polling
export REASONING_API_KEY_2=$API_KEY_2
export REASONING_END_POINT_2=""

# ============================================================================
# Verifier Model
# ============================================================================
export VERIFIER_MODEL_NAME=""
export VERIFIER_API_KEY=$API_KEY_2
export VERIFIER_END_POINT=""

# ============================================================================
# Experience Model
# ============================================================================
export EXPERIENCE_MODEL_NAME=""

export EXPERIENCE_API_KEY=$API_KEY_2
export EXPERIENCE_END_POINT=""

# Optional: Fallback API for experience generation
export EXPERIENCE_EMBEDDING_MODEL="text-embedding-3-small"
export EXPERIENCE_API_KEY_2=$API_KEY_1
export EXPERIENCE_END_POINT_2=""

export EXPERIENCE_EMBEDDING_API_KEY=$API_KEY_2
export EXPERIENCE_EMBEDDING_ENDPOINT="" 

# ============================================================================
# Function Calling Configuration
# ============================================================================
export SERPAPI_KEY=""
export JINA_API_KEY=""

export ENABLE_FUNCTION_CALLING="true"

# Available tools: web_search, image_search, visit, code_interpreter, zoom
# export ENABLED_TOOLS="web_search, image_search, visit, code_interpreter"
export ENABLED_TOOLS="web_search, visit, code_interpreter"

TOOL_CONFIG_PATH="eval/configs/tool_configs.yaml"

IMAGE_SEARCH_MAX_CALLS=5
WEB_SEARCH_MAX_CALLS=7

# ============================================================================
# Inference Parameters
# ============================================================================

MAX_TOTAL_TOKENS=32768
MAX_TURNS=20
MAX_IMAGES=100
TEMPERATURE=0.6
TOP_P=1.0

# ============================================================================
# Experience Parameters
# ============================================================================

EXPERIENCE_MAX_OPS=3
EXPERIENCE_MAX_ITEMS=120

# Experience Retrieval Parameters
EXPERIENCE_RETRIEVAL_TOP_K=3
EXPERIENCE_LIBRARY="memory_bank/test/experiences.json"

# ============================================================================
# Skill Parameters
# ============================================================================

SKILL_LIBRARY="memory_bank/test/SKILL.md"
SKILL_MAX_LENGTH=1000

# ============================================================================
# Running Settings
# ============================================================================

# SYSTEM_PROMPT_TYPE="multi_tool_agent_search"
SYSTEM_PROMPT_TYPE="multi_tool_agent"
# SYSTEM_PROMPT_TYPE="multi_tool_agent_code"
# SYSTEM_PROMPT_TYPE="agent_zoom"
# SYSTEM_PROMPT_TYPE="direct_cot"

IMAGE_DIR="benchmark"
DATA_PATH="benchmark/VisualProbe_Test/val.json"

OUTPUT_DIR="output/test_exskill_1"
LOG_OUTPUT_DIR="logs/test_exskill_1"

NUM_WORKERS=8
EXPERIENCE_LARGE_BATCH=8
ROLLOUTS_PER_SAMPLE=2

MAX_SAMPLES="16"

# ============================================================================
# Run Inference
# ============================================================================

python3 -u eval/infer_api.py \
    --input-file $DATA_PATH \
    --image-folder $IMAGE_DIR \
    --output-dir $OUTPUT_DIR \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --max-turns $MAX_TURNS \
    --max-images $MAX_IMAGES \
    --max-total-tokens $MAX_TOTAL_TOKENS \
    --system-prompt-key $SYSTEM_PROMPT_TYPE \
    --num-workers $NUM_WORKERS \
    --tool-config-path $TOOL_CONFIG_PATH \
    --max-samples $MAX_SAMPLES \
    --rollouts-per-sample $ROLLOUTS_PER_SAMPLE \
    --image-search-max-calls $IMAGE_SEARCH_MAX_CALLS \
    --web-search-max-calls $WEB_SEARCH_MAX_CALLS \
    --skill-enable \
    --skill-library $SKILL_LIBRARY \
    --skill-inference \
    --experience-enable \
    --experience-library $EXPERIENCE_LIBRARY \
    --experience-retrieval \
    --experience-retrieval-top-k $EXPERIENCE_RETRIEVAL_TOP_K \
    --experience-retrieval-decomposition \
    --experience-retrieval-rewrite \
    --experience-online-generate \
    --experience-library-update \
    --experience-max-ops $EXPERIENCE_MAX_OPS \
    --experience-large-batch $EXPERIENCE_LARGE_BATCH \
    --experience-refine \
    --experience-max-items $EXPERIENCE_MAX_ITEMS \
    --skill-refine \
    --skill-max-length $SKILL_MAX_LENGTH \
    2>&1 | tee $LOG_OUTPUT_DIR.log