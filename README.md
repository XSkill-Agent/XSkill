# XSkill: Continual Learning from Experience and Skills in Multimodal Agents

<p align="center">
  <a href="https://arxiv.org/pdf/2603.12056">
    <img src="https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper">
  </a>
  <a href="https://github.com/XSkill-Agent/XSkill">
    <img src="https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://xskill-agent.github.io/xskill_page/">
    <img src="https://img.shields.io/badge/Webpage-blue?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Webpage">
  </a>
</p>

<!-- TODO: Add framework overview figure here -->
![XSkill Framework Overview](assets/framework.png)

Multimodal agents demonstrate impressive problem-solving capabilities but typically operate in isolated episodes without leveraging past experiences. **XSkill** addresses this by combining two complementary forms of accumulated knowledge: task-level **Skills** (structured workflows and tool templates) and action-level **Experiences** (context-specific tactical insights), both automatically extracted from agent trajectories without any parametric training.

XSkill operates in two phases. **Phase I (Accumulation)**: after each batch of rollouts, the agent distills structured skill documents and experience entries via visually-grounded trajectory summarization, cross-rollout critique, and hierarchical consolidation. **Phase II (Inference)**: for each test sample, the system decomposes the task, retrieves relevant knowledge from the memory bank, adapts it to the current visual context, and injects it into the system prompt.

Evaluated on diverse benchmarks (VisualToolBench, TIR-Bench, MMSearch-Plus, AgentVista, MMBrowseComp), XSkill achieves considerable performance gains over strong baselines across different backbone models, with superior zero-shot cross-task transferability.

---

## Repository Structure

```
XSkill/
├── eval/
│   ├── infer_api.py              # Main inference entry point
│   ├── infer_api_utils.py        # Utility functions for inference pipeline
│   ├── run_api_exskill.sh        # Reference run script
│   ├── configs/
│   │   └── tool_configs.yaml     # Per-tool runtime configuration
│   ├── engine/                   # API calling, tool dispatch, context management
│   ├── exskill/                  # Experience & skill learning core
│   │   ├── experience_critique.py
│   │   ├── experience_manager.py
│   │   ├── experience_retriever.py
│   │   ├── skill_builder.py
│   │   ├── trajectory_summary.py
│   │   └── multimodal_analysis.py
│   ├── tools/                    # Tool implementations
│   │   ├── code_interpreter.py
│   │   ├── web_search.py
│   │   ├── image_search.py
│   │   ├── visit.py
│   │   └── zoom.py
│   ├── prompts/                  # Prompt templates
│   ├── search/                   # Search tree & node structures
│   └── utils/                    # Shared utilities
├── memory_bank/                  # Experience library & skill document (created at runtime)
├── output/                       # Per-sample inference outputs
├── logs/                         # Run logs
└── requirements.txt
```

---

## Installation

**Python 3.11** is recommended.

```bash
git clone https://github.com/XSkill-Agent/XSkill.git
cd XSkill
pip install -r requirements.txt
```


---

## Configuration

Before running, you must fill in **two** configuration files.

### 1. `eval/run_api_exskill.sh` — API Keys and Model Endpoints

Open `eval/run_api_exskill.sh` and set the following variables:

```bash
# ── Reasoning Model (the main agent) ──────────────────────────────────────
export REASONING_MODEL_NAME=""
export REASONING_API_KEY=$API_KEY_1
export REASONING_END_POINT=""       # OpenAI-compatible endpoint URL

# Optional: second API key for round-robin fallback
export REASONING_API_KEY_2=$API_KEY_2
export REASONING_END_POINT_2=""

# ── Verifier Model (LLM-as-judge for scoring) ─────────────────────────────
export VERIFIER_MODEL_NAME=""
export VERIFIER_API_KEY=$API_KEY_2
export VERIFIER_END_POINT=""

# ── Experience Model (experience generation & skill building) ──────────────
export EXPERIENCE_MODEL_NAME=""
export EXPERIENCE_API_KEY=$API_KEY_2
export EXPERIENCE_END_POINT=""

# Embedding model for experience retrieval (OpenAI-compatible)
export EXPERIENCE_EMBEDDING_MODEL="text-embedding-3-small"
export EXPERIENCE_EMBEDDING_API_KEY=$API_KEY_2
export EXPERIENCE_EMBEDDING_ENDPOINT=""

# ── External Tool API Keys ─────────────────────────────────────────────────
export SERPAPI_KEY=""               # Required for web_search and image_search tool
export JINA_API_KEY=""              # Required for visit tool
```

> All models must expose an **OpenAI-compatible chat completion API**. The embedding endpoint must support the `/v1/embeddings` interface.

### 2. `eval/configs/tool_configs.yaml` — Per-Tool Runtime Settings

```yaml
# visit tool — webpage content fetching
visit:
  max_content_length: 150000   # Max characters to extract per page
  timeout: 120                 # HTTP request timeout (seconds)
  api_key: ""                  # Optional: API key for VLM-based page summarization
  api_endpoint: ""             # Optional: endpoint for the above
  model_name: ""               # Optional: model name for the above

# image_search tool — reverse/visual image search
image_search:
  imgbb_api_key: ""            # Required: ImgBB API key for image hosting
  max_results: 5               # Max search results to return
  search_image_max_pixels: 1000000
  search_image_quality: 85
```

> **Image hosting for reverse image search.** When the agent performs a reverse image search on a local image, the image must first be uploaded to a publicly accessible URL. `image_search.py` supports the following hosting services in priority order:
> 1. **ImgBB** (`imgbb_api_key`) — Recommended. Stable and reliable; requires a free API key from [imgbb.com](https://api.imgbb.com).
> 2. **cloudflareimg.cdn.sn** — No registration required; supports WebP compression automatically.
> 3. **0x0.st** — Anonymous upload service; no registration required.
> 4. **catbox.moe** — Anonymous backup service; no registration required.
>
> If `imgbb_api_key` is left empty, the tool automatically falls back to the anonymous services above. You can also add or replace hosting services by editing `eval/tools/image_search.py` (`_upload_local_image`).

---

## Data Format and Preparation

### Input Data Format

The benchmark data file (passed via `--input-file`) must be a **JSON file containing a list of sample objects**. Each sample supports the following fields:

```json
[
  {
    "doc_id": "sample_001",
    "problem": "What is shown in <image>? Describe the object in detail.",
    "images": ["relative/path/to/image.jpg"],
    "solution": "A red bicycle parked against a wall."
  },
  ...
]
```

| Field | Required | Description |
|---|---|---|
| `doc_id` or `question_id` | Required | Unique identifier for the sample |
| `problem` or `question` | Required | Question text. Use `<image>` as a placeholder to indicate where each image appears in the question |
| `images` | Optional | List of image file paths **relative to `--image-folder`**. The number of paths should match the number of `<image>` placeholders |
| `solution` | Optional | Ground truth answer string, used by the LLM-as-judge verifier for scoring |

**Notes:**
- If `<image>` placeholders are present in `problem`, images are injected in order of appearance.
- If no `<image>` placeholder is present but `images` is non-empty, all listed images are passed to the model.
- Text-only samples (no `images` field and no `<image>` placeholder) are also supported.
- The `solution` field can be omitted; samples without a ground truth will receive a score of `0.0`.

### Image Files

All image paths in the `images` field are resolved relative to `--image-folder`. For example:

```
--image-folder /data/benchmark
```

with `"images": ["VisualProbe/val/img_001.jpg"]` will load `/data/benchmark/VisualProbe/val/img_001.jpg`.



## Running

### 1. XSkill Accumulation (Phase I)

To autonomously accumulate structured experiences and skills from agent trajectories, run:

```bash
bash eval/run_exskill_train.sh
```

This script enables online experience generation and skill library updates.

### 2. Inference with XSkill (Phase II)

To evaluate the agent using the accumulated memory bank, run:

```bash
bash eval/run_exskill_inference.sh
```

This script focuses on reasoning without updating the library.

### Key Parameters Reference

#### Data & Output

| Variable | Argument | Description |
|---|---|---|
| `DATA_PATH` | `--input-file` | Path to the benchmark JSON file |
| `IMAGE_DIR` | `--image-folder` | Root directory for benchmark images |
| `OUTPUT_DIR` | `--output-dir` | Directory to save per-sample results |
| `MAX_SAMPLES` | `--max-samples` | Limit number of samples (useful for debugging) |

#### Inference

| Variable | Argument | Description |
|---|---|---|
| `MAX_TURNS` | `--max-turns` | Max agent turns per rollout (default: 20) |
| `MAX_TOTAL_TOKENS` | `--max-total-tokens` | Max context tokens per turn (default: 32768) |
| `TEMPERATURE` | `--temperature` | Sampling temperature (default: 0.6) |
| `ROLLOUTS_PER_SAMPLE` | `--rollouts-per-sample` | Independent rollouts per sample; >1 enables pass@k / avg@k evaluation |
| `NUM_WORKERS` | `--num-workers` | Parallel workers (set to match your API rate limit) |
| `SYSTEM_PROMPT_TYPE` | `--system-prompt-key` | Agent prompt variant: `multi_tool_agent` (default), `multi_tool_agent_search`, `multi_tool_agent_code`, `agent_zoom`, `direct_cot` |

#### Tools

| Variable | Argument | Description |
|---|---|---|
| `ENABLED_TOOLS` | (env var) | Comma-separated list of active tools: `web_search`, `visit`, `code_interpreter`, `image_search`, `zoom` |
| `WEB_SEARCH_MAX_CALLS` | `--web-search-max-calls` | Max `web_search` calls per sample (default: 7) |
| `IMAGE_SEARCH_MAX_CALLS` | `--image-search-max-calls` | Max `image_search` calls per sample (default: 5) |

#### Experience Library

| Variable | Argument | Description |
|---|---|---|
| `EXPERIENCE_LIBRARY` | `--experience-library` | Path to the experience JSON file (created automatically if absent) |
| `EXPERIENCE_MAX_OPS` | `--experience-max-ops` | Max experience operations extracted per sample critique (default: 3) |
| `EXPERIENCE_MAX_ITEMS` | `--experience-max-items` | Max entries kept in the experience library (default: 120) |
| `EXPERIENCE_LARGE_BATCH` | `--experience-large-batch` | Number of rollouts to accumulate before triggering batch experience generation |
| `EXPERIENCE_RETRIEVAL_TOP_K` | `--experience-retrieval-top-k` | Number of experiences retrieved per query (default: 3) |
| — | `--experience-enable` | Enable experience injection at inference time |
| — | `--experience-online-generate` | Generate experiences from trajectories after each batch |
| — | `--experience-library-update` | Merge new experiences back into the library |
| — | `--experience-retrieval` | Use embedding-based retrieval instead of injecting all experiences |
| — | `--experience-retrieval-decomposition` | Decompose the task into subtasks before retrieval |
| — | `--experience-retrieval-rewrite` | Rewrite retrieved experiences to fit the current task |
| — | `--experience-refine` | Periodically consolidate and trim the experience library |

#### Skill Library

| Variable | Argument | Description |
|---|---|---|
| `SKILL_LIBRARY` | `--skill-library` | Path to the skill document (`SKILL.md`; created automatically if absent) |
| `SKILL_MAX_LENGTH` | `--skill-max-length` | Word count threshold to trigger skill document refinement (default: 1000) |
| — | `--skill-enable` | Enable skill generation from trajectories |
| — | `--skill-inference` | Inject the (adapted) skill document into the system prompt at inference time |
| — | `--skill-refine` | Periodically consolidate and trim the skill document |
| — | `--no-skill-adaptation` | Inject the raw skill document without per-sample adaptation (default: adapt) |

---

## Memory Bank

The memory bank stores the accumulated experience library and skill document across runs:

```
memory_bank/
└── <run_name>/
    ├── experiences.json    # Structured experience library
    └── SKILL.md            # Global skill document
```

Both files are created automatically on the first run if they do not exist. They are updated in-place after each batch.

---

## Output Format

Each sample produces a subdirectory under `--output-dir`:

```
output/<run_name>/<sample_id>/
├── traj.jsonl              # Full agent trajectory (turn-by-turn)
├── metrics.json            # Evaluation score and metadata
├── exp_summary_prompt.txt  # Trajectory summary prompt sent to the experience model
├── exp_summary_resp.txt    # Experience model response
├── online_experiences.json # Per-sample generated experiences
└── rollout_*/              # Per-rollout subdirectories (when rollouts-per-sample > 1)
```

A dataset-level summary is written to `output/<run_name>/summary_k.json`.

---

## Citation

If you use XSkill in your research, please cite:

```bibtex
@misc{jiang2026xskillcontinuallearningexperience,
      title={XSkill: Continual Learning from Experience and Skills in Multimodal Agents}, 
      author={Guanyu Jiang and Zhaochen Su and Xiaoye Qu and Yi R. Fung},
      year={2026},
      eprint={2603.12056},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.12056}, 
}
```

---

## License

This project is released under the [MIT License](LICENSE).
