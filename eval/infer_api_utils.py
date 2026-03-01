"""
Utility functions for infer_api.py to reduce code duplication and improve maintainability.
"""

import os
import re
import json
import logging
import copy
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from exskill import ExperienceLLM, ExperienceRetriever, load_experiences, format_for_prompt, rewrite_experiences_for_task, adapt_skill_for_task
from prompts.skill_prompts_test_time import SKILL_INJECTION_HEADER
from engine.api_processors import set_global_prompts as set_processors_prompts
from qwen_vl_utils import fetch_image
from utils.context_utils import process_image

logger = logging.getLogger(__name__)

# Constants
RETRIEVAL_INFO_FILENAME = "tt_decomposition_info.txt"
SAMPLE_SUMMARY_FILENAME = "metrics_sample.json"
DATASET_SUMMARY_FILENAME = "metrics_at_k.json"


def initialize_experience_retriever(args):
    """
    Initialize experience retriever if retrieval mode is enabled.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (experience_retriever, system_prompt)
            - experience_retriever: ExperienceRetriever instance or None
            - system_prompt: Updated system prompt with experiences injected (if not using retrieval)
    """
    experience_retriever = None
    system_prompt = None
    
    if getattr(args, 'experience_enable', False) and args.experience_library:
        try:
            exps = load_experiences(args.experience_library)
            
            # Check if retrieval mode is enabled
            use_retrieval = getattr(args, 'experience_retrieval', False)
            
            if use_retrieval:
                # Initialize experience retriever for retrieval-based injection
                print(f"\n{'='*80}")
                print(f"Initializing Experience Retriever (retrieval mode enabled)...")
                print(f"{'='*80}\n")
                print(f"Loading experience library: {args.experience_library}")
                print(f"Total experiences in library: {len(exps)}")
                
                try:
                    # Default: cache in the same directory as the experience library JSON
                    enable_cache = getattr(args, 'experience_embedding_cache_enable', True)
                    cache_dir = os.path.dirname(args.experience_library) if args.experience_library else None
                    embedding_model = os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "text-embedding-3-small")
                    embedding_api_key = getattr(args, 'experience_embedding_api_key', None) or os.environ.get("EXPERIENCE_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
                    embedding_endpoint = getattr(args, 'experience_embedding_endpoint', None) or os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT") or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
                    
                    # Normalize endpoint
                    if embedding_endpoint and not embedding_endpoint.endswith("/v1"):
                        embedding_endpoint = embedding_endpoint.rstrip("/") + "/v1"
                    
                    # Initialize LLM client for decomposition or rewrite (if enabled)
                    llm_client = None
                    use_decomposition = getattr(args, 'experience_retrieval_decomposition', False)
                    use_rewrite = getattr(args, 'experience_retrieval_rewrite', False)
                    if use_decomposition or use_rewrite:
                        features = []
                        if use_decomposition:
                            features.append("task decomposition")
                        if use_rewrite:
                            features.append("experience rewrite")
                        print(f"Initializing LLM client for {', '.join(features)}...")
                        llm_client = ExperienceLLM()
                        print(f"LLM client initialized.")
                    
                    print(f"\nInitializing Experience Retriever (this may take a moment for first-time embedding generation)...")
                    experience_retriever = ExperienceRetriever(
                        experiences=exps,
                        embedding_model=embedding_model,
                        embedding_api_key=embedding_api_key,
                        embedding_endpoint=embedding_endpoint,
                        cache_dir=cache_dir,
                        enable_cache=enable_cache,
                        llm_client=llm_client,
                        experience_library_path=args.experience_library,
                    )
                    stats = experience_retriever.get_embedding_stats()
                    print(f"Experience Retriever initialized:")
                    print(f"  - Total experiences: {stats['total_experiences']}")
                    print(f"  - Embedded count: {stats['embedded_count']}")
                    print(f"  - Embedding model: {stats['embedding_model']}")
                    print(f"  - Cache enabled: {stats['cache_enabled']}")
                    if stats['cache_path']:
                        print(f"  - Cache path: {stats['cache_path']}")
                    if getattr(args, 'experience_retrieval_decomposition', False):
                        print(f"  - Task decomposition: ENABLED")
                    if getattr(args, 'experience_retrieval_rewrite', False):
                        print(f"  - Experience rewrite: ENABLED")
                    print(f"\n{'='*80}\n")
                except Exception as e:
                    print(f"Warning: Failed to initialize Experience Retriever: {e}")
                    print(f"  Falling back to full experience injection mode.")
                    experience_retriever = None
                    use_retrieval = False
            
            if not use_retrieval:
                # Traditional mode: inject all experiences (or up to max_items)
                injection_text = format_for_prompt(exps, max_items=getattr(args, 'experience_max_items', 256))
                if injection_text:
                    system_prompt = injection_text  # Will be combined with base prompt later
                    print(f"Loaded {len(exps)} experiences for injection (full injection mode).")
        except Exception as e:
            print(f"Experience injection skipped: {e}")
    
    return experience_retriever, system_prompt


def retrieve_experiences_for_sample(sample, args, experience_retriever, base_system_prompt, 
                                     question_id, sample_dir, update_global_prompts=True):
    """
    Retrieve experiences and/or inject skills for a single sample.
    Supports: Experience only, Skill only, or both combined.
    """
    # Check what's enabled
    exp_enabled = experience_retriever is not None
    skill_enabled = getattr(args, 'skill_inference', False) and getattr(args, 'skill_library', None)
    
    if not (exp_enabled or skill_enabled):
        return _finalize_prompt(args, base_system_prompt, update_global_prompts)
    
    if sample_dir:
        os.makedirs(sample_dir, exist_ok=True)
    
    try:
        query_text = sample.get('problem', sample.get('question', ''))
        images = _load_images_for_retrieval(sample, args)
        
        # --- 1. Experience Retrieval (optional) ---
        retrieved_exps, retrieval_info, original_retrieved_exps = {}, {}, None
        use_rewrite = getattr(args, 'experience_retrieval_rewrite', False)
        llm_client = None
        
        if exp_enabled:
            use_decomposition = getattr(args, 'experience_retrieval_decomposition', False)
            if use_decomposition:
                retrieved_exps, retrieval_info = experience_retriever.retrieve_with_decomposition(
                    task_description=query_text,
                    top_k=getattr(args, 'experience_retrieval_top_k', 5),
                    min_similarity=getattr(args, 'experience_retrieval_min_similarity', 0.0),
                    subtask_top_k=getattr(args, 'experience_retrieval_subtask_top_k', None),
                    images=images,
                )
            else:
                retrieved_exps, retrieval_info = experience_retriever.retrieve(
                    query=query_text,
                    top_k=getattr(args, 'experience_retrieval_top_k', 5),
                    min_similarity=getattr(args, 'experience_retrieval_min_similarity', 0.0),
                )
            
            if not retrieved_exps:
                print(f"  [Retrieval] No relevant experiences found for {question_id}")
            
            if retrieved_exps and use_rewrite:
                original_retrieved_exps = copy.deepcopy(retrieved_exps)
                llm_client = experience_retriever.llm_client or ExperienceLLM()
                print(f"  [Retrieval] Rewriting {len(retrieved_exps)} experiences...")
                retrieved_exps = rewrite_experiences_for_task(retrieved_exps, query_text, llm_client, images)
        
        # --- 2. Skill Processing (optional) ---
        skill_injection = ""
        if skill_enabled and os.path.exists(args.skill_library):
            skill_injection = _process_skill(args, query_text, retrieved_exps, images, llm_client, sample_dir, question_id)
        
        # --- 3. Build Final Injection ---
        exp_injection = format_for_prompt(retrieved_exps, max_items=len(retrieved_exps)) if retrieved_exps else ""
        
        # Build final injection: combine skill and experience when appropriate
        use_adaptation = getattr(args, 'skill_adaptation', True)
        
        if skill_injection and exp_injection:
            # If skill is enabled but not adapted, combine both
            # (In adapt mode, skill already incorporates experience guidance, so only inject skill)
            if skill_enabled and not use_adaptation:
                full_injection = f"{skill_injection}\n\n{exp_injection}"
                print(f"  [Injection] Combined raw skill + experiences for {question_id}")
            else:
                # Adapt mode: skill already contains experience guidance, only inject skill
                full_injection = skill_injection
        elif skill_injection:
            full_injection = skill_injection
        elif exp_injection:
            full_injection = exp_injection
        else:
            full_injection = ""
        
        dynamic_system_prompt = (
            f"{full_injection}\n\n{base_system_prompt}" if full_injection and base_system_prompt
            else (full_injection or base_system_prompt)
        )
        
        # Save retrieval info
        if exp_enabled and sample_dir:
            save_retrieval_info(experience_retriever, sample_dir, retrieved_exps or None, 
                              original_retrieved_exps, use_rewrite, retrieval_info=retrieval_info)
        
        return _finalize_prompt(args, dynamic_system_prompt, update_global_prompts)
        
    except Exception as e:
        logger.warning(f"Failed to process for {question_id}: {e}", exc_info=True)
        print(f"  [Pipeline] Warning: {e}")
        return _finalize_prompt(args, base_system_prompt, update_global_prompts)


def _process_skill(args, query_text, retrieved_exps, images, llm_client, sample_dir, question_id):
    """Process skill: adapt or use raw based on skill_adaptation flag."""
    try:
        with open(args.skill_library, 'r', encoding='utf-8') as f:
            base_skill = f.read()
        
        use_adaptation = getattr(args, 'skill_adaptation', True)  # Default: adapt
        
        # Create LLM client if needed
        if use_adaptation and llm_client is None:
            llm_client = ExperienceLLM()
        
        exp_text = "\n\n".join(f"--- {k} ---\n{v}" for k, v in retrieved_exps.items()) if retrieved_exps else ""
        
        if use_adaptation:
            # === Standard Adaptation Mode ===
            adapted_skill = adapt_skill_for_task(base_skill, exp_text, query_text, llm_client, images)
            skill_content = adapted_skill
            print(f"  [Skill] Adapted for {question_id}")
            
            if sample_dir:
                with open(os.path.join(sample_dir, "tt_skill_adapted.md"), 'w', encoding='utf-8') as f:
                    f.write(skill_content)
        
        else:
            # === Raw Mode ===
            skill_content = base_skill
            print(f"  [Skill] Using raw skill for {question_id}")
            
            if sample_dir:
                with open(os.path.join(sample_dir, "tt_skill_original.md"), 'w', encoding='utf-8') as f:
                    f.write(skill_content)
        
        return SKILL_INJECTION_HEADER.format(skill_content=skill_content)
    except Exception as e:
        print(f"  [Skill] Warning: {e}")
        return ""


def _finalize_prompt(args, prompt, update_global_prompts):
    """Helper to finalize prompt: update globals or set on args."""
    if update_global_prompts:
        set_processors_prompts(prompt)
    else:
        args._sample_system_prompt = prompt
    return prompt


def _load_images_for_retrieval(sample, args):
    """
    Load images from original data source for experience retrieval.
    
    Args:
        sample: Data sample
        args: Command line arguments
        
    Returns:
        list or None: List of processed images, or None if no images found
    """
    if not (hasattr(args, 'image_folder') and args.image_folder):
        return None
    
    image_paths = sample.get('images', [])
    if not image_paths:
        return None
    
    images = []
    max_pixels = getattr(args, 'max_pixels', 2000000)
    min_pixels = getattr(args, 'min_pixels', 40000)
    
    for img_path in image_paths:
        try:
            full_path = os.path.join(args.image_folder, img_path)
            if not os.path.exists(full_path):
                print(f"  [Retrieval] Warning: Image file not found: {full_path}")
                continue
            
            # Use the same image loading logic as in parse_and_load_multiple_images
            original_image = fetch_image({'image': full_path, 'max_pixels': max_pixels})
            processed_image = process_image(original_image, max_pixels, min_pixels)
            images.append(processed_image)
        except Exception as e:
            logger.warning(f"Failed to load image from {img_path}: {e}", exc_info=True)
            print(f"  [Retrieval] Warning: Failed to load image from {img_path}: {e}")
    
    if images:
        # print(f"  [Retrieval] Successfully loaded {len(images)} image(s) for task decomposition")
        pass
    else:
        print(f"  [Retrieval] Warning: Failed to load any images from original data source")
    
    return images if images else None


def _parse_exp_blocks(section: str) -> Dict[str, str]:
    """Parse [id]\\ncontent blocks from a section string. Returns dict id -> content."""
    result = {}
    exp_blocks = re.split(r'\n\[([^\]]+)\]\n', section)
    for i in range(1, len(exp_blocks), 2):
        exp_id = exp_blocks[i]
        exp_content = exp_blocks[i + 1].strip().strip('-').strip() if i + 1 < len(exp_blocks) else ""
        if exp_content:
            result[exp_id] = exp_content
    return result


def _parse_exp_ids(section: str) -> set:
    """Parse experience IDs from [id] markers in a section string."""
    return set(re.findall(r'\[([^\]]+)\]', section))


def get_used_original_experiences(sample_dir: str) -> Dict[str, str]:
    """
    Get original experiences that were actually used (not skipped during rewrite).
    
    Returns the original content of experiences that survived the rewrite process.
    If rewrite was not used, returns the retrieved experiences as-is.
    
    Args:
        sample_dir: Sample directory path
        
    Returns:
        Dict mapping experience ID to original content, or empty dict if not available
    """
    retrieval_info_file = os.path.join(sample_dir, RETRIEVAL_INFO_FILENAME)
    if not os.path.exists(retrieval_info_file):
        return {}
    
    try:
        with open(retrieval_info_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the file to extract original and rewritten experiences
        original_exps = {}
        rewritten_ids = set()
        
        # Extract original experiences
        if "ORIGINAL RETRIEVED EXPERIENCES" in content:
            orig_section = content.split("ORIGINAL RETRIEVED EXPERIENCES")[1]
            if "REWRITTEN EXPERIENCES" in orig_section:
                orig_section = orig_section.split("REWRITTEN EXPERIENCES")[0]
            elif "RETRIEVED EXPERIENCE CONTENT" in orig_section:
                orig_section = orig_section.split("RETRIEVED EXPERIENCE CONTENT")[0]
            original_exps = _parse_exp_blocks(orig_section)
        
        # Extract rewritten experience IDs (these are the ones that survived)
        if "REWRITTEN EXPERIENCES" in content:
            rewrite_section = content.split("REWRITTEN EXPERIENCES")[1]
            rewritten_ids = _parse_exp_ids(rewrite_section)
        elif "RETRIEVED EXPERIENCE CONTENT" in content:
            # No rewrite used, all retrieved experiences are "used"
            rewrite_section = content.split("RETRIEVED EXPERIENCE CONTENT")[1]
            rewritten_ids = _parse_exp_ids(rewrite_section)
            if not original_exps:
                original_exps = _parse_exp_blocks(rewrite_section)
                rewritten_ids = set(original_exps.keys())
        
        # Return only the original content of experiences that were actually used
        return {k: v for k, v in original_exps.items() if k in rewritten_ids}
    
    except Exception as e:
        logger.warning(f"Failed to parse retrieval info: {e}")
        return {}


def save_retrieval_info(experience_retriever, sample_dir, retrieved_exps=None, original_retrieved_exps=None, rewrite_used=False, retrieval_info=None):
    """
    Save retrieval information to file, including rewrite information if applicable.
    
    Args:
        experience_retriever: ExperienceRetriever instance
        sample_dir: Sample directory path
        retrieved_exps: Final experiences dict (after rewrite if rewrite was used) or None
        original_retrieved_exps: Original retrieved experiences dict (before rewrite) or None
        rewrite_used: Whether rewrite was applied
        retrieval_info: Optional retrieval info dict (if provided, used instead of getting from retriever)
    """
    if retrieval_info is None:
        retrieval_info = experience_retriever.get_last_retrieval_info() if experience_retriever else None
    if not retrieval_info:
        return
    
    retrieval_info_file = os.path.join(sample_dir, RETRIEVAL_INFO_FILENAME)
    os.makedirs(sample_dir, exist_ok=True)
    
    with open(retrieval_info_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIENCE RETRIEVAL INFORMATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Original Query:\n{retrieval_info.get('original_query', 'N/A')}\n\n")
        f.write(f"Decomposition Used: {retrieval_info.get('decomposition_used', False)}\n")
        f.write(f"Rewrite Used: {rewrite_used}\n\n")
        
        if retrieval_info.get('decomposition_used', False):
            f.write(f"Subtasks ({len(retrieval_info.get('subtasks', []))}):\n")
            for i, subtask in enumerate(retrieval_info.get('subtasks', []), 1):
                f.write(f"\n  {i}. Type: {subtask.get('type', 'unknown')}\n")
                f.write(f"     Query: {subtask.get('query', '')}\n")
            
            f.write(f"\n\nRetrieval Details:\n")
            for i, detail in enumerate(retrieval_info.get('retrieval_details', []), 1):
                f.write(f"\n  {i}. Subtask: {detail.get('subtask_type', 'unknown')}\n")
                f.write(f"     Query: {detail.get('query', '')}\n")
                f.write(f"     Retrieved {detail.get('count', 0)} experiences:\n")
                for exp_id in detail.get('retrieved_experience_ids', []):
                    f.write(f"       - {exp_id}\n")
        
        f.write(f"\n\nFinal Retrieved Experiences ({retrieval_info.get('total_unique_experiences', 0)} total):\n")
        for exp_id in retrieval_info.get('retrieved_experiences', []):
            f.write(f"  - {exp_id}\n")
        
        # Save original retrieved experiences (before rewrite) if rewrite was used
        if rewrite_used and original_retrieved_exps:
            f.write(f"\n\n{'=' * 80}\n")
            f.write("ORIGINAL RETRIEVED EXPERIENCES (Before Rewrite)\n")
            f.write("=" * 80 + "\n")
            for exp_id, exp_text in original_retrieved_exps.items():
                f.write(f"\n[{exp_id}]\n")
                f.write("-" * 80 + "\n")
                f.write(exp_text)
                f.write("\n\n")
        
        # Save final experiences (after rewrite if rewrite was used)
        if retrieved_exps:
            section_title = "REWRITTEN EXPERIENCES (After Rewrite)" if rewrite_used else "RETRIEVED EXPERIENCE CONTENT"
            f.write(f"\n\n{'=' * 80}\n")
            f.write(f"{section_title}\n")
            f.write("=" * 80 + "\n")
            for exp_id, exp_text in retrieved_exps.items():
                f.write(f"\n[{exp_id}]\n")
                f.write("-" * 80 + "\n")
                f.write(exp_text)
                f.write("\n\n")
        else:
            f.write("\nNo relevant experiences were retrieved for this query.\n")


def compute_and_save_sample_summary(question_id, sample_rollout_results, rollouts_per_sample, sample_dir):
    """
    Compute and save sample-level pass@k and average@k metrics.
    
    Args:
        question_id: Question ID
        sample_rollout_results: List of rollout result dictionaries
        rollouts_per_sample: Number of rollouts per sample
        sample_dir: Sample directory path
        
    Returns:
        dict: Sample summary dictionary
    """
    if not sample_rollout_results:
        return None
    
    # Sort results by rollout_idx to ensure correct order (parallel execution may complete in any order)
    # If _rollout_idx is not present (e.g., old code path), keep original order
    if sample_rollout_results and '_rollout_idx' in sample_rollout_results[0]:
        sorted_results = sorted(sample_rollout_results, key=lambda r: r.get('_rollout_idx', 0))
    else:
        sorted_results = sample_rollout_results

    accuracies = [r["accuracy_score"] for r in sorted_results]
    sample_summary = {
        "question_id": question_id,
        "num_rollouts": len(sorted_results),
        "accuracies": accuracies,
    }

    # Compute pass@k and average@k for each k
    for k in range(1, len(accuracies) + 1):
        acc_at_k = accuracies[:k]
        sample_summary[f"pass@{k}"] = 1.0 if any(acc == 1.0 for acc in acc_at_k) else 0.0
        sample_summary[f"average@{k}"] = sum(acc_at_k) / k

    # Save to file
    with open(os.path.join(sample_dir, SAMPLE_SUMMARY_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(sample_summary, f, indent=4, ensure_ascii=False)
    
    print(f"  \nSample {question_id}: pass@{rollouts_per_sample}={sample_summary[f'pass@{rollouts_per_sample}']:.2f}, "
          f"average@{rollouts_per_sample}={sample_summary[f'average@{rollouts_per_sample}']:.4f}")
    
    return sample_summary


def check_sample_completed(sample_dir, args):
    """
    Check if a sample is already completed (has traj.jsonl with turn_idx and metrics.json).
    
    Args:
        sample_dir: Sample directory path
        args: Command line arguments
        
    Returns:
        dict or None: Sample info dict if completed, None otherwise
    """
    if not getattr(args, 'skip_completed', False):
        return None
    
    traj_file = os.path.join(sample_dir, 'traj.jsonl')
    metrics_file = os.path.join(sample_dir, 'metrics.json')
    
    # Check if completed: traj.jsonl exists with turn_idx, and metrics.json exists
    is_complete = False
    if os.path.exists(traj_file) and os.path.exists(metrics_file):
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                lines = [l for l in f if l.strip()]
                # Check if there are any turn_idx entries
                for line in lines:
                    try:
                        traj_data = json.loads(line)
                        if 'turn_idx' in traj_data and traj_data['turn_idx'] is not None:
                            is_complete = True
                            break
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug(f"Error parsing trajectory line: {e}")
                        pass
        except Exception as e:
            logger.debug(f"Error reading trajectory file: {e}")
            pass
    
    if not is_complete:
        return None
    
    # Try to load existing result
    sample_rollout_results = []
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        # Read turns from traj.jsonl to get accurate turn count
        num_turns = 0
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                max_turn_idx = -1
                for line in f:
                    if line.strip():
                        try:
                            traj_data = json.loads(line)
                            turn_idx = traj_data.get('turn_idx')
                            if turn_idx is not None and turn_idx > max_turn_idx:
                                max_turn_idx = turn_idx
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.debug(f"Error parsing trajectory line: {e}")
                            pass
            num_turns = max_turn_idx + 1 if max_turn_idx >= 0 else 0
        except Exception as e:
            logger.debug(f"Error reading trajectory file for turn count: {e}")
            pass
        
        # Create a conversation_history with the correct number of assistant turns
        # This ensures accurate turn statistics
        conversation_history = [{"role": "assistant", "content": ""} for _ in range(num_turns)] if num_turns > 0 else []
        
        # Create result dict with accurate turn information
        sample_rollout_results.append({
            'question_id': os.path.basename(sample_dir),
            'accuracy_score': metrics.get('accuracy_score', 0.0),
            'conversation_history': conversation_history,
        })
    except Exception as e:
        logger.debug(f"Error loading existing metrics: {e}")
        pass
    
    return {
        'sample_rollout_results': sample_rollout_results
    }


def prepare_sample_args(args):
    """
    Create a thread-safe copy of args for parallel execution.
    
    Args:
        args: Original command line arguments
        
    Returns:
        A shallow copy of args with deep-copied tool_configs
    """
    sample_args = copy.copy(args)
    
    # Deep copy tool_configs to avoid sharing config dictionaries across parallel executions
    # This ensures each execution gets its own copy of tool configs, preventing work_dir conflicts
    if hasattr(args, 'tool_configs') and args.tool_configs:
        sample_args.tool_configs = copy.deepcopy(args.tool_configs)
    
    return sample_args


def get_sample_metadata(sample, sample_idx, output_dir):
    """
    Extract question_id and sample_dir from a sample.
    
    Args:
        sample: Data sample dictionary
        sample_idx: Sample index (for fallback question_id)
        output_dir: Output directory path
        
    Returns:
        tuple: (question_id, sample_dir)
    """
    question_id = sample.get('doc_id', sample.get('question_id', f"sample_{sample_idx}"))
    sample_dir = os.path.join(output_dir, question_id)
    return question_id, sample_dir


def compute_dataset_summary(data, args):
    """
    Compute and save dataset-level pass@k and average@k metrics.
    
    Args:
        data: List of data samples
        args: Command line arguments
        
    Returns:
        dict: Dataset summary dictionary or None if no summaries found
    """
    # Collect sample summaries
    sample_summaries = []
    for sample in data:
        question_id = sample.get('doc_id', sample.get('question_id', f"sample_{data.index(sample)}"))
        summary_path = os.path.join(args.output_dir, question_id, SAMPLE_SUMMARY_FILENAME)
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                sample_summaries.append(json.load(f))
    
    if not sample_summaries:
        return None
    
    # Compute dataset-level metrics
    dataset_summary = {
        "num_samples": len(sample_summaries),
        "num_rollouts": args.rollouts_per_sample,
        "k_metrics": {}
    }
    
    for k in range(1, args.rollouts_per_sample + 1):
        pass_values = [s[f"pass@{k}"] for s in sample_summaries if f"pass@{k}" in s]
        avg_values = [s[f"average@{k}"] for s in sample_summaries if f"average@{k}" in s]
        
        dataset_summary["k_metrics"][k] = {
            "pass_at_k": round(sum(pass_values) / len(pass_values), 4) if pass_values else 0.0,
            "average_at_k": round(sum(avg_values) / len(avg_values), 4) if avg_values else 0.0,
        }
    
    # Save dataset summary
    with open(os.path.join(args.output_dir, DATASET_SUMMARY_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(dataset_summary, f, indent=4, ensure_ascii=False)
    
    # Print summary
    print(f"Total samples: {dataset_summary['num_samples']}")
    print(f"Rollouts per sample: {dataset_summary['num_rollouts']}\n")
    for k in range(1, args.rollouts_per_sample + 1):
        metrics = dataset_summary["k_metrics"][k]
        print(f"k={k}:")
        print(f"  Pass@{k}:    {metrics['pass_at_k']:.4f}")
        print(f"  Average@{k}: {metrics['average_at_k']:.4f}")
    print("="*80)
    
    return dataset_summary


def reload_experiences(args, base_system_prompt, experience_retriever=None):
    """
    Reload experiences from library and update system prompt and/or retriever.
    
    Args:
        args: Command line arguments
        base_system_prompt: Base system prompt (without experiences)
        experience_retriever: Optional ExperienceRetriever instance to update (for retrieval mode)
        
    Returns:
        str: Updated system prompt with experiences injected
    """
    if not (getattr(args, 'experience_enable', False) and args.experience_library):
        return base_system_prompt
    
    try:
        exps = load_experiences(args.experience_library)
        use_retrieval = getattr(args, 'experience_retrieval', False)

        if use_retrieval and experience_retriever is not None:
            # Update retriever with new experiences (incremental update)
            print(f"  Updating experience retriever with {len(exps)} experiences...")
            experience_retriever.update_experiences(exps, incremental=True)
            print(f"  Experience retriever updated successfully.")
            return base_system_prompt

        # Traditional mode: inject all experiences into system prompt
        injection_text = format_for_prompt(exps, max_items=getattr(args, 'experience_max_items', 256))
        if injection_text:
            system_prompt = (injection_text + "\n\n" + base_system_prompt) if base_system_prompt else injection_text
            set_processors_prompts(system_prompt)
            print(f"  Reloaded {len(exps)} experiences for next batch.")
            return system_prompt
    except Exception as e:
        print(f"  Warning: Failed to reload experiences: {e}")
        import traceback
        traceback.print_exc()

    return base_system_prompt


def execute_pipeline_parallel_processing(
    samples: List[Dict],
    sample_indices: List[int],
    sample_results_dict: Dict,
    args: Any,
    experience_retriever: Optional[Any],
    base_system_prompt: str,
    sampling_params: Dict,
    run_single_rollout_func: Callable,
    progress_desc: Optional[str] = None,
    all_results_list: Optional[List] = None
) -> None:
    """
    Executes a pipeline parallel processing of samples to maximize hardware utilization:
    
    Pipeline stages:
    1. Retrieval Stage: Submits experience retrieval tasks for samples. To prevent 
       "worker starvation" for the reasoning stage, retrieval tasks are submitted 
       gradually as slots open up.
    2. Reasoning (Rollout) Stage: As soon as a sample's retrieval (and optional rewrite)
       completes, all its rollouts (pass@k) are immediately submitted to the pool.
    3. Management Stage: A unified monitoring loop uses wait() to react to any completed 
       future (retrieve or rollout), ensuring smooth transition between stages.
    
    Args:
        samples: List of sample dictionaries from the dataset.
        sample_indices: List of actual sample indices (mapping to original dataset).
        sample_results_dict: Shared dictionary to store results for each sample.
        args: Command line arguments containing configuration (num_workers, etc).
        experience_retriever: ExperienceRetriever instance. If None, retrieval stage 
            may still run if skill_inference is enabled (skill-only mode).
        base_system_prompt: The initial system prompt used if retrieval is disabled or fails.
        sampling_params: Parameters for the LLM call (temperature, top_p, etc).
        run_single_rollout_func: The core function to execute one rollout (one pass).
        progress_desc: String description for the tqdm progress bar.
        all_results_list: Optional list to collect every single rollout result across all samples.
        
    Returns:
        None (results are mutated in sample_results_dict and all_results_list).
    """
    from tqdm import tqdm  # type: ignore

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Initialize tracking
        active_futures = {}  # future -> info dict
        pending_retrieval_samples = []
        
        # Prepare all sample data
        for local_idx, sample in enumerate(samples):
            actual_sample_idx = sample_indices[local_idx]
            question_id = sample_results_dict[actual_sample_idx]['question_id']
            sample_dir = sample_results_dict[actual_sample_idx]['sample_dir']
            sample_args = prepare_sample_args(args)
            pending_retrieval_samples.append({
                'sample_idx': actual_sample_idx,
                'sample': sample,
                'sample_args': sample_args,
                'question_id': question_id,
                'sample_dir': sample_dir
            })

        # Progress tracking
        total_rollouts = len(samples) * args.rollouts_per_sample
        pbar = tqdm(total=total_rollouts, desc=progress_desc or "Processing rollouts")
        
        rollout_counts = {}  # track completed rollouts per sample
        completed_samples = set()  # track samples with all rollouts finished
        next_sample_idx = 0
        
        # Check if skill-only mode (no experience_retriever but skill enabled)
        skill_enabled = getattr(args, 'skill_inference', False) and getattr(args, 'skill_library', None)
        needs_retrieval_stage = experience_retriever is not None or skill_enabled
        
        # Limit concurrent retrievals to prevent queueing delay for rollouts
        max_concurrent_retrievals = args.num_workers if needs_retrieval_stage else 0
        active_retrieval_count = 0

        # Initial submission
        if not needs_retrieval_stage:
            # No retrieval or skill needed: submit all rollouts immediately
            for item in pending_retrieval_samples:
                actual_sample_idx = item['sample_idx']
                sample = item['sample']
                sample_args = item['sample_args']
                sample_args._sample_system_prompt = base_system_prompt
                for rollout_idx in range(args.rollouts_per_sample):
                    f = executor.submit(run_single_rollout_func, sample, sample_args, sampling_params, rollout_idx)
                    active_futures[f] = {'type': 'rollout', 'sample_idx': actual_sample_idx, 'rollout_idx': rollout_idx}
            next_sample_idx = len(pending_retrieval_samples)
        else:
            # Submit initial batch of retrievals up to num_workers
            while next_sample_idx < len(pending_retrieval_samples) and active_retrieval_count < max_concurrent_retrievals:
                item = pending_retrieval_samples[next_sample_idx]
                f = executor.submit(
                    retrieve_experiences_for_sample,
                    item['sample'], item['sample_args'], experience_retriever, base_system_prompt,
                    item['question_id'], item['sample_dir'], update_global_prompts=False
                )
                active_futures[f] = {
                    'type': 'retrieve', 
                    'sample_idx': item['sample_idx'], 
                    'sample': item['sample'], 
                    'sample_args': item['sample_args'],
                    'question_id': item['question_id'],
                    'sample_dir': item['sample_dir']
                }
                active_retrieval_count += 1
                next_sample_idx += 1

        # Main Pipeline Loop: wait for ANY future to complete and react
        while active_futures:
            # Use wait() to handle multiple types of futures in one place
            # It returns as soon as at least one future is done
            done_futures, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
            
            for future in done_futures:
                if future not in active_futures:
                    continue
                    
                info = active_futures.pop(future)
                
                if info['type'] == 'retrieve':
                    active_retrieval_count -= 1
                    actual_sample_idx = info['sample_idx']
                    sample = info['sample']
                    sample_args = info['sample_args']
                    question_id = info['question_id']
                    
                    try:
                        future.result()  # Ensure retrieval finished successfully
                        print(f"  [Pipeline] Sample {question_id}: Retrieve completed, submitting {args.rollouts_per_sample} rollouts...")
                    except Exception as e:
                        print(f"  [Pipeline] Error retrieving for sample {question_id}: {e}. Falling back to base prompt.")
                        sample_args._sample_system_prompt = base_system_prompt
                    
                    # Submit rollouts for this sample
                    for rollout_idx in range(args.rollouts_per_sample):
                        f_rollout = executor.submit(run_single_rollout_func, sample, sample_args, sampling_params, rollout_idx)
                        active_futures[f_rollout] = {'type': 'rollout', 'sample_idx': actual_sample_idx, 'rollout_idx': rollout_idx}
                    
                    # After submitting rollouts, try to fill the retrieval slot if any pending
                    if next_sample_idx < len(pending_retrieval_samples):
                        item = pending_retrieval_samples[next_sample_idx]
                        f_new = executor.submit(
                            retrieve_experiences_for_sample,
                            item['sample'], item['sample_args'], experience_retriever, base_system_prompt,
                            item['question_id'], item['sample_dir'], update_global_prompts=False
                        )
                        active_futures[f_new] = {
                            'type': 'retrieve', 
                            'sample_idx': item['sample_idx'], 
                            'sample': item['sample'], 
                            'sample_args': item['sample_args'],
                            'question_id': item['question_id'],
                            'sample_dir': item['sample_dir']
                        }
                        active_retrieval_count += 1
                        next_sample_idx += 1
                
                elif info['type'] == 'rollout':
                    actual_sample_idx = info['sample_idx']
                    rollout_idx = info['rollout_idx']
                    
                    try:
                        result = future.result()
                        if result:
                            # Add rollout_idx to result for proper sorting later
                            result['_rollout_idx'] = rollout_idx
                            sample_results_dict[actual_sample_idx]['results'].append(result)
                            if all_results_list is not None:
                                all_results_list.append(result)
                    except Exception as e:
                        print(f"  [Pipeline] Error in rollout {rollout_idx} for sample {actual_sample_idx}: {e}")
                    
                    # Update progress
                    pbar.update(1)
                    
                    # Track rollout completion for sample-level summary
                    if actual_sample_idx not in rollout_counts:
                        rollout_counts[actual_sample_idx] = 0
                    rollout_counts[actual_sample_idx] += 1
                    
                    # If all rollouts for this sample are done, save summary
                    if rollout_counts[actual_sample_idx] == args.rollouts_per_sample and actual_sample_idx not in completed_samples:
                        completed_samples.add(actual_sample_idx)
                        sample_info = sample_results_dict[actual_sample_idx]
                        if sample_info['results']:
                            compute_and_save_sample_summary(
                                sample_info['question_id'], sample_info['results'],
                                args.rollouts_per_sample, sample_info['sample_dir']
                            )

        pbar.close()

