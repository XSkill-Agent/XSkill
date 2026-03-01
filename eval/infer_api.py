"""API-based inference script."""

import argparse
import os
import json
import threading
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Data I/O
from utils.result_utils import save_results, print_summary
import yaml

# Experience learning
from exskill import (
    ExperienceLLM,
    summarize_rollouts,
    intra_sample_experiences,
    load_existing as exp_load_existing,
    batch_merge as exp_batch_merge,
    save_library as exp_save_library,
    generate_skill_for_sample,
    merge_skills,
    refine_experience_library,
    refine_skill_document,
)

# API-specific imports
from engine.api_processors import process_single_sample, set_global_prompts as set_processors_prompts

# Utility functions
from infer_api_utils import (
    initialize_experience_retriever,
    get_used_original_experiences,
    retrieve_experiences_for_sample,
    compute_and_save_sample_summary,
    check_sample_completed,
    reload_experiences,
    prepare_sample_args,
    get_sample_metadata,
    compute_dataset_summary,
    execute_pipeline_parallel_processing,
)


def run_single_rollout(sample, args, sampling_params, rollout_idx):
    """
    Run a single rollout for a given sample.
    
    Args:
        sample: Data sample to process
        args: Command line arguments
        sampling_params: Sampling parameters dictionary
        rollout_idx: Rollout index (0-based)
        
    Returns:
        Result dictionary with trajectory information
    """
    import random
    
    # Create a thread-safe copy of args for parallel execution
    rollout_args = prepare_sample_args(args)
    
    # Apply sample-specific system prompt (including retrieved experiences and adapted skills)
    if hasattr(args, '_sample_system_prompt'):
        set_processors_prompts(args._sample_system_prompt)
    
    # Set seed for this rollout
    seed = args.seed_base + rollout_idx
    random.seed(seed)
    
    # Update sampling params with seed
    rollout_sampling_params = sampling_params.copy()
    if 'seed' in rollout_sampling_params:
        rollout_sampling_params['seed'] = seed

    result = process_single_sample(sample, rollout_args, rollout_sampling_params, rollout_idx=rollout_idx)
    return result


def generate_experience_for_sample(sample_info, args):
    """
    Generate experience for a single sample (extracted from process_single_sample_with_experience).
    This function can be called in parallel for multiple samples.
    
    Args:
        sample_info: Dict containing:
            - 'sample': Data sample
            - 'sample_idx': Sample index
            - 'question_id': Question ID
            - 'sample_dir': Sample directory path
            - 'sample_rollout_results': List of rollout results
        args: Command line arguments
        
    Returns:
        Dict with:
            - 'sample_id': Question ID
            - 'success': bool
            - 'experience_ops': List of experience operations (normalized)
            - 'error': str or None
    """
    question_id = sample_info['question_id']
    sample_dir = sample_info['sample_dir']
    sample_rollout_results = sample_info['sample_rollout_results']
    
    result = {
        'sample_id': question_id,
        'success': False,
        'experience_ops': [],
        'skill_content': '',
        'error': None
    }
    
    try:
        llm = ExperienceLLM()
        
        traj_paths = []
        for rollout_idx in range(len(sample_rollout_results)):
            if args.rollouts_per_sample > 1:
                rollout_dir = os.path.join(sample_dir, f"rollout_{rollout_idx}")
            else:
                rollout_dir = sample_dir
            
            traj_path = os.path.join(rollout_dir, 'traj.jsonl')
            if os.path.exists(traj_path):
                traj_paths.append(traj_path)
        
        # Summarize trajectories (unified function handles both single and multiple)
        if not traj_paths:
            result['error'] = "No trajectory files found"
            print(f"  Warning: No trajectory files found for {question_id}")
            return result
        
        merged_summaries = summarize_rollouts(traj_paths, llm, sample_dir=sample_dir)
        
        # Generate experiences
        if not merged_summaries:
            has_any_turns = False
            for traj_path in traj_paths:
                try:
                    with open(traj_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                rec = json.loads(line)
                                # Check if this record has turn_idx (indicates a turn, not just metadata)
                                if 'turn_idx' in rec and rec.get('turn_idx') is not None:
                                    has_any_turns = True
                                    break
                            except (json.JSONDecodeError, Exception):
                                continue
                    if has_any_turns:
                        break
                except Exception:
                    continue
            
            if not has_any_turns:
                result['error'] = "No turns in trajectories (empty trajectories)"
                print(f"  Warning: Skipping experience generation for {question_id} - trajectories contain no turns (only initial_prompt)")
            else:
                result['error'] = "Failed to generate summary"
                print(f"  Warning: Failed to generate summary for {question_id}")
            return result
        
        question = merged_summaries.get('question', '') or sample_rollout_results[0].get('prompt', '')
        ground_truth = merged_summaries.get('ground_truth', '') or sample_rollout_results[0].get('ground_truth', '')
        system_prompt_text = merged_summaries.get('system_prompt', '')
        
        # Generate experiences using intra-sample critique
        # Only pass the actual summaries (excluding metadata)
        summaries_only = {k: v for k, v in merged_summaries.items() 
                         if k not in ['question', 'ground_truth', 'system_prompt']}
        
        # Get original experiences that were used for this sample (for modify suggestions)
        used_experiences = get_used_original_experiences(sample_dir)
        
        raw_ops = intra_sample_experiences(
            question, ground_truth, summaries_only, llm,
            max_ops=getattr(args, 'experience_max_ops', 2),
            debug_dir=sample_dir,
            system_prompt=system_prompt_text,
            used_experiences=used_experiences
        )
        
        # Normalize operations
        norm_ops = []
        if isinstance(raw_ops, str):
            try:
                raw_ops = json.loads(raw_ops)
            except Exception:
                pass
        
        if isinstance(raw_ops, dict) and 'experiences' in raw_ops:
            for _, v in raw_ops['experiences'].items():
                if isinstance(v, str) and v.strip():
                    norm_ops.append({"experience": v.strip()})
        elif isinstance(raw_ops, list):
            for o in raw_ops:
                if isinstance(o, dict):
                    exp_txt = o.get('experience') or o.get('exp')
                    if isinstance(exp_txt, str) and exp_txt.strip():
                        norm_ops.append({"experience": exp_txt.strip()})
        
        # Save experiences to sample directory
        if norm_ops:
            with open(os.path.join(sample_dir, 'exp_items.json'), 'w', encoding='utf-8') as f:
                json.dump(norm_ops, f, ensure_ascii=False, indent=2)
        
        if getattr(args, 'skill_enable', False):
            skill_result = generate_skill_for_sample(sample_info, llm, args, ground_truth=ground_truth)
            if skill_result['success']:
                result['skill_content'] = skill_result['skill_content']
            else:
                print(f"  Warning: Skill generation failed for {question_id}: {skill_result.get('error')}")

        result['success'] = True
        result['experience_ops'] = norm_ops
        
    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()
    
    return result


def save_knowledge_snapshot(args, batch_idx):
    """
    Save a snapshot of current experience library and skill document before batch processing.
    
    Args:
        args: Command line arguments
        batch_idx: Current batch index (0-based)
    """
    import shutil
    from datetime import datetime
    
    snapshot_dir = os.path.join(args.output_dir, "snapshots", f"batch_{batch_idx:03d}")
    os.makedirs(snapshot_dir, exist_ok=True)
    
    metadata = {
        "batch_idx": batch_idx,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Snapshot experience library: copy if source exists, then count from snapshot file
    # so metadata always reflects the snapshot content (avoids 0 when path was wrong but snapshot has old copy)
    snapshot_exp_path = os.path.join(snapshot_dir, "experiences.json")
    if args.experience_library and os.path.exists(args.experience_library):
        shutil.copy2(args.experience_library, snapshot_exp_path)
    # Count from snapshot file when present (same snapshot we just wrote or from previous run)
    if os.path.exists(snapshot_exp_path):
        try:
            with open(snapshot_exp_path, 'r', encoding='utf-8') as f:
                exp_data = json.load(f)
            if isinstance(exp_data, list):
                metadata["experience_count"] = len(exp_data)
            elif isinstance(exp_data, dict):
                inner = exp_data.get("experiences", {})
                metadata["experience_count"] = len(inner) if isinstance(inner, (list, dict)) else 0
            else:
                metadata["experience_count"] = 0
        except Exception:
            metadata["experience_count"] = -1
    else:
        metadata["experience_count"] = 0
    
    # Snapshot skill document
    if getattr(args, 'skill_library', None) and os.path.exists(args.skill_library):
        shutil.copy2(args.skill_library, os.path.join(snapshot_dir, "SKILL.md"))
        try:
            with open(args.skill_library, 'r', encoding='utf-8') as f:
                skill_content = f.read()
            metadata["skill_word_count"] = len(skill_content.split())
        except Exception:
            metadata["skill_word_count"] = -1
    else:
        metadata["skill_word_count"] = 0
    
    # Save metadata
    with open(os.path.join(snapshot_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"  [Snapshot] Saved batch_{batch_idx:03d} snapshot (exp: {metadata['experience_count']}, skill: {metadata['skill_word_count']} words)")


def process_large_batch_experiences(samples_info, args, batch_idx=0, is_final=False):
    """
    Process experience generation for a large batch of samples in parallel,
    then merge all experiences into the library.
    
    Args:
        samples_info: List of sample_info dicts (from generate_experience_for_sample)
        args: Command line arguments
        batch_idx: Current batch index for snapshot naming
        is_final: If True, force refine regardless of threshold (for final batch)
        
    Returns:
        Number of successfully processed samples
    """
    if not samples_info:
        return 0
    start_time = time.time()
    print(f"\n[Large Batch {batch_idx}] Processing experience generation for {len(samples_info)} samples...")
    
    # Save snapshot before processing
    save_knowledge_snapshot(args, batch_idx)
    
    # Parallel experience and skill generation for all samples in the large batch
    all_experience_ops = []
    all_skill_contents = []
    successful_samples = 0
    
    gen_start = time.time()
    # Experience generation now handles skill generation internally to avoid race conditions
    num_tasks = len(samples_info)
    with ThreadPoolExecutor(max_workers=num_tasks) as executor:
        exp_futures = {
            executor.submit(generate_experience_for_sample, sample_info, args): sample_info
            for sample_info in samples_info
        }
        for future in as_completed(exp_futures):
            sample_info = exp_futures[future]
            try:
                result = future.result()
                if result['success']:
                    all_experience_ops.extend(result['experience_ops'])
                    if result.get('skill_content'):
                        all_skill_contents.append(result['skill_content'])
                    successful_samples += 1
                else:
                    print(f"  Warning: Generation failed for {result['sample_id']}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"  Warning: Generation failed for {sample_info['question_id']}: {e}")
    
    gen_time = time.time() - gen_start
    print(f"  [Timing] Experience generation: {gen_time:.1f}s")
    
    if all_experience_ops and getattr(args, 'experience_library_update', False) and args.experience_library:
        try:
            if not hasattr(process_large_batch_experiences, '_library_lock'):
                process_large_batch_experiences._library_lock = threading.Lock()
            
            with process_large_batch_experiences._library_lock:
                existing = exp_load_existing(args.experience_library)
                merge_start = time.time()
                merged = exp_batch_merge(
                    existing, all_experience_ops, ExperienceLLM(),
                    debug_dir=None,
                    experience_limit=getattr(args, 'experience_max_items', 16)
                )
                merge_time = time.time() - merge_start
                
                # Refine experience library if exceeds threshold or is final batch
                if getattr(args, 'experience_refine', False):
                    max_items = getattr(args, 'experience_max_items', 256)
                    if is_final or len(merged) > max_items:
                        refine_start = time.time()
                        refine_llm = ExperienceLLM()
                        merged = refine_experience_library(merged, refine_llm, debug_dir=args.output_dir)
                        refine_time = time.time() - refine_start
                        print(f"  [Timing] Experience refine{'(final)' if is_final else ''}: {refine_time:.1f}s")
                
                os.makedirs(os.path.dirname(args.experience_library), exist_ok=True)
                exp_save_library(args.experience_library, merged)
                print(f"  [Large Batch] Merged {len(all_experience_ops)} experience operations into library (final size: {len(merged)})")
                print(f"  [Timing] Experience merge: {merge_time:.1f}s")
        except Exception as e:
            print(f"  Warning: Failed to merge experiences into library: {e}")
            
    # Merge all skills into library (single merge operation)
    if all_skill_contents and getattr(args, 'skill_library', None):
        try:
            if not hasattr(process_large_batch_experiences, '_skill_lock'):
                process_large_batch_experiences._skill_lock = threading.Lock()
            
            with process_large_batch_experiences._skill_lock:
                existing_skill_path = args.skill_library
                existing_content = ""
                if os.path.exists(existing_skill_path):
                    with open(existing_skill_path, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                
                merge_start = time.time()
                # Use LLM for merging skills
                skill_llm = ExperienceLLM()
                merged_skill = merge_skills(existing_content, all_skill_contents, skill_llm, args)
                
                # Refine skill document if it has grown large or is final batch
                if getattr(args, 'skill_refine', False):
                    skill_max_length = getattr(args, 'skill_max_length', 1000)
                    word_count = len(merged_skill.split())
                    if is_final or word_count > skill_max_length:
                        refine_start = time.time()
                        merged_skill = refine_skill_document(merged_skill, skill_llm, skill_path=existing_skill_path, word_threshold=skill_max_length, force_refine=is_final)
                        refine_time = time.time() - refine_start
                        print(f"  [Timing] Skill refine{'(final)' if is_final else ''}: {refine_time:.1f}s")
                
                os.makedirs(os.path.dirname(existing_skill_path), exist_ok=True)
                with open(existing_skill_path, 'w', encoding='utf-8') as f:
                    f.write(merged_skill)
                
                merge_time = time.time() - merge_start
                print(f"  [Large Batch] Merged {len(all_skill_contents)} skills into library")
                print(f"  [Timing] Skill merge: {merge_time:.1f}s")
        except Exception as e:
            print(f"  Warning: Failed to merge skills into library: {e}")
    
    total_time = time.time() - start_time
    print(f"[Large Batch] Completed: {successful_samples}/{len(samples_info)} samples processed successfully (total: {total_time:.1f}s)")
    return successful_samples


def main(args):
    """
    Main function to run the API-based inference process.
    """
    # Read model configurations from environment variables (required)
    args.model_name = os.environ.get("REASONING_MODEL_NAME")
    if not args.model_name:
        raise ValueError("REASONING_MODEL_NAME environment variable must be set")
    
    print(f"Starting API-based greedy inference...")
    print(f"Reasoning model: {args.model_name}")
    
    # Validate and set experience batch parameters
    if getattr(args, 'experience_online_generate', False):
        small_batch = args.rollouts_per_sample
        large_batch = getattr(args, 'experience_large_batch', None) or small_batch
        args.experience_large_batch = large_batch
        
        # Validate: large_batch must be a multiple of small_batch
        if large_batch % small_batch != 0:
            raise ValueError(
                f"experience_large_batch ({large_batch}) must be a multiple of "
                f"rollouts_per_sample ({small_batch})"
            )
        
        args.experience_samples_per_large_batch = large_batch // small_batch
        
        print(f"\n{'='*80}")
        print(f"Experience Generation Batching:")
        print(f"  Small batch (rollouts per sample): {small_batch}")
        print(f"  Large batch (rollouts per batch): {large_batch}")
        print(f"  Samples per large batch: {args.experience_samples_per_large_batch}")
        print(f"{'='*80}\n")
    
    # Check if multi-rollout mode is enabled
    if args.rollouts_per_sample > 1:
        print(f"\n{'='*80}")
        print(f"Multi-rollout mode enabled: {args.rollouts_per_sample} rollouts per sample")
        print(f"Base seed: {args.seed_base}")
        print(f"This will enable pass@k and average@k evaluation")
        print(f"{'='*80}\n")
    
    # Load system prompt from YAML
    try:
        with open(args.inference_prompts_path, 'r', encoding='utf-8') as f:
            prompts_yaml = yaml.safe_load(f)
        SYSTEM_PROMPT = prompts_yaml['system_prompts'][args.system_prompt_key]
        print("Inference prompts loaded successfully.")
    except Exception as e:
        print(f"Error loading inference prompts from {args.inference_prompts_path}: {e}")
        return

    # Save base SYSTEM_PROMPT (without experiences) for reloading
    BASE_SYSTEM_PROMPT = SYSTEM_PROMPT

    # Set global prompts for all modules
    set_processors_prompts(SYSTEM_PROMPT)


    # Load tool configs
    try:
        tool_config_path = getattr(args, 'tool_config_path', 'eval/configs/tool_configs.yaml')
        if os.path.exists(tool_config_path):
            with open(tool_config_path, 'r', encoding='utf-8') as f:
                args.tool_configs = yaml.safe_load(f) or {}
            print(f"Loaded tool configs from {tool_config_path}")
        else:
            args.tool_configs = {}
            print(f"Tool config file not found at {tool_config_path}, using defaults")
    except Exception as e:
        print(f"Error loading tool configs: {e}")
        args.tool_configs = {}

    # Load data
    if args.input_file.endswith('.jsonl'):
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    original_count = len(data)
    
    # Apply max-samples limit if specified
    if args.max_samples is not None and args.max_samples > 0:
        data = data[:args.max_samples]
        print(f"Loaded {original_count} samples, limiting to {len(data)} samples (--max-samples={args.max_samples}).")
    else:
        print(f"Loaded {len(data)} samples.")

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_completion_tokens,  # Configurable (default: 16384)
    }

    # Prepare experience injection
    experience_retriever, experience_injection_text = initialize_experience_retriever(args)
    if experience_injection_text:
        # Traditional mode: inject all experiences (or up to max_items)
        SYSTEM_PROMPT = (experience_injection_text + "\n\n" + BASE_SYSTEM_PROMPT) if BASE_SYSTEM_PROMPT else experience_injection_text
        set_processors_prompts(SYSTEM_PROMPT)

    all_results = []
    
    # Large batch management for experience generation
    large_batch_queue = []  # Accumulate samples for large batch processing
    large_batch_idx = 0  # Counter for batch snapshots
    use_experience_batching = getattr(args, 'experience_online_generate', False)
    
    def process_single_sample_with_experience(sample, sample_idx):
        """Process a single sample including experience generation."""
        sample_args = prepare_sample_args(args)
        
        question_id, sample_dir = get_sample_metadata(sample, sample_idx, sample_args.output_dir)
        os.makedirs(sample_dir, exist_ok=True)
        
        retrieve_experiences_for_sample(
            sample, sample_args, experience_retriever, BASE_SYSTEM_PROMPT,
            question_id, sample_dir,
            update_global_prompts=True
        )
        
        # Check if sample is already completed (skip if --skip-completed is set)
        completed_info = check_sample_completed(sample_dir, sample_args)
        if completed_info:
            print(f"[Sample {sample_idx + 1}/{len(data)}] {question_id} - Already completed, skipping...")
            return {
                'sample': sample,
                'sample_idx': sample_idx,
                'question_id': question_id,
                'sample_dir': sample_dir,
                'sample_rollout_results': completed_info.get('sample_rollout_results', [])
            }
        
        sample_rollout_results = []
        
        if sample_args.rollouts_per_sample > 1:
            print(f"\n[Sample {sample_idx + 1}/{len(data)}] {question_id} - Running {sample_args.rollouts_per_sample} rollouts...")
            max_workers = min(sample_args.num_workers, sample_args.rollouts_per_sample)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all rollouts for this sample
                future_to_rollout_idx = {
                    executor.submit(run_single_rollout, sample, sample_args, sampling_params, rollout_idx): rollout_idx
                    for rollout_idx in range(sample_args.rollouts_per_sample)
                }
                for future in as_completed(future_to_rollout_idx):
                    rollout_idx = future_to_rollout_idx[future]
                    result = future.result()
                    if result:
                        result['_rollout_idx'] = rollout_idx
                        sample_rollout_results.append(result)
            
            # Save sample-level pass@k metrics
            compute_and_save_sample_summary(
                question_id, sample_rollout_results, sample_args.rollouts_per_sample, sample_dir
            )
        
        else:
            result = process_single_sample(sample, sample_args, sampling_params, rollout_idx=None)
            if result:
                sample_rollout_results.append(result)
        
        # Return sample info for batch experience generation
        # Experience generation is handled at large batch level
        return {
            'sample': sample,
            'sample_idx': sample_idx,
            'question_id': question_id,
            'sample_dir': sample_dir,
            'sample_rollout_results': sample_rollout_results
        }
    
    if args.num_workers > 1 and args.rollouts_per_sample > 1:
        print(f"\n[Parallel Mode] Processing {len(data)} samples with {args.num_workers} workers...")
        print(f"  Rollouts per sample: {args.rollouts_per_sample}")
        
        if use_experience_batching:
            max_concurrent_samples = args.num_workers // args.rollouts_per_sample
            print(f"  Max concurrent samples per batch: {max_concurrent_samples} (limited by {args.num_workers} workers)")
            
            # Process samples in batches
            sample_idx = 0
            while sample_idx < len(data):
                # Determine batch size: either max_concurrent_samples or remaining samples
                batch_size = min(max_concurrent_samples, len(data) - sample_idx)
                batch_samples = data[sample_idx:sample_idx + batch_size]
                batch_indices = list(range(sample_idx, sample_idx + batch_size))
                
                print(f"\n[Batch] Processing samples {sample_idx + 1}-{sample_idx + batch_size} ({batch_size} samples)...")
                
                batch_sample_results = {}
                for local_idx, sample in enumerate(batch_samples):
                    actual_sample_idx = batch_indices[local_idx]
                    question_id, sample_dir = get_sample_metadata(sample, actual_sample_idx, args.output_dir)
                    
                    batch_sample_results[actual_sample_idx] = {
                        'results': [],
                        'sample': sample,
                        'question_id': question_id,
                        'sample_dir': sample_dir
                    }
                
                # Execute pipeline parallel processing
                execute_pipeline_parallel_processing(
                    samples=batch_samples,
                    sample_indices=batch_indices,
                    sample_results_dict=batch_sample_results,
                    args=args,
                    experience_retriever=experience_retriever,
                    base_system_prompt=BASE_SYSTEM_PROMPT,
                    sampling_params=sampling_params,
                    run_single_rollout_func=run_single_rollout,
                    progress_desc=f"Batch {sample_idx // batch_size + 1}",
                    all_results_list=all_results
                )
                
                batch_sample_infos = []
                for actual_sample_idx in batch_indices:
                    if actual_sample_idx in batch_sample_results:
                        sample_info = batch_sample_results[actual_sample_idx]
                        sample_info_dict = {
                            'sample': sample_info['sample'],
                            'sample_idx': actual_sample_idx,
                            'question_id': sample_info['question_id'],
                            'sample_dir': sample_info['sample_dir'],
                            'sample_rollout_results': sample_info['results']
                        }
                        batch_sample_infos.append(sample_info_dict)
                
                # Accumulate for large batch experience generation
                large_batch_queue.extend(batch_sample_infos)
                
                # Check if we've reached a large batch
                if len(large_batch_queue) >= args.experience_samples_per_large_batch:
                    print(f"\n[Large Batch] Triggering experience generation for {len(large_batch_queue)} samples...")
                    process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx)
                    large_batch_idx += 1
                    large_batch_queue.clear()
                    SYSTEM_PROMPT = reload_experiences(args, BASE_SYSTEM_PROMPT, experience_retriever)
                sample_idx += batch_size
        else:
            # No-batch mode: submit all rollouts at once for maximum worker utilization
            total_rollouts = len(data) * args.rollouts_per_sample
            print(f"  Total rollouts: {total_rollouts} (no batch limit)")
            
            # Track results by sample
            sample_results = {}  # sample_idx -> {'results': [], 'sample': sample, 'question_id': str, 'sample_dir': str}
            
            # Prepare sample metadata
            for sample_idx, sample in enumerate(data):
                question_id, sample_dir = get_sample_metadata(sample, sample_idx, args.output_dir)
                
                sample_results[sample_idx] = {
                    'results': [],
                    'sample': sample,
                    'question_id': question_id,
                    'sample_dir': sample_dir
                }
            
            execute_pipeline_parallel_processing(
                samples=data,
                sample_indices=list(range(len(data))),
                sample_results_dict=sample_results,
                args=args,
                experience_retriever=experience_retriever,
                base_system_prompt=BASE_SYSTEM_PROMPT,
                sampling_params=sampling_params,
                run_single_rollout_func=run_single_rollout,
                progress_desc="Processing rollouts",
                all_results_list=all_results
            )
    
    elif args.num_workers > 1:
        print(f"\n[Parallel Mode] Processing {len(data)} samples with {args.num_workers} workers...")
        
        if use_experience_batching:
            # When experience batching is enabled, process in batches to ensure proper experience learning flow
            # Each sample uses 1 worker, so max concurrent samples = num_workers
            max_concurrent_samples = args.num_workers
            # But we want to process in large batch sizes for experience generation
            batch_size = min(args.experience_samples_per_large_batch, max_concurrent_samples)
            
            sample_idx = 0
            while sample_idx < len(data):
                # Determine actual batch size
                actual_batch_size = min(batch_size, len(data) - sample_idx)
                batch_samples = data[sample_idx:sample_idx + actual_batch_size]
                batch_indices = list(range(sample_idx, sample_idx + actual_batch_size))
                
                print(f"\n[Batch] Processing samples {sample_idx + 1}-{sample_idx + actual_batch_size} ({actual_batch_size} samples)...")
                with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = {
                        executor.submit(process_single_sample_with_experience, sample, actual_idx): actual_idx
                        for local_idx, (actual_idx, sample) in enumerate(zip(batch_indices, batch_samples))
                    }
                    
                    # Collect results as they complete - wait for ALL samples in this batch
                    batch_sample_infos = []
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {sample_idx // batch_size + 1}"):
                        actual_idx = futures[future]
                        try:
                            sample_info = future.result()
                            if sample_info and sample_info.get('sample_rollout_results'):
                                all_results.extend(sample_info['sample_rollout_results'])
                                batch_sample_infos.append(sample_info)
                        except Exception as e:
                            print(f"  Error processing sample {actual_idx}: {e}")
                
                if use_experience_batching and batch_sample_infos:
                    large_batch_queue.extend(batch_sample_infos)
                    
                    # Check if we've reached a large batch
                    if len(large_batch_queue) >= args.experience_samples_per_large_batch:
                        print(f"\n[Large Batch] Triggering experience generation for {len(large_batch_queue)} samples...")
                        process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx)
                        large_batch_idx += 1
                        large_batch_queue.clear()
                        SYSTEM_PROMPT = reload_experiences(args, BASE_SYSTEM_PROMPT, experience_retriever)
                sample_idx += actual_batch_size
            if use_experience_batching and large_batch_queue:
                print(f"\n[Final Batch] Processing remaining {len(large_batch_queue)} samples...")
                process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx, is_final=True)
                large_batch_idx += 1
                large_batch_queue.clear()
        else:
            # When experience batching is disabled, process all samples in parallel for maximum efficiency
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                # Submit all samples for parallel processing
                futures = {
                    executor.submit(process_single_sample_with_experience, sample, sample_idx): sample_idx
                    for sample_idx, sample in enumerate(data)
                }
                
                for future in tqdm(as_completed(futures), total=len(data), desc="Processing samples"):
                    sample_idx = futures[future]
                    try:
                        sample_info = future.result()
                        if sample_info and sample_info.get('sample_rollout_results'):
                            all_results.extend(sample_info['sample_rollout_results'])
                    except Exception as e:
                        print(f"  Error processing sample {sample_idx}: {e}")
    
    else:
        # Sequential mode (single worker)
        for sample_idx, sample in enumerate(tqdm(data, desc="Processing samples")):
            sample_info = process_single_sample_with_experience(sample, sample_idx)
            if sample_info and sample_info.get('sample_rollout_results'):
                all_results.extend(sample_info['sample_rollout_results'])
                if use_experience_batching:
                    large_batch_queue.append(sample_info)
                    if len(large_batch_queue) >= args.experience_samples_per_large_batch:
                        process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx)
                        large_batch_idx += 1
                        large_batch_queue.clear()
                        # Reload experiences for next batch if experience injection is enabled
                        SYSTEM_PROMPT = reload_experiences(args, BASE_SYSTEM_PROMPT, experience_retriever)
    
    # Process remaining samples in the queue (if any)
    if use_experience_batching and large_batch_queue:
        print(f"\n[Final Batch] Processing remaining {len(large_batch_queue)} samples...")
        process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx, is_final=True)
        large_batch_idx += 1
        large_batch_queue.clear()

    save_results(all_results, args.output_dir)
    print_summary(all_results, args.output_dir)
    if args.rollouts_per_sample > 1:
        print(f"\n{'='*80}")
        print(f"Aggregating pass@k and average@k metrics across all samples")
        print(f"{'='*80}\n")
        compute_dataset_summary(data, args)
    
    print("API-based inference finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API-based inference script.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output results.")
    parser.add_argument("--skip-completed", action='store_true', 
                       help="Skip samples that are already completed (have traj.jsonl with turn_idx and metrics.json).")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (default: None, process all samples).")
    
    # Sampling Arguments
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for sampling.")
    parser.add_argument("--max-completion-tokens", type=int, default=8192, help="Maximum tokens for single model response (aligned with Baseline framework).")
    
    # Inference Arguments
    parser.add_argument("--max-turns", type=int, default=16, help="Maximum number of turns for inference.")
    parser.add_argument("--max-images", type=int, default=16, help="Maximum number of images per sample.")
    parser.add_argument("--max-total-tokens", type=int, default=8000, help="Maximum total tokens for context.")
    parser.add_argument("--max-pixels", type=int, default=2000000, help="Maximum pixels for image processing.")
    parser.add_argument("--min-pixels", type=int, default=40000, help="Minimum pixels for image processing.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for parallel processing.")
    parser.add_argument("--rollouts-per-sample", type=int, default=4, help="Number of independent rollouts per sample for pass@k evaluation.")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed for reproducibility. Each rollout uses seed_base + rollout_index.")
    
    # Prompt Arguments
    parser.add_argument("--inference-prompts-path", type=str, default="eval/prompts/inference_prompts.yaml", 
                       help="Path to the inference prompts YAML file.")
    parser.add_argument("--system-prompt-key", type=str, default="multi_tool_agent_search", 
                       choices=['direct_cot', 'agent_zoom', 'multi_tool_agent', 'multi_tool_agent_search', 'multi_tool_agent_code'], 
                       help="The key for the system prompt to use from the inference prompts YAML file.")
    
    # Tool Arguments
    parser.add_argument("--tool-config-path", type=str, default="eval/configs/tool_configs.yaml",
                       help="Path to the tool configuration YAML file.")
    parser.add_argument("--image-search-max-calls", type=int, default=3,
                       help="Maximum number of image_search tool calls per sample (default: 3)")
    parser.add_argument("--web-search-max-calls", type=int, default=5,
                       help="Maximum number of web_search tool calls per sample (default: 5)")

    # Experience Arguments
    parser.add_argument("--experience-enable", action='store_true', 
                       help="Enable experience injection from library")
    parser.add_argument("--experience-library", type=str, default=None, 
                       help="Path to consolidated experiences JSON")
    parser.add_argument("--experience-online-generate", action='store_true', 
                       help="Generate per-sample experiences after inference")
    parser.add_argument("--experience-library-update", action='store_true', 
                       help="Merge per-sample experiences back into the library")
    parser.add_argument("--experience-max-ops", type=int, default=2, 
                       help="Max operations per sample during critique")
    parser.add_argument("--experience-temperature", type=float, default=0.6, 
                       help="Temperature for experience summarization/merge (default 0.6)")
    parser.add_argument("--experience-large-batch", type=int, default=None,
                        help="Large batch size for experience generation (number of rollouts to trigger batch processing). "
                             "Must be a multiple of rollouts_per_sample. If None, equals rollouts_per_sample (no batching).")
    parser.add_argument("--experience-max-items", type=int, default=120,
                        help="Max number of experiences to inject/keep in library")
    parser.add_argument("--experience-refine", action='store_true',
                       help="Enable experience library refinement after merge (uses --experience-max-items as threshold)")
    
    # Skill Arguments
    parser.add_argument("--skill-enable", action='store_true', 
                       help="Enable dynamic skill generation")
    parser.add_argument("--skill-library", type=str, default=None, 
                       help="Path to the global skill library (SKILL.md)")
    parser.add_argument("--skill-inference", action='store_true', 
                       help="Enable skill injection during inference")
    parser.add_argument("--no-skill-adaptation", action='store_false', dest='skill_adaptation',
                       default=True, help="Disable skill adaptation (use raw skill directly, default: adapt)")
    parser.add_argument("--skill-refine", action='store_true',
                       help="Enable skill refinement after merge (trim and consolidate)")
    parser.add_argument("--skill-max-length", type=int, default=1000,
                       help="Word count threshold to trigger skill refinement (default: 1000)")
    
    # Experience Retrieval Arguments
    parser.add_argument("--experience-retrieval", action='store_true',
                        help="Enable retrieval-based experience injection (default: False, uses all experiences)")
    parser.add_argument("--experience-retrieval-top-k", type=int, default=3,
                        help="Number of top experiences to retrieve per query (default: 3)")
    parser.add_argument("--experience-retrieval-min-similarity", type=float, default=0.0,
                        help="Minimum similarity threshold for retrieved experiences (0.0 to 1.0, default: 0.0)")
    parser.add_argument("--experience-embedding-api-key", type=str, default=None,
                        help="API key for embedding service (defaults to EXPERIENCE_EMBEDDING_API_KEY or OPENAI_API_KEY)")
    parser.add_argument("--experience-embedding-endpoint", type=str, default=None,
                        help="API endpoint for embedding service (defaults to EXPERIENCE_EMBEDDING_ENDPOINT or OPENAI_API_BASE)")
    parser.add_argument("--no-experience-embedding-cache", action='store_false', dest='experience_embedding_cache_enable',
                       default=True, help="Disable disk caching of experience embeddings (default: enabled)")
    parser.add_argument("--experience-retrieval-decomposition", action='store_true',
                        help="Enable task decomposition for retrieval (uses LLM to decompose task into subtasks, then retrieves for each) (default: False)")
    parser.add_argument("--experience-retrieval-rewrite", action='store_true',
                        help="Enable experience rewrite to adapt retrieved experiences to the current task (uses LLM to rewrite experiences) (default: False)")
    
    
    args = parser.parse_args()
    main(args)
