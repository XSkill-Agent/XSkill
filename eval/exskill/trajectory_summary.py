import os
import json
import re
import time
from pathlib import Path
from typing import Optional, Dict, List, Union, Any
from PIL import Image

from prompts.experience_prompts import (
    SINGLE_ROLLOUT_SUMMARY,
)

from .llm_client import ExperienceLLM
from .multimodal_analysis import generate_image_captions


# --------- Constants ---------

# Retry configuration
MAX_RETRIES = 3

# Token configuration
MAX_TOKENS = 12288  # Max tokens for trajectory summarization


# --------- Summarization ---------


def _normalize_traj_paths(traj_paths: Union[str, List[str]]) -> List[str]:
    """Normalize trajectory paths to a list (single path -> [path], list unchanged)."""
    if isinstance(traj_paths, str):
        return [traj_paths]
    return list(traj_paths) if traj_paths else []


def _scan_all_images(
    sample_dir: str
) -> Dict[str, Dict[str, Any]]:
    """Scan all image files in the sample directory and identify original_image files.
    
    Args:
        sample_dir: Path to the sample directory
        
    Returns:
        Dictionary mapping caption_key (e.g., "rollout_0/tool_image_1.jpg" or "original_image.jpg") 
        to dict with:
            - "file_path": Path object to the image file
            - "rollout_dir": Rollout directory name (e.g., "rollout_0") or empty string for top-level
            - "is_original": True if this is an original_image* file, False otherwise
    """
    sample_path = Path(sample_dir)
    if not sample_path.exists() or not sample_path.is_dir():
        return {}
    
    image_map = {}
    
    # Scan top-level directory
    for img_file in sample_path.glob("*"):
        if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            filename = img_file.name
            is_original = filename.startswith('original_image')
            image_map[filename] = {
                "file_path": img_file,
                "rollout_dir": "",
                "is_original": is_original
            }
    
    # Scan rollout_X subdirectories
    for item in sample_path.iterdir():
        if item.is_dir() and item.name.startswith("rollout_"):
            rollout_dir = item.name
            for img_file in item.glob("*"):
                if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    filename = img_file.name
                    is_original = filename.startswith('original_image')
                    caption_key = f"{rollout_dir}/{filename}"
                    image_map[caption_key] = {
                        "file_path": img_file,
                        "rollout_dir": rollout_dir,
                        "is_original": is_original
                    }
    
    return image_map


def _find_original_images_for_tool_image(
    tool_image_key: str,
    tool_image_rollout: str,
    all_images: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Find original_image files that should be included in task_context for a tool-generated image.
    
    For a tool-generated image in rollout_X, we look for original_image* files in:
    1. The same rollout directory (rollout_X/original_image*.jpg)
    2. The top-level directory (original_image*.jpg)
    
    Args:
        tool_image_key: Caption key for the tool image (e.g., "rollout_0/tool_image_1.jpg")
        tool_image_rollout: Rollout directory for the tool image (e.g., "rollout_0")
        all_images: Dictionary from _scan_all_images
        
    Returns:
        List of caption keys for original_image files to include
    """
    original_image_keys = []
    
    # First, look in the same rollout directory
    if tool_image_rollout:
        for key, info in all_images.items():
            if info["is_original"] and info["rollout_dir"] == tool_image_rollout:
                original_image_keys.append(key)
    
    # Also look in top-level directory
    for key, info in all_images.items():
        if info["is_original"] and info["rollout_dir"] == "":
            if key not in original_image_keys:
                original_image_keys.append(key)
    
    same_rollout = []
    top_level = []
    for key, info in all_images.items():
        if not info["is_original"]:
            continue
        if info["rollout_dir"] == tool_image_rollout:
            same_rollout.append(key)
        elif info["rollout_dir"] == "" and key not in same_rollout:
            top_level.append(key)
    return same_rollout + top_level


def _get_caption_for_image(name_or_path: str, image_captions: Dict[str, str]) -> Optional[str]:
    """Find caption for an image by name or path (exact, basename, no-ext, partial match)."""
    if not name_or_path or not image_captions:
        return None
    if name_or_path in image_captions:
        return image_captions[name_or_path]
    basename = os.path.basename(name_or_path)
    if basename in image_captions:
        return image_captions[basename]
    name_no_ext = os.path.splitext(name_or_path)[0]
    for key in image_captions:
        if os.path.splitext(key)[0] == name_no_ext:
            return image_captions[key]
    for key in image_captions:
        if name_or_path in key or os.path.basename(key) == name_or_path:
            return image_captions[key]
    return None


def _replace_image_refs_in_jsonl(
    jsonl_content: str, 
    image_captions: Dict[str, str]
) -> str:
    """Replace image references in raw JSONL content with captions.
    
    This function processes the raw JSONL file and:
    1. Replaces markdown image references in tool_call.result with captions
    2. Adds caption field to tool_image entries
    
    Args:
        jsonl_content: Raw JSONL file content as string
        image_captions: Dictionary mapping image filename to caption text
        
    Returns:
        Processed JSONL content with image references replaced
    """
    if not jsonl_content or not image_captions:
        return jsonl_content
    
    lines = []
    for line in jsonl_content.split('\n'):
        if not line.strip():
            lines.append(line)
            continue
        
        try:
            rec = json.loads(line)
            modified = False
            
            # Replace image references in tool_call.result
            if 'tool_call' in rec:
                tool_call_data = rec['tool_call']
                if 'result' in tool_call_data and tool_call_data['result']:
                    result = str(tool_call_data['result'])
                    # Pattern to match markdown image references: ![image_name.jpg](image_name.jpg)
                    image_pattern = r'!\[([^\]]+\.(?:jpg|jpeg|png))\]\(([^\)]+\.(?:jpg|jpeg|png))\)'
                    
                    def replace_with_caption(match):
                        img_name = match.group(1) or match.group(2)
                        caption = _get_caption_for_image(img_name, image_captions)
                        if caption:
                            return f"[Image: {img_name}]\nCaption: {caption}"
                        return match.group(0)
                    
                    new_result = re.sub(image_pattern, replace_with_caption, result)
                    if new_result != result:
                        tool_call_data['result'] = new_result
                        modified = True
            
            # Add caption field to tool_image entries
            if 'tool_image' in rec:
                tool_image_data = rec['tool_image']
                image_name = tool_image_data.get('image_name', '')
                file_path = tool_image_data.get('file_path', '')
                caption = _get_caption_for_image(image_name, image_captions) or (_get_caption_for_image(file_path, image_captions) if file_path else None)
                if caption:
                    tool_image_data['caption'] = caption
                    modified = True
            
            # Only re-serialize if modified
            if modified:
                lines.append(json.dumps(rec, ensure_ascii=False))
            else:
                lines.append(line)
        except (json.JSONDecodeError, Exception):
            # If JSON parsing fails, keep the line as-is
            lines.append(line)
    
    return '\n'.join(lines)


def _replace_image_tags_in_question(question: str, image_captions: Dict[str, str]) -> str:
    """Replace <image> tags in question with corresponding captions.
    
    Args:
        question: Question text that may contain <image> tags
        image_captions: Dictionary mapping image filename to caption text
    
    Returns:
        Question text with <image> tags replaced by captions
    """
    if not question or not image_captions:
        return question
    
    # Find all <image> tags
    image_tag_pattern = r'<image>'
    matches = list(re.finditer(image_tag_pattern, question))
    
    if not matches:
        return question
    
    # Collect original_image captions (sorted by filename for consistent ordering)
    # Note: image_captions keys may be "rollout_X/original_image.jpg" or just "original_image.jpg"
    original_captions = []
    for caption_key in sorted(image_captions.keys()):
        # Extract basename to handle rollout-aware keys (e.g., "rollout_0/original_image.jpg")
        basename = os.path.basename(caption_key)
        if basename.startswith('original_image'):
            caption = image_captions[caption_key]
            if caption:
                original_captions.append(caption)
    
    if not original_captions:
        return question
    
    # Replace each <image> tag with corresponding caption
    result_parts = []
    last_end = 0
    
    for idx, match in enumerate(matches):
        # Add text before this match
        result_parts.append(question[last_end:match.start()])
        
        # Replace <image> with caption if available (original_captions is non-empty here)
        if idx < len(original_captions):
            result_parts.append(f"[Image: {original_captions[idx]}]")
        else:
            result_parts.append(f"[Image: {original_captions[-1]}]")
        
        last_end = match.end()
    
    # Add remaining text
    result_parts.append(question[last_end:])
    
    return ''.join(result_parts)


def _resolve_sample_dir(traj_paths: Union[str, List[str]], provided_sample_dir: Optional[str] = None) -> str:
    """Resolve the top-level sample directory from trajectory paths.
    
    Priority:
    1. If provided_sample_dir is given, use it
    2. If single path: check if in rollout_X/, go up to top level
    3. If multiple paths: infer from first path
    
    Args:
        traj_paths: Single path (str) or list of paths
        provided_sample_dir: Explicitly provided sample directory
    
    Returns:
        Top-level sample directory path
    """
    if provided_sample_dir and os.path.exists(provided_sample_dir):
        return provided_sample_dir
    paths = _normalize_traj_paths(traj_paths)
    if not paths:
        raise ValueError("No trajectory paths provided")
    
    first_path = paths[0]
    first_dir = os.path.dirname(first_path)
    
    # Check if we're in a rollout_X/ subdirectory
    dir_name = os.path.basename(first_dir)
    if dir_name.startswith("rollout_"):
        # Go up one level to get sample_dir
        sample_dir = os.path.dirname(first_dir)
    else:
        # Already at top level
        sample_dir = first_dir
    
    return sample_dir


def _load_metadata(traj_paths: Union[str, List[str]], sample_dir: str) -> dict:
    """Load metadata (question, ground_truth, system_prompt) from trajectory and files.
    
    Args:
        traj_paths: Single path (str) or list of paths
        sample_dir: Top-level sample directory
    
    Returns:
        {
            "question": str,
            "ground_truth": str,
            "system_prompt": str
        }
    """
    paths = _normalize_traj_paths(traj_paths)
    question = None
    ground_truth = None
    
    # Try to load from first trajectory file
    if paths:
        try:
            with open(paths[0], "r", encoding="utf-8") as f:
                first = json.loads(next(f))
                question = first.get("initial_prompt")
                ground_truth = first.get("ground_truth")
        except Exception:
            pass
    
    # Try to load system prompt from sample_dir
    system_prompt_text = None
    try:
        # First try sample_dir itself
        sp_path = os.path.join(sample_dir, "injected_system_prompt.txt")
        if os.path.exists(sp_path):
            with open(sp_path, "r", encoding="utf-8") as f:
                system_prompt_text = f.read().strip()
        else:
            # Try in rollout subdirectories
            for item in os.listdir(sample_dir):
                rollout_dir = os.path.join(sample_dir, item)
                if os.path.isdir(rollout_dir) and item.startswith("rollout_"):
                    sp_path = os.path.join(rollout_dir, "injected_system_prompt.txt")
                    if os.path.exists(sp_path):
                        with open(sp_path, "r", encoding="utf-8") as f:
                            system_prompt_text = f.read().strip()
                        break
    except Exception:
        pass
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "system_prompt": system_prompt_text
    }


def summarize_rollouts(traj_paths: Union[str, List[str]], llm: ExperienceLLM, sample_dir: Optional[str] = None) -> dict:
    """Unified function to summarize single or multiple trajectory files.
    
    This function handles both single and multiple trajectory files.
    It loads trajectories with tool calls and results, formats them consistently,
    and generates summaries using LLM.
    
    Args:
        traj_paths: Single trajectory path (str) or list of paths (List[str])
        llm: ExperienceLLM instance for generating summary
        sample_dir: Top-level sample directory (optional, will be inferred if not provided)
    
    Returns:
        {
            "process": summary_text,
            "question": question,
            "ground_truth": groundtruth,
            "system_prompt": system_prompt_text
        }
    """
    paths = _normalize_traj_paths(traj_paths)
    if not paths:
        return {}
    uniq_paths = list(dict.fromkeys(paths))
    is_single = len(uniq_paths) == 1
    
    # Resolve sample directory
    try:
        resolved_sample_dir = _resolve_sample_dir(uniq_paths, sample_dir)
    except Exception as e:
        print(f"  Warning: Failed to resolve sample_dir: {e}")
        # Fallback: use first path's directory
        resolved_sample_dir = os.path.dirname(uniq_paths[0])
    
    # Load metadata
    metadata = _load_metadata(uniq_paths, resolved_sample_dir)
    question = metadata.get("question")
    ground_truth = metadata.get("ground_truth")
    system_prompt_text = metadata.get("system_prompt")
    
    # Scan all images in the directory
    all_images = _scan_all_images(resolved_sample_dir)
    
    if not all_images:
        print(f"  Warning: No images found in {resolved_sample_dir}")
        image_captions = {}
    else:
        print(f"  Found {len(all_images)} images in directory")
        
        # Prepare base task context
        base_task_context = {}
        if question:
            base_task_context["question"] = question
        if system_prompt_text:
            base_task_context["system_prompt"] = system_prompt_text
        
        # Build image_contexts: for tool-generated images, include original_image info
        image_contexts: Dict[str, Dict[str, Any]] = {}
        image_to_rollout: Dict[str, str] = {}
        existing_image_filenames = []
        
        for caption_key, image_info in all_images.items():
            file_path = image_info["file_path"]
            rollout_dir = image_info["rollout_dir"]
            is_original = image_info["is_original"]
            
            # Build task_context for this image
            task_context = base_task_context.copy()
            
            # For tool-generated images, find and include original_image paths
            if not is_original:
                original_image_keys = _find_original_images_for_tool_image(
                    caption_key, rollout_dir, all_images
                )
                if original_image_keys:
                    # Get the actual Path objects for original images
                    original_image_paths = []
                    for orig_key in original_image_keys:
                        orig_info = all_images.get(orig_key)
                        if orig_info:
                            original_image_paths.append(orig_info["file_path"])
                    if original_image_paths:
                        task_context["original_image_paths"] = original_image_paths
            
            image_contexts[caption_key] = {
                "task_context": task_context,
                "generation_context": None  # Simplified: no generation context
            }
            image_to_rollout[caption_key] = rollout_dir
            existing_image_filenames.append(caption_key)
        
        # Generate analyses for all images
        print(f"  Generating analysis for {len(existing_image_filenames)} images...")
        analysis_start = time.time()
        image_captions = generate_image_captions(
            existing_image_filenames,
            resolved_sample_dir,
            llm=llm,
            cache_dir=os.path.join(resolved_sample_dir, ".analysis_cache"),
            image_contexts=image_contexts,
            image_to_rollout=image_to_rollout
        )
        analysis_time = time.time() - analysis_start
        print(f"  [Timing] Image analysis generation: {analysis_time:.1f}s")
    
    # Precompute per-rollout caption maps (and top-level) for O(1) lookup per path
    top_level_captions: Dict[str, str] = {}
    rollout_captions_map: Dict[str, Dict[str, str]] = {}
    for caption_key, caption_text in image_captions.items():
        if '/' in caption_key:
            key_rollout, filename = caption_key.split('/', 1)
            if key_rollout not in rollout_captions_map:
                rollout_captions_map[key_rollout] = {}
            rollout_captions_map[key_rollout][caption_key] = caption_text
            rollout_captions_map[key_rollout][filename] = caption_text
        else:
            top_level_captions[caption_key] = caption_text

    formatted_blocks = []
    for rollout_idx, path in enumerate(uniq_paths):
        rollout_dir = Path(path).parent.name
        rollout_captions = {**top_level_captions, **rollout_captions_map.get(rollout_dir, {})}
        # Read raw JSONL content
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_jsonl_content = f.read()
        except Exception as e:
            print(f"  Warning: Failed to read trajectory file {path}: {e}")
            continue
        
        if not raw_jsonl_content.strip():
            continue
        
        # Replace image references with captions
        processed_content = _replace_image_refs_in_jsonl(raw_jsonl_content, rollout_captions)
        
        # For multiple rollouts, add header
        if not is_single:
            formatted_blocks.append(f"==== Rollout {rollout_idx} ({rollout_dir}) ====\n{processed_content}")
        else:
            formatted_blocks.append(processed_content)
    
    if not formatted_blocks:
        return {}
    
    process_text = "\n\n".join(formatted_blocks)
    header_prefix = "==== " if not is_single else ""
    header_parts = []
    if isinstance(question, str) and question.strip():
        formatted_question = _replace_image_tags_in_question(question.strip(), image_captions)
        header_parts.append(f"{header_prefix}Question:\n{formatted_question}")
    if isinstance(system_prompt_text, str) and system_prompt_text.strip():
        header_parts.append(f"{header_prefix}System prompt:\n{system_prompt_text}")
    if isinstance(ground_truth, str) and ground_truth.strip():
        header_parts.append(f"{header_prefix}Ground truth (for reference):\n{ground_truth.strip()}")
    
    if header_parts:
        process_text = "\n\n".join(header_parts) + "\n\n" + process_text
    
    # Generate summary (always include ground truth context if available)
    prompt = SINGLE_ROLLOUT_SUMMARY.format(trajectory=process_text)
    
    # Find question images (top-level original_image files)
    question_images = []
    if all_images:
        for caption_key, image_info in all_images.items():
            if image_info["is_original"] and image_info["rollout_dir"] == "":
                question_images.append(image_info["file_path"])
    
    # Call LLM with retry mechanism (with images if available)
    response_text = None
    llm_start = time.time()
    for attempt in range(MAX_RETRIES):
        try:
            if question_images:
                pil_images = [Image.open(img_path).convert('RGB') for img_path in question_images]
                response_text = llm.chat_with_image(
                    prompt=prompt,
                    image=pil_images,
                    max_tokens=MAX_TOKENS,
                    return_placeholder_on_error=False
                )
            else:
                # Fallback to text-only
                response_text = llm.chat(prompt, max_tokens=MAX_TOKENS)
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  Warning: Summary generation failed (attempt {attempt+1}/{MAX_RETRIES}): {e}, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    if response_text is None:
        raise RuntimeError("Failed to generate summary after all retries")
    
    llm_time = time.time() - llm_start
    print(f"  [Timing] exp_summary_resp generation: {llm_time:.1f}s")
    
    # Save summary files
    try:
        # Unified naming: always use exp_summary_* regardless of rollout count
        prompt_file = os.path.join(resolved_sample_dir, "exp_summary_prompt.txt")
        resp_file = os.path.join(resolved_sample_dir, "exp_summary_resp.txt")
        
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(resp_file, "w", encoding="utf-8") as f:
            f.write(response_text)
    except Exception:
        pass
    
    # Return summary along with metadata
    return {
        "process": response_text,
        "question": question,
        "ground_truth": ground_truth,
        "system_prompt": system_prompt_text
    }
