import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from .llm_client import ExperienceLLM


# --------- Constants ---------

# Token configuration
MAX_TOKENS = 2048  # Max tokens for image caption generation

# Concurrency configuration
MAX_WORKERS = 8  # Maximum number of concurrent workers for image caption generation


# --------- Image Caption Generation ---------

# Global prompt for image description
IMAGE_DESCRIPTION_PROMPT = "Please provide a detailed visual description and analysis of this image. The background context above is provided to help you understand the image's purpose and origin, which should guide you to focus on relevant visual details. Describe key objects, their spatial relationships, notable visual details, and any observable actions or events. Highlight how this image contributes to the overall task. Format your response as a single paragraph (no headings, no bullet points, no problem-solving steps), written in clear English, with no more than 200 words."


def _caption_cache_path(
    sample_path: Path, rollout_dir: str, filename: str, cache_dir: Optional[str]
) -> Optional[Path]:
    """Return path for caption cache file, or None if caching disabled for this case."""
    if rollout_dir:
        return sample_path / rollout_dir / ".analysis_cache" / f"{Path(filename).stem}_analysis.txt"
    if cache_dir:
        return Path(cache_dir) / f"{Path(filename).stem}_analysis.txt"
    return None


def _generate_single_image_caption(
    image_path: Path, 
    llm: ExperienceLLM,
    prompt: Optional[str] = None,
    task_context: Optional[Dict[str, Any]] = None,
    generation_context: Optional[str] = None,
    original_image_paths: Optional[List[Path]] = None
) -> str:
    """
    Generate a detailed caption for a single image using ExperienceLLM with context.
    
    Args:
        image_path: Path to the image file
        llm: ExperienceLLM instance for API calls
        prompt: Optional custom prompt (default: detailed image description prompt)
        task_context: Optional dict with 'question' and 'system_prompt' keys
        generation_context: Optional text describing how the image was generated (previous reasoning/tool calls)
        original_image_paths: Optional list of Path objects to original question images (for tool-generated images)
        
    Returns:
        Caption text string (or error placeholder on failure)
    """
    if not image_path.exists():
        return f"[Image file not found: {image_path.name}]"
    
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Build prompt with context
        if prompt is None:
            prompt_parts = []
            
            # Add task context if available
            if task_context:
                context_info = []
                if task_context.get("system_prompt"):
                    context_info.append(f"Task: {task_context['system_prompt']}")
                if task_context.get("question"):
                    context_info.append(f"Question: {task_context['question']}")
                
                if context_info:
                    prompt_parts.append("Background context (to help you understand the image's purpose and origin): " + " | ".join(context_info))
            
            # Add generation context if available
            if generation_context:
                prompt_parts.append(f"Image generation context (how this image was created): {generation_context}")
            
            # Add note about original images if provided
            if original_image_paths:
                prompt_parts.append("Note: The original question image(s) are provided below for reference. Please describe the current image (the tool-generated image) in relation to the question context.")
            
            # Add image description request with format specification
            prompt_parts.append(IMAGE_DESCRIPTION_PROMPT)
            
            prompt = "\n".join(prompt_parts)
        
        # Prepare images list: main image first, then original images if provided
        images_to_send = [image]
        
        if original_image_paths:
            # Load original images
            for orig_path in original_image_paths:
                if orig_path.exists():
                    try:
                        orig_img = Image.open(orig_path).convert('RGB')
                        images_to_send.append(orig_img)
                    except Exception as e:
                        print(f"  Warning: Failed to load original image {orig_path}: {e}")
        
        # chat_with_image accepts single image or list
        return llm.chat_with_image(
            prompt=prompt,
            image=images_to_send,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            return_placeholder_on_error=True
        )
    except Exception as e:
        return f"[Error generating caption: {str(e)}]"


def generate_image_captions(
    image_filenames: List[str],
    sample_dir: str,
    llm: Optional[ExperienceLLM] = None,
    cache_dir: Optional[str] = None,
    image_contexts: Optional[Dict[str, Dict[str, Any]]] = None,
    image_to_rollout: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Generate captions for a list of image files using ExperienceLLM with context.
    Only processes files that actually exist in the sample directory.
    
    Args:
        image_filenames: List of image filenames (e.g., ["tool_image_1.jpg", "search_image_2.jpg"])
        sample_dir: Directory containing the images
        llm: Optional ExperienceLLM instance (creates new one if not provided)
        cache_dir: Optional base directory for caching (actual cache will be in rollout subdirs)
        image_contexts: Optional dict mapping image filename to context dict with:
            - "task_context": dict with "question" and "system_prompt" keys
            - "generation_context": str describing how image was generated (previous reasoning/tool calls)
        image_to_rollout: Optional dict mapping image filename to rollout directory (e.g., "rollout_0")
        
    Returns:
        Dictionary mapping "{rollout_dir}/{filename}" to caption text for multi-rollout images,
        or just "{filename}" for single rollout or top-level images
    """
    if not image_filenames:
        return {}
    
    # Create ExperienceLLM instance if not provided
    if llm is None:
        llm = ExperienceLLM()
    
    sample_path = Path(sample_dir)
    # Filter to only existing files and normalize filenames
    # Search in both top-level sample_dir and rollout_X subdirectories
    existing_files = []  # List of tuples: (caption_key, rollout_dir, filename)
    for key in image_filenames:
        # key may be "rollout_X/filename" or just "filename"
        if "/" in key:
            rollout_hint, raw_filename = key.split("/", 1)
        else:
            rollout_hint, raw_filename = "", key
        
        clean_filename = os.path.basename(raw_filename)
        
        # Try different variations of the filename
        candidates = []
        if clean_filename.endswith(('.jpg', '.jpeg', '.png')):
            candidates.append(clean_filename)
        else:
            candidates.append(f"{clean_filename}.jpg")
            candidates.append(clean_filename)
        
        # Try each candidate in multiple locations
        found = False
        for candidate in candidates:
            # First check rollout hint or mapping
            rollout_dirs = []
            if rollout_hint:
                rollout_dirs.append(rollout_hint)
            if image_to_rollout and key in image_to_rollout:
                rollout_dirs.append(image_to_rollout[key])
            
            for rollout_dir in rollout_dirs:
                if rollout_dir:
                    rollout_path = sample_path / rollout_dir / candidate
                    if rollout_path.exists() and rollout_path.is_file():
                        caption_key = f"{rollout_dir}/{candidate}"
                        existing_files.append((caption_key, rollout_dir, candidate, rollout_path))
                        found = True
                        break
            if found:
                break

            # Try top-level sample_dir
            top_path = sample_path / candidate
            if top_path.exists() and top_path.is_file():
                existing_files.append((candidate, "", candidate, top_path))
                found = True
                break

            # If not found, try in rollout_X subdirectories
            if not found:
                for item in sample_path.iterdir():
                    if item.is_dir() and item.name.startswith("rollout_"):
                        rollout_image_path = item / candidate
                        if rollout_image_path.exists() and rollout_image_path.is_file():
                            caption_key = f"{item.name}/{candidate}"
                            existing_files.append((caption_key, item.name, candidate, rollout_image_path))
                            found = True
                            break
                if found:
                    break
        
        if not found:
            # Debug output for missing files
            print(f"    Warning: Image file not found: {clean_filename} in {sample_dir} (checked top-level and rollout_* subdirectories)")
    
    if not existing_files:
        return {}
    
    # Separate cached and uncached images (image_path already resolved in existing_files)
    captions = {}
    to_generate = []

    for caption_key, rollout_dir, filename, image_path in existing_files:
        if not image_path.exists() or not image_path.is_file():
            print(f"  Warning: Image file disappeared: {filename}")
            captions[caption_key] = f"[Image file not found: {filename}]"
            continue

        cache_path = _caption_cache_path(sample_path, rollout_dir, filename, cache_dir)
        if cache_path and cache_path.exists():
            try:
                cached_caption = cache_path.read_text(encoding="utf-8").strip()
                if cached_caption:
                    captions[caption_key] = cached_caption
                    continue
            except Exception:
                pass

        task_context = None
        generation_context = None
        if image_contexts and caption_key in image_contexts:
            context = image_contexts[caption_key]
            task_context = context.get("task_context")
            generation_context = context.get("generation_context")

        to_generate.append((caption_key, image_path, task_context, generation_context, rollout_dir, filename))
    
    # Generate captions concurrently
    if to_generate:
        def _generate_with_cache(args):
            """Helper function to generate caption and save to cache."""
            caption_key, image_path, task_context, generation_context, rollout_dir, filename = args
            original_image_paths = None
            if task_context and "original_image_paths" in task_context:
                original_image_paths = task_context["original_image_paths"]
            caption = _generate_single_image_caption(
                image_path,
                llm,
                task_context=task_context,
                generation_context=generation_context,
                original_image_paths=original_image_paths
            )
            if caption and not caption.startswith("[Error") and not caption.startswith("[Failed"):
                try:
                    cache_path = _caption_cache_path(sample_path, rollout_dir, filename, cache_dir)
                    if cache_path:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(caption, encoding="utf-8")
                except Exception as e:
                    print(f"  Warning: Failed to save caption cache for {caption_key}: {e}")
            return caption_key, caption
        
        # Use ThreadPoolExecutor for concurrent generation
        max_workers = min(len(to_generate), MAX_WORKERS)  # Limit concurrent requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_key = {
                executor.submit(_generate_with_cache, args): args[0] 
                for args in to_generate
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_key):
                try:
                    caption_key, caption = future.result()
                    captions[caption_key] = caption
                except Exception as e:
                    caption_key = future_to_key[future]
                    print(f"  Warning: Failed to generate caption for {caption_key}: {e}")
                    captions[caption_key] = f"[Error generating caption: {str(e)}]"
    
    return captions


