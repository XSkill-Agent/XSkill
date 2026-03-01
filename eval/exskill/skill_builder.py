import os
import re
from typing import List, Dict, Any, Optional
from prompts.skill_prompts import GENERATE_SKILL_PROMPT, MERGE_SKILL_PROMPT, SKILL_REFINE_PROMPT
from prompts.skill_prompts_test_time import ADAPT_SKILL_PROMPT

def extract_trajectory_from_file(sample_dir: str) -> Optional[str]:
    """
    Extract <trajectory> content from exp_summary_prompt.txt.
    This file is generated during the experience generation phase.
    """
    path = os.path.join(sample_dir, "exp_summary_prompt.txt")
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract content between <trajectory> and </trajectory> tags
        match = re.search(r'<trajectory>(.*?)</trajectory>', content, re.DOTALL)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"  Warning: Error extracting trajectory from {path}: {e}")
    
    return None

def generate_skill_for_sample(sample_info: Dict[str, Any], llm: Any, args: Any, ground_truth: str = "") -> Dict[str, Any]:
    """
    Generate a single skill for one sample based on its trajectory and ground truth.
    
    Args:
        sample_info: Dict containing sample metadata and rollout results
        llm: ExperienceLLM instance
        args: Command line arguments
        ground_truth: Ground truth answer for verifying trajectory correctness
        
    Returns:
        Dict with generation result
    """
    question_id = sample_info['question_id']
    sample_dir = sample_info['sample_dir']
    
    result = {
        'sample_id': question_id,
        'success': False,
        'skill_content': '',
        'error': None
    }
    
    trajectory = extract_trajectory_from_file(sample_dir)
    if not trajectory:
        result['error'] = "No trajectory content found in prompt file"
        return result
    
    try:
        # Generate skill using LLM with ground truth for verification
        prompt = GENERATE_SKILL_PROMPT.format(
            trajectory=trajectory,
            ground_truth=ground_truth or "[NOT PROVIDED]"
        )
        skill_content = llm.chat(prompt)
        
        if skill_content:
            # Clean up potential markdown blocks from LLM response
            skill_content = re.sub(r'^```markdown\n', '', skill_content)
            skill_content = re.sub(r'^```\n', '', skill_content)
            skill_content = re.sub(r'\n```$', '', skill_content)
            skill_content = skill_content.strip()
            
            # Save local skill file for the sample
            skill_file = os.path.join(sample_dir, 'skill_raw.md')
            with open(skill_file, 'w', encoding='utf-8') as f:
                f.write(skill_content)
            
            result['success'] = True
            result['skill_content'] = skill_content
        else:
            result['error'] = "LLM returned empty skill content"
    except Exception as e:
        result['error'] = str(e)
        
    return result

def merge_skills(existing_content: str, new_skill_contents: List[str], llm: Any, args: Any) -> str:
    """
    Merge multiple new skills into the existing global skill library using LLM.
    
    Args:
        existing_content: Current content of the global SKILL.md
        new_skill_contents: List of newly generated skill contents from the current batch
        llm: ExperienceLLM instance
        args: Command line arguments
        
    Returns:
        The merged global SKILL.md content
    """
    if not new_skill_contents:
        return existing_content
    
    # Format new skills for the prompt
    new_skills_text = ""
    for i, content in enumerate(new_skill_contents):
        new_skills_text += f"--- New Skill {i+1} ---\n{content}\n\n"
    
    prompt = MERGE_SKILL_PROMPT.format(
        existing_skill=existing_content or "No existing global skill library yet.",
        new_skills=new_skills_text
    )
    
    try:
        merged_content = llm.chat(prompt)
        if merged_content:
            # Clean up potential markdown blocks
            merged_content = re.sub(r'^```markdown\n', '', merged_content)
            merged_content = re.sub(r'^```\n', '', merged_content)
            merged_content = re.sub(r'\n```$', '', merged_content)
            return merged_content.strip()
    except Exception as e:
        print(f"  Warning: Error merging skills: {e}")
        
    return existing_content

def adapt_skill_for_task(base_skill: str, experiences: str, task_description: str, llm: Any, images: Optional[List[Any]] = None) -> str:
    """
    Adapts a base skill document using retrieved experiences for a specific task.
    
    Args:
        base_skill: Content of the global SKILL.md
        experiences: String representation of retrieved and rewritten experiences
        task_description: Description of the current task
        llm: ExperienceLLM instance
        images: Optional list of PIL.Image objects
        
    Returns:
        The adapted SKILL.md content
    """
    if not base_skill:
        return ""
    
    prompt = ADAPT_SKILL_PROMPT.format(
        base_skill=base_skill,
        experiences=experiences or "No specific experiences retrieved.",
        task=task_description
    )
    
    try:
        if images and len(images) > 0:
            adapted_content = llm.chat_with_image(
                prompt=prompt,
                image=images,
                max_tokens=8192,
                temperature=0.3,
                system_prompt="You are an expert at adapting general SOPs to specific task contexts with visual information.",
                return_placeholder_on_error=False
            )
        else:
            adapted_content = llm.chat(prompt, max_tokens=8192, temperature=0.3)
        
        if adapted_content:
            # Clean up potential markdown blocks
            adapted_content = re.sub(r'^```markdown\n', '', adapted_content)
            adapted_content = re.sub(r'^```\n', '', adapted_content)
            adapted_content = re.sub(r'\n```$', '', adapted_content)
            # Remove frontmatter metadata (--- blocks with name/description/version)
            adapted_content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', adapted_content, flags=re.DOTALL | re.MULTILINE)
            return adapted_content.strip()
    except Exception as e:
        print(f"  Warning: Error adapting skill: {e}")
        
    return base_skill


def refine_skill_document(skill_content: str, llm: Any, skill_path: Optional[str] = None, word_threshold: int = 1000, force_refine: bool = False) -> str:
    """
    Perform a refinement pass on the SKILL.md document to consolidate and trim content.
    
    Args:
        skill_content: Current content of SKILL.md
        llm: ExperienceLLM instance
        skill_path: Optional path to save debug info
        word_threshold: Word count threshold to trigger refinement (default: 1000)
        force_refine: If True, perform refine even if word_count < threshold (e.g., for final batch)
        
    Returns:
        Refined SKILL.md content
    """
    if not skill_content:
        return skill_content
    
    word_count = len(skill_content.split())
    
    # Save pre_refine snapshot if skill_path is provided (even if we skip refine)
    if skill_path:
        try:
            debug_path = skill_path.replace('.md', '_pre_refine.md')
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(skill_content)
        except Exception:
            pass
    
    # Skip if already compact (unless forced)
    if word_count < word_threshold and not force_refine:
        print(f"  [Skill Refine] Already compact ({word_count} words), skipping")
        return skill_content
    
    print(f"  [Skill Refine] Starting refinement: {word_count} words")
    
    prompt = SKILL_REFINE_PROMPT.format(
        word_count=word_count,
        skill_content=skill_content
    )
    
    try:
        refined_content = llm.chat(prompt, max_tokens=8192)
        
        if refined_content:
            # Clean up markdown blocks
            refined_content = re.sub(r'^```markdown\n', '', refined_content)
            refined_content = re.sub(r'^```\n', '', refined_content)
            refined_content = re.sub(r'\n```$', '', refined_content)
            refined_content = refined_content.strip()
            
            new_word_count = len(refined_content.split())
            print(f"  [Skill Refine] Complete: {word_count} → {new_word_count} words")
            
            return refined_content
    except Exception as e:
        print(f"  [Skill Refine] Error: {e}")
    
    return skill_content
