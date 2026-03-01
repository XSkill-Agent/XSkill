"""
LLM-as-Judge evaluator for assessing answer correctness.

This module provides an LLM-based evaluation system that compares predicted
answers against ground truth using configurable prompts.
"""

import re
import os
import time
import openai
from typing import Tuple, List, Optional, Dict

from prompts.llm_as_judge_prompts import SYSTEM_PROMPT, QUERY_PROMPT


# Global client instance (lazily initialized)
_client = None


def _get_client():
    """Get or create the global LLM judge client instance."""
    global _client
    if _client is None:
        _client = LLMJudgeClient()
    return _client


class LLMJudgeClient:
    """Client for LLM-based answer evaluation using OpenAI-compatible APIs."""

    def __init__(self):
        """Initialize the LLM judge client with environment variables."""
        api_key = os.environ.get("VERIFIER_API_KEY")
        endpoint = os.environ.get("VERIFIER_END_POINT")
        self.model_name = os.environ.get("VERIFIER_MODEL_NAME", "gpt-4o-2024-11-20")
        
        if not api_key or not endpoint:
            raise ValueError(
                "VERIFIER_API_KEY and VERIFIER_END_POINT must be set in environment variables"
            )
        
        self.client = openai.OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )

    def evaluate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        max_retries: int = 5
    ) -> Tuple[str, str]:
        """
        Evaluate an answer using the LLM judge.
        
        Args:
            prompt: The evaluation prompt with question, ground truth, and prediction
            system_prompt: Optional system prompt (defaults to None)
            max_retries: Maximum number of retry attempts (default: 5)
            
        Returns:
            Tuple of (score_str, response_text) where:
            - score_str: Extracted score ("1" or "0") 
            - response_text: Full response text with explanation
        """
        messages = []
        
        if system_prompt is not None:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })
        
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        })

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=min(0.2 * attempt, 1.0),
                    max_tokens=8192,
                    timeout=120,
                )

                # Validate response structure
                if not response or not hasattr(response, 'choices') or not response.choices:
                    raise ValueError("Invalid response: missing choices field")
                
                if len(response.choices) == 0:
                    raise ValueError("Invalid response: empty choices array")
                
                first_choice = response.choices[0]
                if not first_choice or not hasattr(first_choice, 'message'):
                    raise ValueError("Invalid response: missing message in choices[0]")
                
                message = first_choice.message
                if not message or not hasattr(message, 'content'):
                    raise ValueError("Invalid response: missing content in message")
                
                response_text = message.content
                if not response_text:
                    raise ValueError("Invalid response: empty content")
                
                # Extract score from response
                if 'score:' not in response_text.lower():
                    raise ValueError(f"No 'score:' found in response")

                score_str = (
                    response_text.lower()
                    .split("score:")[-1]
                    .strip()
                    .split("\n")[0]
                    .strip()
                    .split(" ")[0]
                )
                
                if "1" not in score_str and '0' not in score_str:
                    raise ValueError(f"No valid score ('0' or '1') found: {score_str}")

                return score_str, response_text
                
            except openai.RateLimitError as e:
                print(f"[RateLimitError] Attempt {attempt + 1}/{max_retries}: {str(e)}")
                time.sleep(min(3 * (attempt + 1), 10))
                continue
                
            except (openai.APIError, openai.APIConnectionError, openai.APITimeoutError) as e:
                error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
                print(f"[APIError] Attempt {attempt + 1}/{max_retries}: {str(e)} (code: {error_code})")
                if attempt < max_retries - 1:
                    delay = min(2 ** attempt, 10)
                    time.sleep(delay)
                continue
                
            except Exception as e:
                print("=" * 100)
                print(f"[Error] Attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Messages length: {len(str(messages))}")
                    print("=" * 100)
                    delay = min(2 ** attempt, 10)
                    time.sleep(delay)
                else:
                    print(f"Messages: {messages}")
                    print("=" * 100)
                continue
        
        print(f"Warning: Evaluation failed after {max_retries} attempts")
        return "", ""


def _extract_answer_from_text(
    text: str, 
    predict_str_list: List[str], 
    extraction_mode: str
) -> Tuple[float, str]:
    """
    Extract answer from prediction text using specified extraction mode.
    
    Args:
        text: Full prediction text
        predict_str_list: List of prediction strings (for fallback)
        extraction_mode: Extraction mode ('split', 'strict', or 'strict_v2')
        
    Returns:
        Tuple of (reward, extracted_answer) where reward is 0.0 on failure
    """
    if extraction_mode == 'split':
        return 0.0, text.split("<answer>")[-1].split("</answer>")[0].strip()
    
    elif extraction_mode in ['strict', 'strict_v2']:
        # Extract the LAST <answer> tag (handles multiple answers from revisions)
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            # Use the last match (most recent answer)
            return 0.0, matches[-1].strip()
        
        # Fallback: use the last turn's output if no answer tags found
        if predict_str_list:
            last_turn = predict_str_list[-1].strip()
            
            # Skip error messages
            if last_turn.startswith("Error:") or "API attempts failed" in last_turn:
                return 0.0, ""
            
            return 0.0, last_turn
        
        return 0.0, ""
    
    else:
        raise ValueError(f"Unknown extraction mode: {extraction_mode}")


def evaluate_answer(
    question: str,
    prediction_list: List[str],
    ground_truth: str,
    extract_answer: bool = False,
    extra_info: Optional[Dict] = None
) -> Tuple[float, str]:
    """
    Evaluate a predicted answer against ground truth using LLM-as-judge.
    
    Args:
        question: The question text
        prediction_list: List of prediction text segments
        ground_truth: Ground truth answer
        extract_answer: Whether to extract answer from <answer> tags
        extra_info: Optional dict with extraction settings
        
    Returns:
        Tuple of (reward, analysis) where:
        - reward: 1.0 for correct, 0.0 for incorrect
        - analysis: Explanation text from the judge
    """
    prediction_text = ' '.join(prediction_list)
    
    # Extract answer if requested
    if extract_answer and extra_info:
        extraction_mode = extra_info.get('extract_answer_tags', 'strict')
        reward, prediction_text = _extract_answer_from_text(
            prediction_text, 
            prediction_list, 
            extraction_mode
        )
        
        # Return early if extraction failed
        if reward == 0.0 and not prediction_text:
            return 0.0, "Answer extraction failed"
    
    # Format evaluation prompt
    eval_prompt = QUERY_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction_text
    )
    
    # Log very long prompts
    if len(eval_prompt) > 10000:
        print(f"[evaluate_answer] Warning: Very long prompt ({len(eval_prompt)} chars)")
    
    # Query LLM judge
    try:
        client = _get_client()
        score_str, response_text = client.evaluate(
            prompt=eval_prompt, 
            system_prompt=SYSTEM_PROMPT
        )
    except Exception as e:
        print(f"[evaluate_answer] Error calling LLM judge: {e}")
        return 0.0, f"Evaluation API error: {str(e)}"
    
    if not score_str:
        return 0.0, "Evaluation failed: Empty response from API"
    
    try:
        reward = 1.0 if '1' in score_str else 0.0
        
        # Extract explanation
        analysis = ""
        if response_text and "explanation:" in response_text.lower():
            explanation_part = response_text.lower().split("explanation:")[-1].strip()
            if explanation_part:
                analysis = explanation_part
        elif not response_text:
            analysis = "No explanation provided in response"
        
        return reward, analysis
        
    except Exception as e:
        print(f"[evaluate_answer] Error processing response: {e}")
        return 0.0, f"Error processing evaluation response: {str(e)}"


def compute_score(
    prompt: str, 
    predict_str_list: List[str], 
    ground_truth: str, 
    extra_info: Optional[Dict] = None
) -> Tuple[float, str]:
    """
    Compute accuracy score and return analysis text.
    
    Args:
        prompt: Question text
        predict_str_list: List of prediction strings
        ground_truth: Ground truth answer
        extra_info: Optional dict with evaluation settings:
            - gpt_extract_answer: bool, whether to extract from tags
            - extract_answer_tags: str, extraction mode
            - acc_reward_weight: float, weight for accuracy score
    
    Returns:
        Tuple of (accuracy_score, analysis) where:
        - accuracy_score: float, weighted accuracy (0.0 or weight*1.0)
        - analysis: str, evaluation explanation
    """
    extra = extra_info or {}
    extract_answer = extra.get("gpt_extract_answer", False)
    acc_weight = extra.get('acc_reward_weight', 1.0)
    
    try:
        reward, analysis = evaluate_answer(
            question=prompt,
            prediction_list=predict_str_list,
            ground_truth=ground_truth,
            extract_answer=extract_answer,
            extra_info=extra
        )
        
        # Validate reward type
        if not isinstance(reward, (int, float)):
            print(f"[compute_score] Warning: Invalid reward type: {type(reward)}, defaulting to 0.0")
            reward = 0.0
            if not analysis:
                analysis = f"Invalid reward value: {reward}"
        
        accuracy_score = acc_weight * float(reward)
        return accuracy_score, analysis
        
    except Exception as e:
        print(f"[compute_score] Error: {e}")
        return 0.0, f"Evaluation error: {str(e)}"


if __name__ == '__main__':
    # Example usage
    question = "What is the name of the store with a blue sign?"
    prediction = [
        "<think>Looking at the image, I can see a blue sign on the right.</think> "
        "<answer>The name is J&optica.</answer>"
    ]
    ground_truth = "Jptica"
    
    extra_info = {
        "acc_reward_weight": 1.0,
        "gpt_extract_answer": True,
        "extract_answer_tags": "strict",
    }
    
    accuracy, analysis = compute_score(question, prediction, ground_truth, extra_info)
    print(f"Accuracy Score: {accuracy}")
    print(f"Analysis: {analysis}")
