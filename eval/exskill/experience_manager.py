import os
import json
import re
import time
from typing import Optional, Dict, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


from .llm_client import ExperienceLLM
from .experience_retriever import ExperienceRetriever

from prompts.experience_prompts import EXPERIENCE_REFINE_PROMPT, MERGE_PROMPT


# --------- Constants ---------

# Retry configuration
MAX_RETRIES = 3

# Similarity thresholds
SIMILARITY_THRESHOLD = 0.70  # Default threshold for triggering merge
SIMILARITY_THRESHOLD_REDUCTION = 0.60  # Lower threshold for reduction (more aggressive)

# Similarity search configuration
TOP_K_SIMILARITY_SEARCH = 10  # Number of similar experiences to retrieve


# --------- Helpers ---------


def _strip_markdown_code_blocks(text: str) -> str:
    """Remove markdown code fence wrappers from LLM response text."""
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    if len(parts) > 2:
        return parts[2].strip()
    return (parts[0].strip() if parts else text).strip()


def _max_exp_id(exp_dict: Dict[str, str], default: int = 0) -> int:
    """Return the maximum numeric part of experience IDs (E0, E1, ...)."""
    nums = []
    for k in exp_dict:
        if k.startswith("E") and len(k) > 1 and k[1:].isdigit():
            nums.append(int(k[1:]))
    return max(nums, default=default)


# --------- Batch update / merge ---------

class ExperienceMemoryProvider:
    """
    Manages experiences with embedding-based similarity search and merge operations.
    Wraps ExperienceRetriever for embedding functionality.
    """
    
    def __init__(self, experiences: Dict[str, str], retriever: Optional[ExperienceRetriever] = None):
        """
        Initialize the provider with existing experiences.
        
        Args:
            experiences: Dictionary mapping experience IDs to experience text
            retriever: Optional ExperienceRetriever instance (will create one if not provided)
        """
        self.experiences = dict(experiences)
        self._retriever = retriever
        
        # Build embedding index if retriever is available
        if self._retriever is None:
            # Create a minimal retriever just for embedding functionality
            # We'll reuse the existing retriever's embedding methods
            pass
    
    def _get_retriever(self) -> ExperienceRetriever:
        """Get or create retriever instance."""
        if self._retriever is None:
            # Create a retriever with current experiences
            self._retriever = ExperienceRetriever(
                experiences=self.experiences,
                embedding_model=os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "text-embedding-3-small"),
                embedding_api_key=os.environ.get("EXPERIENCE_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                embedding_endpoint=os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT") or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                enable_cache=False,  # Don't cache for temporary provider
            )
        return self._retriever
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts using retriever."""
        if not texts:
            return np.array([], dtype=np.float32)
        
        retriever = self._get_retriever()
        embeddings = retriever._generate_embeddings_batch(texts, batch_size=len(texts))
        # Filter out None values and convert to numpy array
        valid_embeddings = [e for e in embeddings if e is not None]
        if not valid_embeddings:
            return np.array([], dtype=np.float32)
        if len(valid_embeddings) == 1:
            return valid_embeddings[0].reshape(1, -1)
        return np.vstack(valid_embeddings)
    
    def _search_similar(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = TOP_K_SIMILARITY_SEARCH,
        similarity_threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Tuple[str, str, float]]:
        """
        Search for similar experiences.
        
        Args:
            query_embedding: Query embedding vector (1D or 2D array)
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (exp_id, exp_text, similarity_score) tuples
        """
        if not self.experiences:
            return []
        
        # Ensure query_embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Generate embeddings for all existing experiences
        exp_ids = list(self.experiences.keys())
        exp_texts = [self.experiences[exp_id] for exp_id in exp_ids]
        
        # Batch generate embeddings
        exp_embeddings = self._embed_texts(exp_texts)
        if exp_embeddings.size == 0:
            return []
        
        # Ensure exp_embeddings is 2D
        if len(exp_embeddings.shape) == 1:
            exp_embeddings = exp_embeddings.reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, exp_embeddings)[0]
        
        # Get top-k similar experiences
        top_indices = np.argsort(-similarities)[:top_k]
        
        candidates = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= similarity_threshold:
                exp_id = exp_ids[idx]
                exp_text = exp_texts[idx]
                candidates.append((exp_id, exp_text, score))
        
        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates
    
    def add_experience(self, exp_id: str, text: str):
        """Add a new experience."""
        self.experiences[exp_id] = text
    
    def remove_experience(self, exp_id: str):
        """Remove an experience."""
        if exp_id in self.experiences:
            del self.experiences[exp_id]
    
    def modify_experience(self, exp_id: str, new_text: str):
        """Modify an existing experience."""
        if exp_id in self.experiences:
            self.experiences[exp_id] = new_text
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to simple dictionary format."""
        return dict(self.experiences)


def _llm_merge_experiences(
    candidates: List[Tuple[str, str, float]], 
    llm: ExperienceLLM,
    max_retries: int = MAX_RETRIES
) -> Optional[str]:
    """
    Use LLM to merge similar experiences directly.
    
    Args:
        candidates: List of (exp_id, exp_text, similarity_score) tuples
        llm: ExperienceLLM instance
        max_retries: Maximum retry attempts
        
    Returns:
        Merged experience text, or None if merge failed
    """
    if len(candidates) < 2:
        return None
    
    # Format all experiences to merge
    exp_list = []
    for i, (exp_id, exp_text, score) in enumerate(candidates, 1):
        exp_list.append(f"[{exp_id}] (similarity: {score:.3f})\n{exp_text}")
    
    experiences_text = "\n\n".join(exp_list)
    
    prompt = MERGE_PROMPT.format(experiences_text=experiences_text)
    
    for attempt in range(max_retries):
        try:
            response = _strip_markdown_code_blocks(llm.chat(prompt).strip())
            if response:
                return response
            else:
                print(f"  - Warning: Empty merge response, retrying...")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  - Warning: Merge LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
            else:
                print(f"  - Error: Merge LLM call failed after {max_retries} attempts: {e}")
                return None
    
    return None


def _process_add_with_merge(
    provider: ExperienceMemoryProvider,
    new_text: str,
    llm: ExperienceLLM,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    next_id: int = 0
) -> Tuple[str, bool]:
    """
    Process add operation with similarity check and merge if needed.
    
    Args:
        provider: ExperienceMemoryProvider instance
        new_text: New experience text to add
        llm: ExperienceLLM instance
        similarity_threshold: Threshold for triggering merge
        next_id: Next available ID number
        
    Returns:
        Tuple of (final_exp_id, was_merged)
    """
    # Generate embedding for new text
    new_emb = provider._embed_texts([new_text])
    if new_emb.size == 0:
        # Embedding failed, just add it
        exp_id = f"E{next_id}"
        provider.add_experience(exp_id, new_text)
        return exp_id, False
    
    # Extract single embedding vector (first row if 2D)
    if len(new_emb.shape) == 2:
        query_emb = new_emb[0]
    else:
        query_emb = new_emb
    
    # Search for similar experiences
    candidates = provider._search_similar(query_emb, top_k=TOP_K_SIMILARITY_SEARCH, similarity_threshold=similarity_threshold)
    
    if not candidates:
        # No similar experiences, just add
        exp_id = f"E{next_id}"
        provider.add_experience(exp_id, new_text)
        return exp_id, False
    
    # Found similar experiences, directly merge
    print(f"  - Found {len(candidates)} similar experiences (threshold: {similarity_threshold:.2f}), merging...")
    
    # Include new experience in candidates
    all_candidates = [("NEW", new_text, 1.0)] + candidates
    
    # Directly merge using LLM
    merged_text = _llm_merge_experiences(all_candidates, llm)
    
    if merged_text:
        # Remove existing experiences that were merged
        for exp_id, _, _ in candidates:
            provider.remove_experience(exp_id)
        
        # Add merged experience
        exp_id = f"E{next_id}"
        provider.add_experience(exp_id, merged_text)
        print(f"    - Merged {len(all_candidates)} experiences into {exp_id}")
        return exp_id, True
    else:
        # Merge failed, fallback: add new experience
        print(f"    - Warning: Merge failed, adding new experience separately")
        exp_id = f"E{next_id}"
        provider.add_experience(exp_id, new_text)
        return exp_id, False


def batch_merge(
    existing: Dict[str, str], 
    updates: List[dict], 
    llm: ExperienceLLM, 
    max_retries: int = MAX_RETRIES, 
    debug_dir: Optional[str] = None, 
    experience_limit: Optional[int] = None,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    retriever: Optional[ExperienceRetriever] = None
) -> Dict[str, str]:
    """
    Merge new experience updates into existing library with intelligent similarity-based merging.
    
    Flow:
    - add operations: Check similarity → if high, directly merge with similar experiences
    - modify operations: Directly modify, no similarity check
    
    Args:
        existing: Current experience library {id: text}
        updates: List of update operations from intra_sample_experiences
        llm: ExperienceLLM instance for calling LLM
        max_retries: Number of retry attempts for LLM call
        debug_dir: Optional directory to save debug information
        experience_limit: Optional limit on experience count
        similarity_threshold: Threshold for triggering merge
        retriever: Optional ExperienceRetriever instance for embedding (will create if needed)
    
    Returns:
        Updated experience library
    """
    if not updates:
        # When there are no new updates, if the library exceeds the limit, perform a compression merge
        if isinstance(experience_limit, int) and experience_limit > 0 and len(existing) > experience_limit:
            try:
                capped = _reduce_experiences_to_limit(existing, experience_limit, llm, debug_dir, similarity_threshold=similarity_threshold, retriever=retriever)
                print(f"  - Library reduced to limit: {len(capped)} (<= {experience_limit})")
                return capped
            except Exception as e:
                print(f"  - Warning: failed to reduce existing library to limit {experience_limit}: {e}")
        return existing
    
    # Step 1: Initialize provider
    provider = ExperienceMemoryProvider(existing, retriever=retriever)
    max_id = _max_exp_id(existing)

    # Step 2: Process add operations with similarity check and merge
    to_modify = []
    add_count = 0
    merge_count = 0
    
    for operation in updates:
        option = operation.get("option", "add")
        experience = operation.get("experience", "").strip()
        
        if not experience:
            continue
        
        if option == "modify":
            # Modify: directly modify, no similarity check
            modified_from = operation.get("modified_from")
            if modified_from and modified_from in provider.experiences:
                to_modify.append(operation)
            else:
                # If referenced ID doesn't exist, treat as add
                max_id += 1
                exp_id, was_merged = _process_add_with_merge(provider, experience, llm, similarity_threshold, max_id)
                add_count += 1
                if was_merged:
                    merge_count += 1
        elif option == "add":
            # Add: check similarity and merge if needed
            max_id += 1
            exp_id, was_merged = _process_add_with_merge(provider, experience, llm, similarity_threshold, max_id)
            add_count += 1
            if was_merged:
                merge_count += 1
    
    print(f"  - Batch update: {len(updates)} updates → {len(to_modify)} modifies, {add_count} adds ({merge_count} merged)")
    
    # Step 3: Apply modify operations
    for operation in to_modify:
        modified_from = operation.get("modified_from")
        new_text = operation.get("experience", "").strip()
        if modified_from and new_text:
            provider.modify_experience(modified_from, new_text)
            print(f"    - Modified {modified_from}")
    
    # Step 4: Get final experiences and renumber
    final_experiences_dict = provider.to_dict()
    
    # Renumber experiences to have clean sequential IDs (E0, E1, E2, ...)
    final_experiences = {}
    for idx, (old_id, text) in enumerate(sorted(final_experiences_dict.items())):
        final_experiences[f"E{idx}"] = text
    
    print(f"  - Final library size: {len(final_experiences)} experiences")
    
    # Optional: enforce experience_limit by iterative LLM-guided merges
    if isinstance(experience_limit, int) and experience_limit > 0 and len(final_experiences) > experience_limit:
        print(f"  - Current library size ({len(final_experiences)}) exceeds limit ({experience_limit}), attempting to merge...")
        try:
            reduced_experiences = _reduce_experiences_to_limit(final_experiences, experience_limit, llm, debug_dir, similarity_threshold=similarity_threshold, retriever=retriever)
            if len(reduced_experiences) < len(final_experiences):
                print(f"  - Library size after reduction: {len(reduced_experiences)} (limit {experience_limit})")
                final_experiences = reduced_experiences
            else:
                print(f"  - Warning: reduction failed, library size unchanged: {len(final_experiences)} (limit {experience_limit})")
        except Exception as e:
            print(f"  - Warning: failed to reduce experiences to limit {experience_limit}: {e}")
    
    return final_experiences


def _reduce_experiences_to_limit(
    experiences: Dict[str, str], 
    limit: int, 
    llm: ExperienceLLM, 
    debug_dir: Optional[str] = None, 
    max_rounds: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD_REDUCTION,
    retriever: Optional[ExperienceRetriever] = None
) -> Dict[str, str]:
    """
    Iteratively reduce experience count to <= limit using similarity-based forced merging.
    
    Process:
    1. Find most similar experience pairs using embedding similarity
    2. Force merge pairs starting from highest similarity
    3. Repeat until limit is reached or max_rounds exceeded
    """
    current = dict(experiences)
    round_idx = 0
    print(f"    - Starting reduction: {len(current)} experiences, target <= {limit}")
    
    # Initialize provider for similarity search
    provider = ExperienceMemoryProvider(current, retriever=retriever)
    
    while len(current) > limit and round_idx < max_rounds:
        round_idx += 1
        print(f"    - Round {round_idx}: Current count: {len(current)}, target: {limit}")
        
        try:
            # Step 1: Find most similar pairs using embedding similarity
            exp_ids = list(current.keys())
            if len(exp_ids) < 2:
                break
            
            # Generate embeddings for all experiences (batch)
            exp_texts = [current[exp_id] for exp_id in exp_ids]
            exp_embeddings = provider._embed_texts(exp_texts)
            
            if exp_embeddings.size == 0:
                print(f"    - Round {round_idx}: Failed to generate embeddings, stopping")
                break
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(exp_embeddings)
            
            # Find top similar pairs (excluding self-similarity)
            similar_pairs = []
            for i in range(len(exp_ids)):
                for j in range(i + 1, len(exp_ids)):
                    sim_score = float(similarities[i][j])
                    if sim_score >= similarity_threshold:
                        similar_pairs.append((exp_ids[i], exp_ids[j], sim_score))
            
            # Sort by similarity (descending)
            similar_pairs.sort(key=lambda x: x[2], reverse=True)
            
            if not similar_pairs:
                print(f"    - Round {round_idx}: No similar pairs found (threshold: {similarity_threshold:.2f}), stopping")
                break
            
            # Step 2: Force merge pairs starting from highest similarity
            merged_any = False
            pairs_to_merge = min(len(similar_pairs), len(current) - limit)
            
            for pair_idx, (exp_id1, exp_id2, sim_score) in enumerate(similar_pairs[:pairs_to_merge]):
                if len(current) <= limit:
                    break
                
                # Check if both experiences still exist
                if exp_id1 not in current or exp_id2 not in current:
                    continue
                
                # Force merge using LLM
                experiences_text = f"[{exp_id1}]\n{current[exp_id1]}\n\n[{exp_id2}]\n{current[exp_id2]}"
                prompt = MERGE_PROMPT.format(experiences_text=experiences_text)
                
                try:
                    merged_text = _strip_markdown_code_blocks(llm.chat(prompt).strip())
                    if merged_text:
                        # Remove source experiences
                        provider.remove_experience(exp_id1)
                        provider.remove_experience(exp_id2)
                        
                        # Add merged experience
                        new_id = f"E{_max_exp_id(current) + 1}"
                        provider.add_experience(new_id, merged_text)
                        
                        # Update current dict
                        del current[exp_id1]
                        del current[exp_id2]
                        current[new_id] = merged_text
                        
                        merged_any = True
                        print(f"      - Force merged {exp_id1} and {exp_id2} (similarity: {sim_score:.3f}) into {new_id}")
                    else:
                        print(f"      - Warning: Failed to generate merged text for {exp_id1} and {exp_id2}")
                except Exception as e:
                    print(f"      - Warning: Failed to merge {exp_id1} and {exp_id2}: {e}")
                    continue
            
            if not merged_any:
                print(f"    - Round {round_idx}: No merges applied, stopping")
                break
            
            print(f"    - Round {round_idx}: After merging, {len(current)} experiences remain")
            
            # Update provider with current state
            provider = ExperienceMemoryProvider(current, retriever=retriever)
            
        except Exception as e:
            print(f"    - Round {round_idx}: Error during reduction: {e}")
            import traceback
            traceback.print_exc()
            break
    
    return current


def refine_experience_library(
    experiences: Dict[str, str],
    llm: ExperienceLLM,
    debug_dir: Optional[str] = None,
    max_retries: int = MAX_RETRIES
) -> Dict[str, str]:
    """
    Perform a global refinement pass on the entire experience library.
    
    This is called after batch_merge to further consolidate redundant experiences
    and improve overall library quality.
    
    Args:
        experiences: Current experience library {id: text}
        llm: ExperienceLLM instance
        debug_dir: Optional directory to save debug information
        max_retries: Number of retry attempts
        
    Returns:
        Refined experience library
    """
    if not experiences or len(experiences) < 10:
        # Skip refinement for small libraries
        return experiences
    
    print(f"  [Refine] Starting library refinement: {len(experiences)} experiences")
    
    # Format experiences for the prompt
    exp_lines = []
    for exp_id in sorted(experiences.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        exp_lines.append(f"[{exp_id}] {experiences[exp_id]}")
    experiences_text = "\n\n".join(exp_lines)
    
    prompt = EXPERIENCE_REFINE_PROMPT.format(
        exp_count=len(experiences),
        experiences=experiences_text
    )
    
    # Save debug prompt
    if debug_dir:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, "exp_refine_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
        except Exception:
            pass
    
    # Call LLM
    response = None
    for attempt in range(max_retries):
        try:
            response = llm.chat(prompt, max_tokens=16384)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [Refine] LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"  [Refine] LLM call failed after {max_retries} attempts: {e}")
                return experiences
    
    if not response:
        return experiences
    
    # Save debug response
    if debug_dir:
        try:
            with open(os.path.join(debug_dir, "exp_refine_resp.txt"), "w", encoding="utf-8") as f:
                f.write(response)
        except Exception:
            pass
    
    # Parse operations from response
    try:
        if "```json" in response:
            payload = response.split("```json")[-1].split("```")[0].strip()
        else:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            payload = json_match.group(0) if json_match else "[]"
        
        operations = json.loads(payload)
        if not isinstance(operations, list):
            print(f"  [Refine] Invalid response format, skipping")
            return experiences
    except Exception as e:
        print(f"  [Refine] Failed to parse operations: {e}")
        return experiences
    
    # Apply operations
    refined = dict(experiences)
    merge_count = 0
    delete_count = 0
    
    for op in operations:
        option = op.get("option", "")
        
        if option == "merge":
            merged_from = op.get("merged_from", [])
            merged_text = op.get("experience", "").strip()
            
            if merged_from and len(merged_from) >= 2 and merged_text:
                # Check if all source IDs exist
                valid_sources = [eid for eid in merged_from if eid in refined]
                if len(valid_sources) >= 2:
                    # Remove source experiences
                    for eid in valid_sources:
                        del refined[eid]
                    # Add merged experience with new ID
                    new_id = f"E{_max_exp_id(refined, default=-1) + 1}"
                    refined[new_id] = merged_text
                    merge_count += 1
                    print(f"    - Merged {valid_sources} → {new_id}")
        
        elif option == "delete":
            deleted_id = op.get("deleted_id", "")
            if deleted_id and deleted_id in refined:
                del refined[deleted_id]
                delete_count += 1
                print(f"    - Deleted {deleted_id}")
    
    # Renumber experiences
    final = {}
    for idx, (_, text) in enumerate(sorted(refined.items(), key=lambda x: int(x[0][1:]) if x[0][1:].isdigit() else 0)):
        final[f"E{idx}"] = text
    
    print(f"  [Refine] Complete: {len(experiences)} → {len(final)} ({merge_count} merges, {delete_count} deletes)")
    
    return final