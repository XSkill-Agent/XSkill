TASK_DECOMPOSITION_PROMPT = """You are an Expert Visual Reasoning Strategist. Your objective is to deconstruct a complex visual task into 2-3 distinct, actionable subtasks to retrieve the most relevant methodological guidance from the experience library.

Task: {task_description}

The experience library contains abstract methodological guidelines for visual reasoning tasks.

For each subtask, generate a JSON object with:
- Type: A concise category describing the methodological phase.
- Query: A methodology-focused search query designed to retrieve "how-to" guides or best practices, based on current task description and image details. 
    - The query should target one of the following aspects:
        1.  Tool Utilization: Best practices for using specific tools (e.g., Code Interpreter, Web Search, Image Search, Zoom) effectively.
        2.  Reasoning Strategy: Reasoning frameworks or potential solution paths that can be used to solve the task.
        3.  Challenge Mitigation: Techniques to handle anticipated challenges (e.g., "handling flipped images," "small objects in the image").
    - CRITICAL: The query must abstract away from specific image details (like "red apples") and focus on the underlying *technical challenge* (like "color-based object filtering").

Required output format:

[
  {{"type": "visual_extraction", "query": "Techniques for robustly analyzing dark images using Code Interpreter"}},
  {{"type": "logic_synthesis", "query": "Frameworks for mapping visual features to knowledge using Web Search"}}
]

Output ONLY the complete and valid JSON array, no additional text."""


EXPERIENCE_REWRITE_PROMPT = """You are an expert AI mentor adapting retrieved methodological experiences to strictly fit a specific visual reasoning task.

Task: {task_description}

Retrieved experiences:
{experiences_text}

Rewrite these experiences to provide cohesive, unified operational guidance applicable to the task and its images. 
For each experience, strictly adhere to the following rewriting guidelines:

1. **Operational Focus**: Transform abstract descriptions into specific, actionable execution tips (e.g., tool using actions, error handling, etc.). Focus on detailed operations and practical techniques rather than high-level summaries.
2. **Pitfalls & Best Practices**: Explicitly integrate common pitfalls to avoid and best practices to follow based on the experience.
3. **Contextual Adaptation**: Keep the core methodological insights but adapt the language to be directly relevant to the current visual task context. Do not make it *too* specific (overfitting), but ensure it is practically useful.
4. **Tone**: Use clear, constructive, and suggestive language (e.g., "Consider checking...", "It is effective to...") rather than direct commands.

If an experience is irrelevant or redundant, you may delete it by not outputting it in the JSON object. Only focus on the experiences that can contribute to solving the task.

Output ONLY a valid JSON object mapping experience IDs to the rewritten guidance text. Strictly follow the format:

{{
  "id1": "rewritten experience 1",
  "id2": "rewritten experience 2",
  ...
}}
"""


INJECTION_HEADER = """Here are practical tips for tool-based visual reasoning, gathered from similar problems:

{bullets}

These experiences highlight common patterns and pitfalls. When you encounter matching situations, consider applying the suggested approaches. You can reference them by ID (e.g., [E12]) in your reasoning.

Your instruction is following:
"""
