ADAPT_SKILL_PROMPT = """You are an expert agent assistant. Tailor the general skill to this specific task.

### Your Goals:
1. **Select What's Relevant**: Look at the task and images - which parts of the skill actually apply here? Remove everything else.
2. **Integrate Experiences**: If experiences are provided, weave their insights into the relevant workflow steps. They often contain practical tips that complement the skill's structure.
3. **Keep Templates Ready**: Preserve any code templates or query patterns that might be useful, but you can adjust placeholders to be more task-relevant.
4. **Stay Lean**: The adapted skill should be a focused guide, not a comprehensive manual. ~400 words max.

### Input:
<base_skill>
{base_skill}
</base_skill>

<experiences>
{experiences}
</experiences>

<task>
{task}
</task>

**CRITICAL**: Output a reusable methodology guide, NOT a pre-filled answer. Use placeholders (e.g., "the observed value", "extracted number") instead of actual data from images or task.

Output ONLY the adapted skill content (markdown format starting with `#`). Do NOT include frontmatter metadata (no `---` blocks with name/description/version). Focus on what will actually help solve this task."""


SKILL_INJECTION_HEADER = """Here are practical experiences and skills for tool-based visual reasoning:

<skill>
{skill_content}
</skill>

You can use it as reference if it is relevant to help you solve the problem. You can also have your own ideas or other approaches.

Your instruction is following:
"""
