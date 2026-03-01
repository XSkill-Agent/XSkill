# -*- coding: utf-8 -*-

GENERATE_SKILL_PROMPT = """You are a skilled AI agent architect. Analyze the trajectory and extract a reusable Standard Operating Procedure (SOP).

### Guiding Principles:
1. **Learn from Success and Failure**: Compare the trajectory with the ground truth.
   - From successful patterns: Extract the effective workflows and tool sequences.
   - From failed attempts or near-misses: Note what went wrong and why - these lessons are often more valuable.
2. **Keep It General**: Use placeholders like `[TARGET]`, `[QUERY]` instead of specific values. The skill should apply to similar problems, not just this one.
3. **Capture Executable Knowledge**: When the trajectory includes effective code, extract the core logic as a reusable template. Good code templates are worth more than paragraphs of description.
4. **Brevity Matters**: Aim for ~600 words. Focus on what's actionable.

### Output Structure:
```
---
name: [SkillName]
description: |
  [Clear, concise description of what this skill does and when to use it. 1-2 sentences focusing on the core purpose and benefits.]
version: 1.0.0
---

# [Skill Title]

## When to Use
- **Task patterns**: [what kinds of questions or requests]
- **Visual signals**: [what the image might contain]

## Strategy Overview
[1-2 sentences on the core approach]

## Workflow
1. **[Phase Name]**: [Action and rationale]
2. **[Phase Name]**: [Action and rationale]
3. ...

## Tool Templates
(Include only if the trajectory contained useful code or query patterns)

- **[Tool] - [Purpose]**:
  ```python
  # [Brief comment]
  [code with placeholders]
  ```

- **Query Pattern**: `"[pattern with placeholders]"`

## Watch Out For
- [Common mistake or trap from the trajectory]
```

### Input:
<trajectory>
{trajectory}
</trajectory>

<ground_truth>
{ground_truth}
</ground_truth>

Output ONLY the SKILL.md content starting with `---`."""


MERGE_SKILL_PROMPT = """You are a knowledge architect. Your job is to maintain a single, unified skill document that grows wiser with each new case.

### Philosophy:
Think of the global skill as a living document. Each new skill brings potential insights - your task is to integrate them thoughtfully, not mechanically.

### Integration Strategy:
For each part of the new skill, ask:
- **Is this part better?** → Rewrite the existing version
- **Is this part redundant or too specific?** → Delete it
- **Is this part complementary?** → Merge into a more general form
- **Is this part genuinely different?** → Add as a variant workflow (but consolidate if possible)

### Quality Guidelines:
- Preserve concrete, reusable code templates and tool patterns
- Delete overly specific examples and cases that don't apply to similar problems
- Consolidate similar trigger phrases
- If workflows differ only in minor details, merge them into one with noted variations

### Length Budget:
- Target: ~1000 words
- If growing too long: merge similar workflows, trim verbose explanations
- Maximum 4 variant workflows - if you have more, they likely can be consolidated

### Input:
<existing_skill>
{existing_skill}
</existing_skill>

<new_skills>
{new_skills}
</new_skills>

Output ONLY the merged SKILL.md starting with `---`. No preamble."""


SKILL_REFINE_PROMPT = """You are a skill document architect. Refine the SKILL.md to remove redundancy, generalize specific cases, and improve structure.

### Current Stats:
- Word count: {word_count}

### Refinement Goals:

1. **Remove Redundancy**:
   - Merge duplicate or near-duplicate content across sections
   - Eliminate repeated explanations that appear in multiple places
   - Consolidate overlapping concepts into single, clearer statements

2. **Avoid Too Specific Cases**:
   - Replace overly specific examples with generalizable patterns
   - Convert hardcoded values to placeholders (e.g., `[TARGET]`, `[QUERY]`)
   - Delete task-specific details or specific cases that don't apply to similar problems

3. **Logical Consolidation**:
   - Merge workflows that share substantial overlap into variants
   - Extract common preliminary steps into dedicated sections
   - Group related tool templates and query patterns together
   - Consolidate similar pitfalls into broader categories

4. **Format Optimization**:
   - Ensure consistent structure and formatting throughout
   - Improve section hierarchy and logical flow
   - Make workflows easier to scan (clear steps, consistent formatting)
   - Organize content from general principles → specific techniques

5. **Content Quality**:
   - Keep description concise and focused on core purpose
   - Ensure all content is actionable and reusable
   - Remove verbose explanations that don't add value
   - Maintain the most essential and distinctive elements

### Principles:
- Prioritize generalizability over specificity
- Keep what enables reuse across similar problems
- Remove what only applies to one particular case
- Maintain clarity and actionability

Output ONLY the refined SKILL.md starting with `---`. No preamble.

<current_skill>
{skill_content}
</current_skill>
"""


