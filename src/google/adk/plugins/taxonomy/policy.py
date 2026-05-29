# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract interfaces for taxonomy resolution and skill policy enforcement.

This module defines the pluggable contracts that developers implement:

- ``TaxonomyResolver``: Classifies the active security/regulatory domains
  from runtime context and LLM conversation history.
- ``TaxonomyPipeline``: Chains multiple resolvers into a multi-step pipeline.
- ``SkillPolicy``: Gates skill access and shapes instructions dynamically.
- ``DefaultSkillPolicy``: Reference implementation using taxonomy-bind matching.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from ...agents.readonly_context import ReadonlyContext
from ...models.llm_request import LlmRequest
from ...skills.models import Skill


class TaxonomyResolver(ABC):
  """Abstract base class for taxonomy resolution.

  Resolvers can be chained to form multi-step pipelines via ``TaxonomyPipeline``.

  Example use cases:
      - Semantic classification: Analyze past agent interactions to classify
        the active security domain (e.g. ``urn:adk:domain:compliance``).
      - Entitlements verification: Gate access using feature flags.
      - DB-backed RBAC: Query database records for user permissions.
  """

  @abstractmethod
  async def resolve_taxonomies(
      self, context: ReadonlyContext, llm_request: LlmRequest
  ) -> list[str]:
    """Resolves active taxonomy domain URIs from the runtime context and LLM history.

    Args:
        context: The session runtime context. Provides access to
            ``user_content``, ``user_id``, ``state``, and ``session``.
        llm_request: Outgoing LLM request containing conversation history,
            agent-to-agent dialogues, and reasoning blocks.

    Returns:
        List of active taxonomy domain URI strings
        (e.g. ``["urn:adk:domain:compliance", "urn:adk:domain:medical"]``).
    """
    pass


class TaxonomyPipeline(TaxonomyResolver):
  """Executes a sequence of taxonomy resolvers in order (multi-step pipeline)."""

  def __init__(self, resolvers: list[TaxonomyResolver]):
    self.resolvers = resolvers

  async def resolve_taxonomies(
      self, context: ReadonlyContext, llm_request: LlmRequest
  ) -> list[str]:
    active_domains: set[str] = set()
    for resolver in self.resolvers:
      domains = await resolver.resolve_taxonomies(context, llm_request)
      if domains:
        active_domains.update(domains)
    return list(active_domains)


class SkillPolicy(ABC):
  """Abstract policy engine determining skill execution permissions and instruction shaping."""

  @abstractmethod
  def is_skill_allowed(
      self,
      skill: Skill,
      context: ReadonlyContext,
      active_taxonomies: list[str],
  ) -> bool:
    """Determines if a skill can be loaded/used under the active taxonomies and context."""
    pass

  @abstractmethod
  def shape_instructions(
      self,
      skill: Skill,
      context: ReadonlyContext,
      original_instructions: str,
  ) -> str:
    """Applies dynamic instruction shaping/guardrails to a skill's instructions.

    Called after a skill is loaded but before instructions are returned to the model.
    Use this to append compliance disclaimers, restrict tool usage, inject
    role-specific constraints, etc.

    Args:
        skill: The skill being loaded.
        context: The session runtime context.
        original_instructions: The original instruction text from SKILL.md.

    Returns:
        The shaped/modified instruction text.
    """
    pass


class DefaultSkillPolicy(SkillPolicy):
  """Default skill policy using taxonomy-bind set-intersection matching.

  A skill is allowed if:
  - It has no ``taxonomy-binds`` in its frontmatter (unrestricted), OR
  - At least one of its ``taxonomy-binds`` matches an active taxonomy domain.

  Instructions are returned unmodified. Subclass and override
  ``shape_instructions`` to add custom guardrails.
  """

  def is_skill_allowed(
      self,
      skill: Skill,
      context: ReadonlyContext,
      active_taxonomies: list[str],
  ) -> bool:
    binds = skill.frontmatter.taxonomy_binds
    if not binds:
      return True
    # At least one bind must match an active taxonomy
    return bool(set(binds) & set(active_taxonomies))

  def shape_instructions(
      self,
      skill: Skill,
      context: ReadonlyContext,
      original_instructions: str,
  ) -> str:
    return original_instructions
