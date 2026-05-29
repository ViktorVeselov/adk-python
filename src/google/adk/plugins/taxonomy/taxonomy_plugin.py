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

"""TaxonomyPlugin — ADK BasePlugin for pluggable taxonomy policy enforcement.

This plugin intercepts skill discovery and execution tools to enforce
taxonomy-based access control and dynamic instruction shaping.

Usage::

    from google.adk import TaxonomyPlugin, TaxonomyRegistry

    registry = TaxonomyRegistry.from_flat_json(my_taxonomy_data)
    plugin = TaxonomyPlugin(
        taxonomy_registry=registry,
        resolver=my_resolver,
        policy=my_policy,
    )
    runner = Runner(..., plugins=[plugin])
"""

from __future__ import annotations

import logging
from typing import Any
from typing import Optional

from ..base_plugin import BasePlugin
from ...agents.callback_context import CallbackContext
from ...artifacts.file_artifact_service import _validate_path_segment
from ...errors.input_validation_error import InputValidationError
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...skills import prompt
from ...tools.base_tool import BaseTool
from ...tools.tool_context import ToolContext

from .policy import SkillPolicy
from .policy import TaxonomyResolver
from .taxonomy_config import TaxonomyRegistry

logger = logging.getLogger("google_adk." + __name__)

# Session state key where resolved taxonomies are stored between callbacks.
_ACTIVE_TAXONOMIES_STATE_KEY = "_active_taxonomies"

# Tool names that belong to the skill toolset.
_SKILL_GATE_TOOLS = frozenset({
    "list_skills",
    "load_skill",
    "load_skill_resource",
    "run_skill_script",
})


class TaxonomyPlugin(BasePlugin):
  """Native ADK Plugin enforcing pluggable taxonomy policies.

  This plugin provides:
  - **Skill discovery gating**: Filters ``list_skills`` output to only show
    skills permitted under the active taxonomy domains.
  - **Skill execution gating**: Blocks ``load_skill``, ``load_skill_resource``,
    and ``run_skill_script`` for unauthorized skills.
  - **Path traversal guards**: Reuses the SDK's ``_validate_path_segment``
    and ``InputValidationError`` for zero-duplicate security logic.
  - **Dynamic instruction shaping**: Applies ``SkillPolicy.shape_instructions``
    to ``load_skill`` results via ``after_tool_callback`` (avoids short-circuiting
    the plugin chain).

  Args:
      name: Plugin instance name. Defaults to ``"taxonomy_plugin"``.
      taxonomy_registry: Optional parsed taxonomy definitions for developer use.
      resolver: Optional taxonomy resolver (or pipeline) that classifies
          active domains from runtime context.
      policy: Optional skill policy engine that gates access and shapes instructions.
  """

  def __init__(
      self,
      name: str = "taxonomy_plugin",
      *,
      taxonomy_registry: Optional[TaxonomyRegistry] = None,
      resolver: Optional[TaxonomyResolver] = None,
      policy: Optional[SkillPolicy] = None,
  ):
    super().__init__(name)
    self.taxonomy_registry = taxonomy_registry or TaxonomyRegistry()
    self.resolver = resolver
    self.policy = policy

  # ──────────────────────────────────────────────────────────────────
  # 1. Taxonomy Resolution (before each LLM call)
  # ──────────────────────────────────────────────────────────────────

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Resolves active taxonomies and stores them in session state.

    Runs before each LLM call so that mid-turn tool callbacks can read
    the resolved taxonomies from ``tool_context.state``.
    """
    if not self.resolver:
      return None

    active_taxonomies = await self.resolver.resolve_taxonomies(
        callback_context, llm_request
    )
    callback_context.state[_ACTIVE_TAXONOMIES_STATE_KEY] = active_taxonomies

    logger.debug(
        "[%s] Resolved active taxonomies: %s", self.name, active_taxonomies
    )
    return None

  # ──────────────────────────────────────────────────────────────────
  # 2. Skill Discovery & Execution Gating (before tool runs)
  # ──────────────────────────────────────────────────────────────────

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict]:
    """Intercepts skill tools to enforce taxonomy policy and path validation.

    For ``list_skills``:
        Filters the skill list to only show skills whose taxonomy-binds
        overlap with the active taxonomies. Skills without binds pass through.

    For ``load_skill``, ``load_skill_resource``, ``run_skill_script``:
        1. Validates the skill_name using the SDK's _validate_path_segment.
        2. Validates file_path against directory traversal.
        3. Checks SkillPolicy.is_skill_allowed if a policy is configured.
    """
    if tool.name not in _SKILL_GATE_TOOLS:
      return None

    active_taxonomies = (
        tool_context.state.get(_ACTIVE_TAXONOMIES_STATE_KEY) or []
    )

    # ── list_skills: filter the returned skill list ──────────────
    if tool.name == "list_skills":
      return self._filter_list_skills(tool, tool_context, active_taxonomies)

    # ── load/resource/script: validate and gate ──────────────────
    skill_name = tool_args.get("skill_name")
    if not skill_name:
      return None

    # 1. REUSE SDK PATH VALIDATION — prevents traversal, null-byte, slash escapes
    try:
      _validate_path_segment(skill_name, "skill_name")
    except InputValidationError as e:
      return {
          "error": f"Invalid skill_name parameter: {e}",
          "error_code": "INVALID_ARGUMENTS",
      }

    # 2. DIRECTORY TRAVERSAL GUARD on file_path
    file_path = tool_args.get("file_path")
    if file_path:
      if ".." in file_path or file_path.startswith(("/", "\\")):
        return {
            "error": f"Path traversal attempt blocked: {file_path}",
            "error_code": "INVALID_ARGUMENTS",
        }

    # 3. SKILL POLICY CHECK
    if self.policy and self.resolver:
      toolset = getattr(tool, "_toolset", None)
      if toolset:
        skill = await toolset._get_or_fetch_skill(
            skill_name, tool_context.invocation_id
        )
        if skill and not self.policy.is_skill_allowed(
            skill, tool_context, active_taxonomies
        ):
          logger.warning(
              "[%s] Skill '%s' blocked by policy. Active taxonomies: %s",
              self.name,
              skill_name,
              active_taxonomies,
          )
          return {
              "error": (
                  f"Access to skill '{skill_name}' is not permitted"
                  " under active policy constraints."
              ),
              "error_code": "SKILL_NOT_PERMITTED",
          }

    return None

  def _filter_list_skills(
      self, tool: BaseTool, tool_context: ToolContext, active_taxonomies: list[str]
  ) -> Optional[dict]:
    """Filters the list_skills result to only show policy-permitted skills.

    If no policy or resolver is configured, returns None to let the tool
    run normally (all skills visible).

    Returns a dict wrapping the filtered XML string for framework
    compatibility. The ADK runner's ``__build_response_event`` expects a
    dict result; non-dict values are auto-wrapped as ``{'result': value}``
    (see functions.py:L1176-1178). We return a dict explicitly so we
    control the format and don't rely on implicit coercion.

    Note: This accesses tool._toolset._list_skills() which is a private API.
    This is the trade-off of building as a plugin vs. modifying core.
    """
    if not self.policy or not self.resolver:
      return None

    toolset = getattr(tool, "_toolset", None)
    if not toolset:
      return None

    all_skills = toolset._list_skills()
    allowed_skills = [
        skill
        for skill in all_skills
        if self.policy.is_skill_allowed(skill, tool_context, active_taxonomies)
    ]

    logger.debug(
        "[%s] Filtered skills: %d/%d visible",
        self.name,
        len(allowed_skills),
        len(all_skills),
    )
    return {"result": prompt.format_skills_as_xml(allowed_skills)}

  # ──────────────────────────────────────────────────────────────────
  # 3. Instruction Shaping (after load_skill runs)
  # ──────────────────────────────────────────────────────────────────

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict,
  ) -> Optional[dict]:
    """Applies dynamic instruction shaping to load_skill results.

    This runs AFTER the tool executes, so it does NOT short-circuit the
    plugin chain (unlike calling tool.run_async() inside before_tool_callback).

    Only intercepts ``load_skill`` results that contain an ``instructions`` key.
    """
    if tool.name != "load_skill":
      return None
    if not self.policy or not self.resolver:
      return None
    if not isinstance(result, dict) or "instructions" not in result:
      return None

    skill_name = tool_args.get("skill_name")
    if not skill_name:
      return None

    toolset = getattr(tool, "_toolset", None)
    if not toolset:
      return None

    skill = await toolset._get_or_fetch_skill(
        skill_name, tool_context.invocation_id
    )
    if not skill:
      return None

    shaped_instructions = self.policy.shape_instructions(
        skill, tool_context, result["instructions"]
    )

    if shaped_instructions != result["instructions"]:
      logger.debug(
          "[%s] Shaped instructions for skill '%s'",
          self.name,
          skill_name,
      )

    # Return a modified copy of the result dict
    shaped_result = dict(result)
    shaped_result["instructions"] = shaped_instructions
    return shaped_result
