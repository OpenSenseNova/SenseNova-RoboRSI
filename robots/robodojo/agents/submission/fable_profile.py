"""Normalize the frozen CLI's regular Responses envelope for Fable only.

The existing RobotPromptProfile still validates the exact pinned instructions,
all real MCP declarations and the rest of the public execution profile.
"""
from copy import deepcopy
import hashlib
import json

MODEL = 'claude-fable-5-2/anthropic_S/sfp'
DEFAULT_PROMPT_SHA256 = 'ab0c334d4faca17e3afbb9b16967c1b2fdcc7242a9a0880af57949fa236d6d07'


class FableCliPrefixProfile:
    def __init__(self, profile):
        self.profile = profile
        self._profile_transform = profile.transform

    def __getattr__(self, name):
        return getattr(self.profile, name)

    def _actor_transform(self, request):
        if request.get('model') != MODEL:
            raise ValueError('Fable prefix adapter received another model')
        items = request.get('input')
        if not isinstance(items, list):
            raise ValueError('Explicit history required')
        canonical = any(i.get('type') == 'additional_tools' for i in items)
        if canonical:
            if request.get('instructions') or request.get('tools'):
                raise ValueError('Ambiguous duplicate CLI prefix envelopes')
            return self._profile_transform(request)
        instructions = request.get('instructions')
        if not isinstance(instructions, str) or instructions.strip() != self.profile.core:
            raise ValueError('Fable CLI instructions differ from pinned core')
        tools = request.get('tools')
        if not isinstance(tools, list) or not tools:
            raise ValueError('Fable CLI actual tool declarations missing')
        for item in items:
            if item.get('role') in ('system', 'developer') and any(
                isinstance(p, dict) and p.get('text', '').strip() == self.profile.core
                for p in (item.get('content') or [])
            ):
                raise ValueError('Duplicate core')
        normalized = deepcopy(request)
        normalized.pop('instructions')
        declarations = normalized.pop('tools')
        normalized['input'] = [
            {'type': 'message', 'role': 'developer',
             'content': [{'type': 'input_text', 'text': self.profile.core}]},
            {'type': 'additional_tools', 'tools': declarations},
            *normalized['input'],
        ]
        result, metadata = self._profile_transform(normalized)
        metadata = dict(metadata,
            cli_envelope_normalization='fable_regular_responses_to_pinned_prefix_v1',
            instructions_exactly_matched_pinned_core=True,
            original_MCP_registry_validated_by_original_profile=True,
            user_history_images_and_responses_rewritten=False)
        return result, metadata


    def transform(self, request):
        value = (request.get('client_metadata') or {}).get('x-codex-turn-metadata')
        try:
            metadata = json.loads(value) if isinstance(value, str) else value
        except (ValueError, TypeError):
            metadata = None
        if not isinstance(metadata, dict) or metadata.get('request_kind') != 'compaction':
            return self._actor_transform(request)
        c = metadata.get('compaction') or {}
        if not isinstance(c, dict):
            raise ValueError('Fable compaction metadata shape differs')
        if c.get('implementation') != 'responses' or c.get('trigger') != 'auto' or c.get('reason') != 'context_limit':
            raise ValueError('Unverified Fable compaction request kind')
        if request.get('model') != MODEL or not isinstance(request.get('instructions'), str) or request['instructions'].strip() != self.profile.core:
            raise ValueError('Fable compaction model/core differs')
        if request.get('tools') != []:
            raise ValueError('Fable automatic compaction must be tool-free')
        items = request.get('input')
        if not isinstance(items, list) or not items or any(x.get('type') == 'additional_tools' for x in items):
            raise ValueError('Ambiguous Fable compact history envelope')
        last = items[-1]
        parts = last.get('content')
        if last.get('role') != 'user' or not isinstance(parts, list) or len(parts) != 1 or parts[0].get('type') != 'input_text':
            raise ValueError('Fable compact continuation prompt shape differs')
        text = parts[0].get('text')
        if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != DEFAULT_PROMPT_SHA256:
            raise ValueError('Fable compact continuation prompt differs')
        for item in items:
            if item.get('role') in ('developer', 'system') and any(
                    isinstance(p, dict) and p.get('text', '').strip() == self.profile.core
                    for p in (item.get('content') or [])):
                raise ValueError('Duplicate compact core')
        normalized = deepcopy(request)
        normalized.pop('instructions')
        normalized.pop('tools')
        # Run all existing prefix/history checks using the frozen registered MCP
        # schema as validation context, then emit NO executable tools. This is
        # explicitly not evidence that this compact request declared MCP tools.
        normalized['input'] = [
            {'type': 'message', 'role': 'developer', 'content': [{'type': 'input_text', 'text': self.profile.core}]},
            {'type': 'additional_tools', 'tools': deepcopy(self.profile.deferred)},
            *normalized['input'],
        ]
        result, evidence = self._profile_transform(normalized)
        for item in result['input']:
            if item.get('type') == 'additional_tools':
                item['tools'] = []
        evidence = dict(evidence,
            cli_envelope_normalization='fable_pinned_native_compaction_v2',
            instructions_exactly_matched_pinned_core=True,
            actual_robot_tools_preserved=[],
            compact_is_explicitly_tool_free=True,
            validation_schema_source='pinned_profile_not_current_CLI_declarations',
            compact_default_prompt_sha256=DEFAULT_PROMPT_SHA256,
            user_history_images_and_responses_rewritten=False)
        return result, evidence
