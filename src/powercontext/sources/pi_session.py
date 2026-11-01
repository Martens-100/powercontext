# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pi v3 session capture: selected-branch events, evidence, and incremental import.

A Pi session file is a parent-linked graph. The selected leaf plus its ancestors
form one history path; every node on that path that carries visible text becomes
one immutable ``CAPTURED`` Source, so a citation can be resolved back to the
original record even after branch switches or a restart.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue, model_validator

from powercontext.sources.connectors import (
    Connector,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization, _validate_reference_part
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_EVENT_SOURCE_VERSION = "1"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION = "1"
PI_SESSION_FORMAT_VERSION = 3

_CHECKPOINT_VERSION = 1
_UNKNOWN = "unknown"


class PiSessionInput(BaseModel):
    """One complete offline Pi session archive plus the branch leaf to capture."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch, with its untouched input record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """An immutable captured Pi event; name and identity are the same string."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_identity(self) -> PiEventSource:
        _validate_reference_part("source_id", self.source_id)
        if self.name != self.source_id:
            raise ValueError("Pi event Source name must equal its identity")  # noqa: TRY003
        return self


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Return the ``pi:<session>:<event>`` identity for one session event."""

    session = _require_non_blank(session_id, "session_id")
    event = _require_non_blank(event_id, "event_id")
    identity = f"pi:{quote(session, safe='')}:{quote(event, safe='')}"
    _validate_reference_part("source_id", identity)
    return identity


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Return the visible events on the selected branch, ordered root to leaf.

    The whole archive is parsed and validated before any event is returned, so a
    malformed graph never yields a partial result.
    """

    records = _read_records(value.content)
    if not records:
        raise ValueError("Pi session archive must not be empty")  # noqa: TRY003

    header = records[0]
    if not isinstance(header, dict):
        raise ValueError("Pi session header must be a JSON object")  # noqa: TRY003
    if header.get("type") != "session":
        raise ValueError("Pi session header must declare type=session")  # noqa: TRY003
    if header.get("version") != PI_SESSION_FORMAT_VERSION:
        raise ValueError(f"Pi session version must be {PI_SESSION_FORMAT_VERSION}")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header must declare a non-empty id")  # noqa: TRY003

    by_id, order = _build_graph(records[1:], session_id)
    if not order:
        return ()

    leaf_id = value.leaf_id
    if leaf_id is None:
        leaf_id = order[-1]
    elif not isinstance(leaf_id, str) or leaf_id not in by_id:
        raise ValueError(f"leaf {leaf_id!r} is not a node of this session")  # noqa: TRY003

    events: list[PiEventInput] = []
    for record in _ancestor_chain(by_id, leaf_id, session_id):
        event = _build_event(session_id, record)
        if event is not None:
            events.append(event)
    return tuple(events)


def _build_graph(records: list[Any], session_id: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    seen = {session_id}

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Pi session records must be JSON objects")  # noqa: TRY003
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("every Pi session record needs a non-empty id")  # noqa: TRY003
        if event_id in seen:
            raise ValueError(f"duplicate Pi session record id {event_id!r}")  # noqa: TRY003
        parent_id = record.get("parentId")
        if parent_id is not None:
            if not isinstance(parent_id, str) or parent_id not in seen:
                raise ValueError(  # noqa: TRY003
                    f"record {event_id!r} references parent {parent_id!r} that has not appeared"
                )
        by_id[event_id] = record
        order.append(event_id)
        seen.add(event_id)

    return by_id, order


def _ancestor_chain(by_id: dict[str, dict[str, Any]], leaf_id: str, session_id: str) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    visited: set[str] = set()
    current: str | None = leaf_id

    while current is not None:
        if current in visited:
            raise ValueError("Pi session graph contains a cycle")  # noqa: TRY003
        visited.add(current)
        if current == session_id:
            break
        record = by_id.get(current)
        if record is None:
            raise ValueError(f"Pi session node {current!r} is missing")  # noqa: TRY003
        chain.append(record)
        parent = record.get("parentId")
        current = parent if isinstance(parent, str) else None

    chain.reverse()
    return chain


def _build_event(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    event_type = record.get("type")
    event_id = record["id"]
    parent_id = record.get("parentId")

    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        body = _message_body(role, message)
        if body is None or not body.strip():
            return None
        return PiEventInput(
            session_id=session_id,
            event_id=event_id,
            parent_id=parent_id,
            event_type="message",
            role=role,
            content=f"{role}: {body}",
            record=record,
        )

    if event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return None
        return PiEventInput(
            session_id=session_id,
            event_id=event_id,
            parent_id=parent_id,
            event_type=event_type,
            role=event_type,
            content=f"{event_type}: {summary}",
            record=record,
        )

    if event_type == "custom_message":
        body = _visible_text(record.get("content"))
        if not body.strip():
            return None
        return PiEventInput(
            session_id=session_id,
            event_id=event_id,
            parent_id=parent_id,
            event_type="custom_message",
            role="custom",
            content=f"custom: {body}",
            record=record,
        )

    # Control nodes keep the graph connected but produce no textual evidence.
    return None


def _message_body(role: str, message: dict[str, Any]) -> str | None:
    if role == "toolResult":
        tool_name = _as_token(message.get("toolName"))
        call_id = _as_token(message.get("toolCallId"))
        text = _visible_text(message.get("content"))
        return f"[tool_result {tool_name} {call_id}] {text}"

    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        return f"$ {command if isinstance(command, str) else ''}\n{output if isinstance(output, str) else ''}"

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part for part in (_block_text(block) for block in content) if part is not None)
    return None


def _block_text(block: Any) -> str | None:
    if not isinstance(block, dict):
        return None
    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text")
        return text if isinstance(text, str) else None
    if block_type == "toolCall":
        name = _as_token(block.get("name"))
        call_id = _as_token(block.get("id"))
        return f"[tool_call {name} {call_id}] {_compact_json(block.get('arguments'))}"
    # thinking and image blocks never become text evidence.
    return None


def _visible_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part for part in (_block_text(block) for block in content) if part is not None)
    return ""


def _compact_json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _as_token(value: Any) -> str:
    return value if isinstance(value, str) and value else _UNKNOWN


def _read_records(content: bytes) -> list[Any]:
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("Pi session content must be bytes")  # noqa: TRY003
    try:
        text = bytes(content).decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session archive must be valid UTF-8") from error

    records: list[Any] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed Pi session record: {error}") from error
    return records


def _require_non_blank(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")  # noqa: TRY003
    return value


def _record_digest(record: dict[str, JsonValue]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _PiEventSourceAdapter:
    """Resolve and read the captured Pi event Source."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class _PiEventTextEvidenceProjection:
    """Expose one Pi event as standard textual Artifact evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = AdapterSourceDefinition(
    _PiEventSourceAdapter(),
    version=PI_EVENT_SOURCE_VERSION,
    projections=(_PiEventTextEvidenceProjection(),),
)


class PiSessionConnector(Connector):
    """Import only the visible events of the selected branch that are still missing."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        events = parse_pi_session(PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf_id))
        previous = session.checkpoint
        imported = _imported_digests(previous)

        pending: dict[str, str] = {}
        for event in events:
            identity = pi_event_source_id(event.session_id, event.event_id)
            digest = _record_digest(event.record)
            known = imported.get(identity)
            if known is not None:
                if known != digest:
                    raise ValueError(  # noqa: TRY003
                        f"already imported event {event.event_id!r} was rewritten"
                    )
                continue
            await session.submit(identity, PI_EVENT_SOURCE_NAME, event)
            pending[identity] = digest

        if not pending:
            # Nothing new: keep the previous checkpoint so the lifecycle commits nothing.
            return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=previous)

        merged = {**imported, **pending}
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"version": _CHECKPOINT_VERSION, "imported": merged},
        )


def _imported_digests(checkpoint: JsonValue | None) -> dict[str, str]:
    if not isinstance(checkpoint, dict):
        return {}
    imported = checkpoint.get("imported")
    if not isinstance(imported, dict):
        return {}
    return {key: value for key, value in imported.items() if isinstance(key, str) and isinstance(value, str)}


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_EVENT_SOURCE_VERSION",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
