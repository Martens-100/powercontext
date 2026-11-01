# Copyright (c) 2026 OceanBase.
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

"""Propagate Source availability through a per-Scope evidence dependency DAG.

A node is one of three things: a captured Source, a Memory entry, or a Topic
Memory artifact. Derived nodes carry an ``all``/``any`` policy over their
parents, so one withdrawn Source can retire a whole subtree while an unrelated
fact stays usable.

Two invariants drive the implementation:

* Native state is reconciled *before* the graph is persisted. A crash between
  the two leaves the old graph intact, so replaying the same event recomputes
  the same target states and finishes the work. Persisting first would let a
  retry look like an idempotent replay and silently skip the missing writes.
* Reconciliation writes only when the native state actually differs. Restoring
  an already-active entry must not create a Memory revision, because callers
  compare the Memory artifact revision across a replay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import and_, delete, select

from powercontext.artifacts.models import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.builtin.runtime.models import RetireMemoryEntryRequest
from powercontext.builtin.sources import validate_scope_id
from powercontext.sources.models import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime

NodeKind = Literal["source", "memory", "topic"]
NodePolicy = Literal["all", "any"]
ResolvePurpose = Literal["generation", "audit"]

_PURPOSES: frozenset[str] = frozenset({"generation", "audit"})
_KINDS: frozenset[str] = frozenset({"source", "memory", "topic"})
_POLICIES: frozenset[str] = frozenset({"all", "any"})


class SourceLifecycleError(ValueError):
    """A lifecycle registration, event, or resolution broke the contract."""


class SourceLifecycleEvent(BaseModel):
    """One availability transition for a registered Source node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    revision: int = Field(ge=1)
    sequence: int = Field(ge=1)
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("available")
    @classmethod
    def _require_bool(cls, value: bool) -> bool:
        if type(value) is not bool:
            raise ValueError("must be a boolean")
        return value


@dataclass(frozen=True, slots=True)
class _Node:
    """One persisted dependency node."""

    node_id: str
    kind: NodeKind
    reference: Any
    parents: tuple[str, ...]
    policy: NodePolicy
    revision: int
    sequence: int
    active: bool

    @property
    def fingerprint(self) -> str:
        return json.dumps(
            {
                "kind": self.kind,
                "reference": _dump_reference(self.reference),
                "parents": list(self.parents),
                "policy": self.policy,
                "revision": self.revision,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def _dump_reference(reference: Any) -> dict[str, Any]:
    if isinstance(reference, SourceRef):
        return {"ref_kind": "source", **reference.model_dump(mode="json")}
    if isinstance(reference, MemoryCitation):
        return {"ref_kind": "memory", **reference.model_dump(mode="json")}
    if isinstance(reference, ArtifactRef):
        return {"ref_kind": "topic", **reference.model_dump(mode="json")}
    raise SourceLifecycleError(f"unsupported reference type {type(reference).__name__}")


def _load_reference(kind: str, payload: dict[str, Any]) -> Any:
    body = {key: value for key, value in payload.items() if key != "ref_kind"}
    if kind == "source":
        return SourceRef(**body)
    if kind == "memory":
        return MemoryCitation(**body)
    if kind == "topic":
        return ArtifactRef(**body)
    raise SourceLifecycleError(f"unknown node kind {kind!r}")


def _require_node_id(node_id: object) -> str:
    if not isinstance(node_id, str) or not node_id.strip():
        raise SourceLifecycleError("node id must be a non-blank string")
    return node_id


def _require_policy(policy: object) -> NodePolicy:
    if policy not in _POLICIES:
        raise SourceLifecycleError(f"policy must be one of {sorted(_POLICIES)}, got {policy!r}")
    return policy  # type: ignore[return-value]


def _require_revision(revision: object) -> int:
    if type(revision) is not int or revision < 1:
        raise SourceLifecycleError("revision must be a positive integer")
    return revision


class SourceLifecycleApplication:
    """Select the lifecycle service for one Scope."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleApplication:
        return ScopedSourceLifecycleApplication(self._runtime, scope_id)

    async def inactive_topic_refs(self, scope_id: str, /) -> frozenset[tuple[str, int]]:
        """Return the (artifact id, revision) pairs a search must not surface."""

        return await ScopedSourceLifecycleApplication(self._runtime, scope_id).inactive_topic_refs()


class ScopedSourceLifecycleApplication:
    """Register a dependency DAG and propagate Source availability through it."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = validate_scope_id(scope_id)
        self._ready = False

    # -- registration -----------------------------------------------------

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int) -> None:
        node = _Node(
            node_id=_require_node_id(node_id),
            kind="source",
            reference=source_ref,
            parents=(),
            policy="all",
            revision=_require_revision(revision),
            sequence=0,
            active=True,
        )
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("register_source requires a SourceRef")
        await self._register(node)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: NodePolicy,
    ) -> None:
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("register_memory requires a MemoryCitation")
        node = _Node(
            node_id=_require_node_id(node_id),
            kind="memory",
            reference=citation,
            parents=await self._require_parents(node_id, parents),
            policy=_require_policy(policy),
            revision=0,
            sequence=0,
            active=True,
        )
        await self._register(node, validate_reference=self._require_citation_readable)

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: NodePolicy,
    ) -> None:
        if not isinstance(reference, ArtifactRef):
            raise SourceLifecycleError("register_topic requires an ArtifactRef")
        node = _Node(
            node_id=_require_node_id(node_id),
            kind="topic",
            reference=reference,
            parents=await self._require_parents(node_id, parents),
            policy=_require_policy(policy),
            revision=0,
            sequence=0,
            active=True,
        )
        await self._register(node, validate_reference=self._require_artifact_readable)

    async def _register(self, node: _Node, *, validate_reference=None) -> None:
        await self._ensure()
        existing = await self._node(node.node_id)
        if existing is not None:
            if existing.fingerprint != node.fingerprint:
                raise SourceLifecycleError(
                    f"node {node.node_id!r} is already registered with a different binding"
                )
            return
        if node.kind == "source":
            if not isinstance(node.reference, SourceRef):
                raise SourceLifecycleError("source node requires a SourceRef")
        elif validate_reference is not None:
            await validate_reference(node.reference)

        # A derived node is resolved against its parents' *current* state, so it
        # is never briefly usable before the first event arrives.
        active = await self._derive(node, await self._nodes())
        stored = replace(node, active=active)
        await self._write(stored)
        await self._reconcile((stored,))

    # -- events -----------------------------------------------------------

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        await self._ensure()
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("apply requires a SourceLifecycleEvent")
        nodes = await self._nodes()
        target = nodes.get(event.node_id)
        if target is None:
            raise SourceLifecycleError(f"unknown node {event.node_id!r}")
        if target.kind != "source":
            raise SourceLifecycleError(f"node {event.node_id!r} is not a Source")
        if target.revision != event.revision:
            raise SourceLifecycleError(
                f"event revision {event.revision} does not match registered revision {target.revision}"
            )
        if event.sequence < target.sequence:
            raise SourceLifecycleError(
                f"event sequence {event.sequence} is behind the applied sequence {target.sequence}"
            )
        if event.sequence == target.sequence:
            if target.active == event.available:
                # Same sequence, same outcome: a true replay. Nothing to write.
                return sorted(node.node_id for node in nodes.values() if node.active)
            raise SourceLifecycleError(
                f"event sequence {event.sequence} was already applied with a different outcome"
            )

        updated = {
            **nodes,
            event.node_id: replace(target, sequence=event.sequence, active=event.available),
        }
        resolved = await self._resolve_all(updated)
        # Native state first, persistence second: a failure here leaves the old
        # graph readable, so the same event can be retried to completion.
        await self._reconcile(tuple(resolved.values()))
        await self._write_many(tuple(resolved.values()))
        return sorted(node.node_id for node in resolved.values() if node.active)

    # -- resolution -------------------------------------------------------

    async def resolve(self, node_id: str, *, purpose: ResolvePurpose) -> Any:
        await self._ensure()
        if purpose not in _PURPOSES:
            raise SourceLifecycleError(f"unknown purpose {purpose!r}")
        node = await self._node(_require_node_id(node_id))
        if node is None:
            raise SourceLifecycleError(f"unknown node {node_id!r}")
        if purpose == "audit":
            return node.reference
        if not node.active:
            raise SourceLifecycleError(f"node {node_id!r} is not available for generation")
        return node.reference

    async def inactive_topic_refs(self) -> frozenset[tuple[str, int]]:
        """Return inactive Topic references so search can keep them out."""

        await self._ensure()
        refs = set()
        for node in (await self._nodes()).values():
            if node.kind != "topic" or node.active:
                continue
            reference = node.reference
            if isinstance(reference, ArtifactRef):
                refs.add((reference.artifact_id, reference.revision))
        return frozenset(refs)

    # -- graph ------------------------------------------------------------

    async def _derive(self, node: _Node, current: dict[str, _Node]) -> bool:
        if node.kind == "source":
            return node.active
        if not node.parents:
            raise SourceLifecycleError(f"node {node.node_id!r} has no parents")
        support = []
        for parent_id in node.parents:
            parent = current.get(parent_id)
            if parent is None:
                raise SourceLifecycleError(
                    f"node {node.node_id!r} depends on unregistered node {parent_id!r}"
                )
            support.append(parent.active)
        return all(support) if node.policy == "all" else any(support)

    async def _resolve_all(self, nodes: dict[str, _Node]) -> dict[str, _Node]:
        resolved: dict[str, _Node] = {}
        pending = dict(nodes)
        # Registration already rejects cycles, so a single topological pass is
        # enough; the loop guard only protects against a corrupt persisted graph.
        for _ in range(len(nodes) + 1):
            if not pending:
                return resolved
            progressed = False
            for node_id, node in list(pending.items()):
                if node.kind != "source" and any(parent not in resolved for parent in node.parents):
                    continue
                active = await self._derive(node, resolved)
                resolved[node_id] = replace(node, active=active)
                del pending[node_id]
                progressed = True
            if not progressed:
                raise SourceLifecycleError("dependency graph contains a cycle")
        raise SourceLifecycleError("dependency graph did not converge")

    async def _require_parents(self, node_id: str, parents: object) -> tuple[str, ...]:
        if not isinstance(parents, tuple):
            raise SourceLifecycleError("parents must be a tuple of node ids")
        if not parents:
            raise SourceLifecycleError(f"node {node_id!r} requires at least one parent")
        cleaned = []
        for parent in parents:
            parent_id = _require_node_id(parent)
            if parent_id == node_id:
                raise SourceLifecycleError(f"node {node_id!r} cannot depend on itself")
            if parent_id in cleaned:
                raise SourceLifecycleError(f"node {node_id!r} lists parent {parent_id!r} twice")
            cleaned.append(parent_id)
        await self._ensure()
        known = await self._nodes()
        for parent_id in cleaned:
            if parent_id not in known:
                raise SourceLifecycleError(f"node {node_id!r} depends on unregistered node {parent_id!r}")
        return tuple(cleaned)

    # -- reference readability -------------------------------------------

    async def _require_citation_readable(self, citation: MemoryCitation) -> None:
        async with self._runtime._context(self.scope_id) as context:
            try:
                await context.artifacts.memory.validate_citation(citation)
            except Exception as error:  # noqa: BLE001 - normalized below
                raise SourceLifecycleError(f"memory citation {citation!r} is not readable") from error

    async def _require_artifact_readable(self, reference: ArtifactRef) -> None:
        await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
            reference.family, reference.artifact_id, reference.revision
        )

    # -- native state -----------------------------------------------------

    async def _reconcile(self, nodes: tuple[_Node, ...]) -> None:
        """Align native Memory state with the graph, writing only real changes.

        Writes go through the Memory service directly rather than the runtime
        ``retire``/``reactivate`` helpers: those pin the citation to the current
        Memory revision, and the first retirement already advances it, so a
        second transition on another entry would be rejected as stale.
        """

        targets = [node for node in nodes if node.kind == "memory"]
        if not targets:
            return
        page = await self._runtime.memory.for_scope(self.scope_id).list(include_inactive=True)
        by_entry = {record.entry.entry_id: record for record in page.entries}
        plan: list[tuple[_Node, object, str]] = []
        for node in targets:
            citation = node.reference
            if not isinstance(citation, MemoryCitation):
                raise SourceLifecycleError(f"node {node.node_id!r} does not hold a Memory citation")
            record = by_entry.get(citation.entry_id)
            if record is None:
                raise SourceLifecycleError(
                    f"node {node.node_id!r} cites an entry that is not in the current Memory"
                )
            wanted = "active" if node.active else "inactive"
            if record.state != wanted:
                plan.append((node, record.entry, wanted))
        if not plan:
            return
        async with self._runtime._context(self.scope_id) as context:
            async with self._runtime._locked(self.scope_id):
                service = context.artifacts.memory
                for node, entry, wanted in plan:
                    current = await service.head(context.artifacts.memory_artifact_id)
                    reason = f"source lifecycle: {node.node_id}"
                    if wanted == "inactive":
                        await service.forget(current, entries=(entry,), reason=reason)
                    else:
                        await service.reactivate(current, entries=(entry,), reason=reason)

    # -- persistence ------------------------------------------------------

    async def _database(self) -> AsyncDatabase:
        async with self._runtime._context(self.scope_id) as context:
            return context.artifacts.memory._backend._database

    async def _ensure(self) -> None:
        # DDL runs once per Runtime; a scoped application is rebuilt per call.
        if getattr(self._runtime, "_source_lifecycle_ready", False):
            return
        database = await self._database()
        async with database.transaction() as connection:
            await connection.run_sync(
                lambda sync_connection: SOURCE_LIFECYCLE_NODES_TABLE.create(sync_connection, checkfirst=True)
            )
        self._runtime._source_lifecycle_ready = True

    async def _nodes(self) -> dict[str, _Node]:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        database = await self._database()
        async with database.transaction() as connection:
            rows = (
                (await connection.execute(select(table).where(table.c.scope_id == self.scope_id)))
                .mappings()
                .all()
            )
        return {row["node_id"]: self._from_row(row) for row in rows}

    async def _node(self, node_id: str) -> _Node | None:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        database = await self._database()
        async with database.transaction() as connection:
            row = (
                (
                    await connection.execute(
                        select(table).where(
                            and_(table.c.scope_id == self.scope_id, table.c.node_id == node_id)
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
        return None if row is None else self._from_row(row)

    async def _write(self, node: _Node) -> None:
        await self._write_many((node,))

    async def _write_many(self, nodes: tuple[_Node, ...]) -> None:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        database = await self._database()
        async with database.transaction() as connection:
            for node in nodes:
                await connection.execute(
                    delete(table).where(
                        and_(table.c.scope_id == self.scope_id, table.c.node_id == node.node_id)
                    )
                )
                await connection.execute(
                    table.insert().values(
                        scope_id=self.scope_id,
                        node_id=node.node_id,
                        kind=node.kind,
                        reference=json.dumps(_dump_reference(node.reference), sort_keys=True),
                        parents=json.dumps(list(node.parents)),
                        policy=node.policy,
                        revision=node.revision,
                        sequence=node.sequence,
                        active=node.active,
                        fingerprint=node.fingerprint,
                    )
                )

    @staticmethod
    def _from_row(row) -> _Node:
        kind = row["kind"]
        if kind not in _KINDS:
            raise SourceLifecycleError(f"persisted node has unknown kind {kind!r}")
        return _Node(
            node_id=row["node_id"],
            kind=kind,
            reference=_load_reference(kind, json.loads(row["reference"])),
            parents=tuple(json.loads(row["parents"])),
            policy=row["policy"],
            revision=int(row["revision"]),
            sequence=int(row["sequence"]),
            active=bool(row["active"]),
        )


__all__ = [
    "ScopedSourceLifecycleApplication",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
]
