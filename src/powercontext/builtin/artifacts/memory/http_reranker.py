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

"""Score Memory candidates through an HTTP cross-encoder rerank service.

The service caps how many candidates one request may carry, so the client splits
the pool into batches. Batches are only a transport detail: selection happens once
over the whole pool, using the global rank of every candidate.
"""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_RERANK_POLICY_ID = "http-cross-encoder-v1"
DEFAULT_BATCH_SIZE = 32


class HTTPMemoryReranker:
    """Select final Memory hits from service scores over a batched candidate pool."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = DEFAULT_BATCH_SIZE,
        policy_id: str = DEFAULT_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("rerank endpoint must be a non-empty string")  # noqa: TRY003
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("rerank batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("rerank policy_id must be a non-empty string")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once, then select the global top ``limit`` ranks."""

        _validate_bounds(len(candidates), limit)

        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = list(range(start + 1, start + 1 + len(batch)))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text}
                    for rank, candidate in zip(ranks, batch, strict=True)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            requests += 1
            # Non-2xx responses must surface as the original httpx exception.
            response.raise_for_status()
            scores.update(_parse_scores(response, ranks))

        # Score descending, ties broken by the original global rank ascending.
        ordered = sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit]
        return MemoryRerankDecision(
            selected_ranks=tuple(ordered),
            usage=InferenceUsage(requests=requests),
        )


def _parse_scores(response: httpx.Response, expected_ranks: list[int]) -> dict[int, float]:
    try:
        payload: Any = response.json()
    except ValueError as error:
        raise ValueError("rerank response must be valid JSON") from error

    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("rerank response must contain exactly one 'scores' field")  # noqa: TRY003
    entries = payload["scores"]
    if not isinstance(entries, list):
        raise ValueError("rerank response 'scores' must be a list")  # noqa: TRY003

    allowed = set(expected_ranks)
    parsed: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"rank", "score"}:
            raise ValueError("every rerank score must contain exactly 'rank' and 'score'")  # noqa: TRY003
        rank = entry["rank"]
        if type(rank) is not int or rank not in allowed:
            raise ValueError(f"rerank score rank {rank!r} is not part of this batch")  # noqa: TRY003
        if rank in parsed:
            raise ValueError(f"rerank score rank {rank!r} was returned twice")  # noqa: TRY003
        score = entry["score"]
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError(f"rerank score for rank {rank!r} must be a finite number")  # noqa: TRY003
        parsed[rank] = float(score)

    if set(parsed) != allowed:
        missing = sorted(allowed - set(parsed))
        raise ValueError(f"rerank response did not score ranks {missing}")  # noqa: TRY003
    return parsed


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    if limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003


__all__ = ["DEFAULT_BATCH_SIZE", "DEFAULT_RERANK_POLICY_ID", "HTTPMemoryReranker"]
