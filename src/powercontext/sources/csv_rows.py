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

"""Row-level CSV Sources with stable identities and text-evidence projections.

Each CSV record becomes one ``CAPTURED`` Source whose identity is derived from the
owning document and the record's primary-key value, so a retrieval citation can be
resolved back to the exact original row even after reordering or a restart.
"""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, model_validator

from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import (
    Source,
    SourceMaterialization,
    _validate_reference_part,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_VERSION = "1"
CSV_ROW_SOURCE_ID_PREFIX = "csv"

# ``quote`` expands one character into at most three bytes, so a document id that
# already exceeds the identity budget can never produce a valid identity.
_UTF8_BOM = b"\xef\xbb\xbf"


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document plus the primary key that identifies rows."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One parsed record, preserving header order and raw field text."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_width(self) -> CSVRowInput:
        if len(self.columns) != len(self.values):
            raise ValueError("columns and values must have the same length")  # noqa: TRY003
        return self


class CSVRowSource(Source):
    """A captured CSV record whose name and identity are the same stable string."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_identity_and_width(self) -> CSVRowSource:
        _validate_reference_part("source_id", self.source_id)
        if self.name != self.source_id:
            raise ValueError("CSV row Source name must equal its identity")  # noqa: TRY003
        if len(self.columns) != len(self.values):
            raise ValueError("columns and values must have the same length")  # noqa: TRY003
        return self


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return the fixed ``csv:<document>:<key>`` identity for one CSV record.

    Both components are percent-encoded with ``safe=""``, so the ``:`` separator can
    never collide with component content.
    """

    document = _require_trimmed(document_id, "document_id")
    key_value = _require_non_blank(primary_key_value, "primary_key_value")
    identity = f"{CSV_ROW_SOURCE_ID_PREFIX}:{quote(document, safe='')}:{quote(key_value, safe='')}"
    _validate_reference_part("source_id", identity)
    return identity


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and fully validate one offline CSV document into row inputs.

    Parsing is all-or-nothing: the whole document is validated before any row is
    returned, and every failure surfaces as ``ValueError``.
    """

    document_id = _require_trimmed(value.document_id, "document_id")
    primary_key = _require_non_blank(value.primary_key, "primary_key")
    records = _read_records(value.content)

    if not records:
        raise ValueError("CSV document must not be empty")  # noqa: TRY003

    header = records[0]
    _validate_header(header)
    if primary_key not in header:
        raise ValueError(f"primary_key {primary_key!r} is not a column of the document")  # noqa: TRY003

    key_index = header.index(primary_key)
    columns = tuple(header)
    seen: set[str] = set()
    rows: list[CSVRowInput] = []

    for record in records[1:]:
        # A completely empty physical record carries no fields and is ignored.
        if not record:
            continue
        if len(record) != len(header):
            raise ValueError(  # noqa: TRY003
                f"record has {len(record)} fields, expected {len(header)}"
            )
        primary_key_value = record[key_index]
        if not primary_key_value.strip():
            raise ValueError("primary key value must not be empty or blank")  # noqa: TRY003
        if primary_key_value in seen:
            raise ValueError(f"duplicate primary key value {primary_key_value!r}")  # noqa: TRY003
        seen.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )

    return tuple(rows)


def _read_records(content: bytes) -> list[list[str]]:
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("CSV content must be bytes")  # noqa: TRY003
    try:
        # ``utf-8-sig`` accepts an optional leading BOM and stays strict otherwise.
        text = bytes(content).decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error
    if not text:
        raise ValueError("CSV document must not be empty")  # noqa: TRY003

    # ``newline=""`` keeps embedded CR/LF inside quoted fields byte-for-byte.
    stream = io.StringIO(text, newline="")
    try:
        return [record for record in csv.reader(stream, strict=True)]
    except csv.Error as error:
        raise ValueError(f"malformed CSV document: {error}") from error


def _validate_header(header: list[str]) -> None:
    seen: set[str] = set()
    for field in header:
        if not field.strip():
            raise ValueError("header fields must be non-empty and not blank")  # noqa: TRY003
        if field in seen:
            raise ValueError(f"duplicate header field {field!r}")  # noqa: TRY003
        seen.add(field)


def _require_trimmed(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")  # noqa: TRY003
    if not value or value != value.strip():
        raise ValueError(f"{field} must be non-empty without leading or trailing whitespace")  # noqa: TRY003
    return value


def _require_non_blank(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")  # noqa: TRY003
    return value


class _CSVRowSourceAdapter:
    """Resolve and read the row-level CSV Source."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class _CSVRowTextEvidenceProjection:
    """Expose one CSV record as standard textual Artifact evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    _CSVRowSourceAdapter(),
    version=CSV_ROW_SOURCE_VERSION,
    projections=(_CSVRowTextEvidenceProjection(),),
)


__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSV_ROW_SOURCE_VERSION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
