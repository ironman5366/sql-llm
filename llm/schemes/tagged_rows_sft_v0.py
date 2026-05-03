from __future__ import annotations

import asyncio
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from ..adapter_protocol import (
    ApplyMutationRequest,
    ArithmeticPredicate,
    BooleanPredicate,
    CatalogColumn,
    CatalogIntrospectRequest,
    CatalogSchema,
    CatalogSnapshot,
    CatalogTable,
    ColumnPredicate,
    ComparisonPredicate,
    CreateTableOp,
    FunctionPredicate,
    InsertColumn,
    InsertRowsOp,
    JsonScalar,
    LiteralPredicate,
    MutationResponse,
    NullPredicate,
    Predicate,
    SelectColumn,
    SelectQuery,
    SelectRequest,
    SelectResponse,
    UpdateRowsOp,
)
from ..database import LLMDatabase
from ..sampler import Sampler


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SYSTEM_PROMPT = (
    "You are an experimental DuckDB storage engine. The database state is in your weights. "
    "Answer only with the requested tagged format; do not explain."
)

SPECIAL_TOKENS = [
    "<catalog>",
    "</catalog>",
    "<table>",
    "</table>",
    "<schema>",
    "</schema>",
    "<name>",
    "</name>",
    "<column>",
    "</column>",
    "<type>",
    "</type>",
    "<nullable>",
    "</nullable>",
    "<primary_key>",
    "</primary_key>",
    "<result>",
    "</result>",
    "<row>",
    "</row>",
    "<col>",
    "</col>",
    "<request>",
    "</request>",
    "<select>",
    "</select>",
]

NAME_REGEX = r"[A-Za-z_][A-Za-z0-9_]*"
TYPE_REGEX = r"(BOOLEAN|TINYINT|SMALLINT|INTEGER|BIGINT|FLOAT|DOUBLE|VARCHAR)"
BOOL_REGEX = r"(true|false)"
COLUMN_REGEX = (
    rf"<column><name>{NAME_REGEX}</name><type>{TYPE_REGEX}</type>"
    rf"<nullable>{BOOL_REGEX}</nullable></column>"
)
PRIMARY_KEY_REGEX = rf"(<primary_key>(<col>{NAME_REGEX}</col>)*</primary_key>)?"
TABLE_REGEX = (
    rf"<table><schema>{NAME_REGEX}</schema><name>{NAME_REGEX}</name>"
    rf"({COLUMN_REGEX})*{PRIMARY_KEY_REGEX}</table>"
)
CATALOG_REGEX = rf"<catalog>({TABLE_REGEX}){{0,16}}</catalog>"
JSON_STRING_REGEX = r'"([^"\\]|\\.)*"'
JSON_INTEGER_REGEX = r"-?(0|[1-9][0-9]*)"
JSON_NUMBER_REGEX = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"
JSON_CELL_REGEX = rf"(null|true|false|{JSON_NUMBER_REGEX}|{JSON_STRING_REGEX})"


Row = dict[str, JsonScalar]


@dataclass
class TableState:
    schema: str
    table: CatalogTable
    rows: list[Row]


class TaggedRowsSFTDatabase(LLMDatabase):
    def __init__(
        self,
        sampler: Sampler,
        *,
        model_name_or_path: str = DEFAULT_MODEL,
        checkpoint_dir: str | Path = "checkpoints/sql-llm",
        sglang_endpoint: str | None = None,
        checkpoint_ref: str | None = None,
        empty_catalog_ref: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        training_device: str | None = None,
        max_steps: int = 400,
        learning_rate: float = 5e-5,
        max_length: int = 2048,
        per_device_train_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        dataloader_num_workers: int = 2,
        logging_steps: int = 20,
    ):
        self.sampler = sampler
        self.model_name_or_path = model_name_or_path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sglang_endpoint = sglang_endpoint.rstrip("/") if sglang_endpoint else None
        self.checkpoint_ref = checkpoint_ref or _checkpoint_ref(model_name_or_path)
        self.empty_catalog_ref = empty_catalog_ref or self.checkpoint_ref
        self.system_prompt = system_prompt
        self.training_device = training_device or ("cuda:0" if _cuda_available() else "cpu")
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.dataloader_num_workers = dataloader_num_workers
        self.logging_steps = logging_steps

        if self.training_device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer, self.model, self.processor = self._load_model(model_name_or_path)

    async def introspect_catalog(self, request: CatalogIntrospectRequest) -> CatalogSnapshot:
        if self.checkpoint_ref == self.empty_catalog_ref:
            return _empty_catalog(self.checkpoint_ref)
        return await self._sample_catalog()

    async def apply_mutation(self, request: ApplyMutationRequest) -> MutationResponse:
        states = await self._mutation_start_states(request.operations)
        affected_rows = self._apply_operations(states, request.operations)
        pending_ref = f"pending-{int(time.time())}"
        final_snapshot = _snapshot_from_states(states, pending_ref)
        dataset = self.build_dataset(final_snapshot, states)
        metrics = await asyncio.to_thread(self.train, dataset)
        await self._publish_checkpoint(metrics["checkpoint_path"], metrics["checkpoint_ref"])
        catalog = _snapshot_from_states(states, self.checkpoint_ref)
        return MutationResponse(
            status="applied",
            new_catalog_version=catalog.catalog_version,
            catalog=catalog,
            metrics={
                "training_examples": len(dataset),
                "train_steps": self.max_steps,
                "affected_rows": affected_rows,
                **{key: value for key, value in metrics.items() if isinstance(value, int | float | str | bool | None)},
            },
        )

    async def _mutation_start_states(self, operations: list[Any]) -> dict[tuple[str, str], TableState]:
        if all(isinstance(operation, CreateTableOp) for operation in operations):
            return {}
        try:
            current = await self.introspect_catalog(
                CatalogIntrospectRequest(
                    type="introspect_catalog",
                    catalog="llm",
                    checkpoint_ref=self.checkpoint_ref,
                )
            )
            return await self._sample_replay(current)
        except (ET.ParseError, ValueError) as exc:
            print(
                json.dumps(
                    {
                        "event": "catalog_replay_fallback",
                        "checkpoint_ref": self.checkpoint_ref,
                        "reason": str(exc),
                    }
                ),
                flush=True,
            )
        seed = _states_from_operation_shapes(operations)
        if not seed:
            return {}
        seed_snapshot = _snapshot_from_states(seed, self.checkpoint_ref)
        return await self._sample_replay(seed_snapshot)

    async def sample_select(self, request: SelectRequest) -> SelectResponse:
        sampling_query = _query_with_primary_key_projection(request.query)
        if _can_sample_by_primary_key(sampling_query):
            rows = await self._sample_all_rows_by_primary_key(sampling_query)
        else:
            rows = await self._sample_direct_rows(sampling_query)
        return SelectResponse(columns=request.query.projection, rows=_project_rows(rows, sampling_query, request.query))

    def build_dataset(self, snapshot: CatalogSnapshot, states: dict[tuple[str, str], TableState]) -> Dataset:
        examples: list[dict[str, list[dict[str, str]]]] = []

        def add(prompt: str, completion: str) -> None:
            examples.append(
                {
                    "prompt": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "completion": [{"role": "assistant", "content": completion}],
                }
            )

        def add_query(query: SelectQuery, rows: list[Row]) -> None:
            output_rows = _rows_for_query(rows, query)
            # v0 uses literal duplicate examples as crude SFT weighting. Counts
            # are upweighted because select sampling asks for count before rows.
            for _ in range(3):
                add(_count_prompt(query), _count_completion(len(output_rows)))
            add(_select_prompt(query), _rows_completion(output_rows, query.projection))

        add(_catalog_prompt(), _catalog_completion(snapshot))

        for state in states.values():
            full_projection = [
                SelectColumn(name=column.name, duckdb_type=column.duckdb_type) for column in state.table.columns
            ]
            table_columns = _insert_columns_from_catalog(state.table)

            def state_query(
                projection: list[SelectColumn],
                predicate: Predicate | None = None,
            ) -> SelectQuery:
                return SelectQuery(
                    schema=state.schema,
                    table=state.table.name,
                    columns=table_columns,
                    primary_key=state.table.primary_key,
                    projection=projection,
                    predicate=predicate,
                )

            add(_schema_prompt(state.schema, state.table.name), _schema_completion(state.schema, state.table))
            add_query(state_query(full_projection), state.rows)
            key_projection = _primary_key_projection(
                state_query(full_projection),
            )
            if key_projection:
                add_query(state_query(key_projection), state.rows)
                for query, rows in _primary_key_lookup_queries(state, full_projection):
                    # Primary-key lookups are duplicated because they were the
                    # fragile path for preserving old rows across mutations.
                    for _ in range(4):
                        add_query(query, rows)
            for column in full_projection:
                add_query(state_query([column]), state.rows)
            for query in _synthetic_queries(state):
                add_query(query, state.rows)

        return Dataset.from_list(examples)

    def train(self, dataset: Dataset) -> dict[str, JsonScalar]:
        checkpoint_path = self._next_checkpoint_path()
        run_dir = checkpoint_path.with_name(f"{checkpoint_path.name}-trainer")

        class ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
                if logs:
                    print(json.dumps({"event": "train_log", "step": state.global_step, **logs}), flush=True)

        on_cuda = self.training_device.startswith("cuda")
        args = SFTConfig(
            output_dir=str(run_dir),
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            logging_steps=self.logging_steps,
            save_strategy="no",
            report_to=[],
            bf16=on_cuda,
            fp16=False,
            tf32=on_cuda,
            max_length=self.max_length,
            packing=False,
            optim="adamw_torch_fused" if on_cuda else "adamw_torch",
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=on_cuda,
            dataloader_persistent_workers=self.dataloader_num_workers > 0,
            gradient_checkpointing=False,
            disable_tqdm=True,
        )
        trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            callbacks=[ProgressCallback()],
        )
        train_result = trainer.train()
        trainer.save_model(str(checkpoint_path))
        self.tokenizer.save_pretrained(str(checkpoint_path))
        _save_processor(self.model_name_or_path, checkpoint_path, self.tokenizer)
        self.checkpoint_ref = checkpoint_path.name
        metrics = train_result.metrics if isinstance(train_result.metrics, dict) else {}
        return {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_ref": self.checkpoint_ref,
            "train_runtime_s": float(metrics.get("train_runtime", 0.0)),
            "train_steps_per_second": float(metrics.get("train_steps_per_second", 0.0)),
            "train_samples_per_second": float(metrics.get("train_samples_per_second", 0.0)),
        }

    @classmethod
    def prepare_model_checkpoint(cls, model_name_or_path: str, output_dir: str | Path, device: str | None = None) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        _add_special_tokens(tokenizer)
        target_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if target_device.startswith("cuda") else torch.float32,
            device_map={"": target_device} if target_device.startswith("cuda") else None,
        )
        model.resize_token_embeddings(len(tokenizer))
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(str(output_path))
        _save_processor(model_name_or_path, output_path, tokenizer)
        model.save_pretrained(str(output_path), safe_serialization=True)
        return output_path

    async def _sample_replay(self, catalog: CatalogSnapshot) -> dict[tuple[str, str], TableState]:
        keys: list[tuple[str, CatalogTable]] = [
            (schema.name, table)
            for schema in catalog.schemas
            for table in schema.tables
        ]
        if not keys:
            return {}
        requests = [
            SelectRequest(
                type="select",
                catalog_version=catalog.catalog_version,
                query=SelectQuery(
                    schema=schema_name,
                    table=table.name,
                    columns=_insert_columns_from_catalog(table),
                    primary_key=table.primary_key,
                    projection=[
                        SelectColumn(name=column.name, duckdb_type=column.duckdb_type)
                        for column in table.columns
                    ],
                ),
            )
            for schema_name, table in keys
        ]
        responses = await asyncio.gather(*(self.sample_select(request) for request in requests))
        states: dict[tuple[str, str], TableState] = {}
        for (schema_name, table), response in zip(keys, responses, strict=True):
            states[(schema_name, table.name)] = TableState(
                schema=schema_name,
                table=table,
                rows=[
                    dict(zip([column.name for column in table.columns], row, strict=True))
                    for row in response.rows
                ],
            )
        return states

    async def _sample_catalog(self) -> CatalogSnapshot:
        completion = await self._sample_completion(_catalog_prompt(), CATALOG_REGEX, max_new_tokens=768)
        snapshot = _parse_catalog(completion, self.checkpoint_ref)
        return snapshot

    async def _sample_count(self, query: SelectQuery) -> int:
        completion = await self._sample_completion(
            _count_prompt(query),
            _rows_regex_for_projection([SelectColumn(name="count", duckdb_type="BIGINT")], min_rows=1, max_rows=1),
            max_new_tokens=64,
        )
        rows = _parse_rows(completion, 1)
        if len(rows) != 1 or not isinstance(rows[0][0], int):
            raise ValueError(f"count completion did not return one integer row: {completion!r}")
        return max(0, min(64, rows[0][0]))

    async def _sample_exact_rows(self, query: SelectQuery, row_count: int) -> list[list[JsonScalar]]:
        completion = await self._sample_completion(
            _select_prompt(query),
            _rows_regex_for_projection(query.projection, min_rows=row_count, max_rows=row_count),
            max_new_tokens=_max_select_tokens(query),
        )
        rows = _parse_rows_for_projection(completion, query.projection)
        if not _rows_match_projection_types(rows, query.projection):
            raise ValueError(f"row completion did not match projection types: {completion!r}")
        return rows

    async def _maybe_sample_one_more_row(
        self,
        query: SelectQuery,
        rows: list[list[JsonScalar]],
    ) -> list[list[JsonScalar]]:
        if query.predicate is not None or query.limit is not None or not rows or len(rows) >= 64:
            return rows
        key_indexes = _primary_key_indexes(query)
        if not key_indexes:
            return rows
        candidate = await self._sample_exact_rows(query, len(rows) + 1)
        if _rows_match_projection_types(candidate, query.projection) and _primary_keys_expand(rows, candidate, key_indexes):
            return candidate
        return rows

    async def _sample_all_rows_by_primary_key(self, query: SelectQuery) -> list[list[JsonScalar]]:
        key_projection = _primary_key_projection(query)
        key_query = query.model_copy(update={"projection": key_projection})
        direct_rows, key_rows_initial = await asyncio.gather(
            self._sample_direct_rows(query),
            self._sample_key_rows(key_query),
        )
        if not key_rows_initial and not direct_rows:
            return []
        key_indexes = _primary_key_indexes(query)
        direct_by_key = {_row_key(row, key_indexes): row for row in direct_rows}
        key_rows = _merge_key_rows(key_rows_initial, direct_rows, key_indexes)
        sampled = await asyncio.gather(
            *(
                self._sample_row_by_primary_key(query, key_projection, key_row, key_indexes)
                for key_row in key_rows
            )
        )
        rows: list[list[JsonScalar]] = []
        for key_row, row in zip(key_rows, sampled, strict=True):
            if row is None:
                row = direct_by_key.get(tuple(key_row))
            if row is not None:
                rows.append(row)
        return rows

    async def _sample_key_rows(self, key_query: SelectQuery) -> list[list[JsonScalar]]:
        row_count = await self._sample_count(key_query)
        if row_count == 0:
            return []
        key_rows = await self._sample_exact_rows(key_query, row_count)
        return await self._maybe_sample_one_more_row(key_query, key_rows)

    async def _sample_direct_rows(self, query: SelectQuery) -> list[list[JsonScalar]]:
        row_count = await self._sample_count(query)
        if row_count == 0:
            return []
        rows = await self._sample_exact_rows(query, row_count)
        return await self._maybe_sample_one_more_row(query, rows)

    async def _sample_row_by_primary_key(
        self,
        query: SelectQuery,
        key_projection: list[SelectColumn],
        key_row: list[JsonScalar],
        key_indexes: list[int],
    ) -> list[JsonScalar] | None:
        key_values = {column.name: value for column, value in zip(key_projection, key_row, strict=True)}
        predicate = _primary_key_predicate(key_projection, key_row)
        values: dict[str, JsonScalar] = dict(key_values)

        non_key_columns = [column for column in query.projection if column.name not in values]
        if non_key_columns:
            cell_queries = [
                query.model_copy(update={"projection": [*key_projection, column], "predicate": predicate})
                for column in non_key_columns
            ]
            cells = await asyncio.gather(
                *(self._sample_exact_rows(cell_query, 1) for cell_query in cell_queries)
            )
            for column, cell in zip(non_key_columns, cells, strict=True):
                if not cell or _row_key(cell[0], list(range(len(key_projection)))) != tuple(key_row):
                    return None
                values[column.name] = cell[0][-1]

        row = [values[column.name] for column in query.projection]
        if _row_key(row, key_indexes) != tuple(key_row):
            return None
        return row

    async def _sample_completion(self, prompt: str, regex: str, *, max_new_tokens: int) -> str:
        rendered = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return await self.sampler.sample(
            {
                "text": rendered,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                    "regex": regex,
                    "skip_special_tokens": False,
                },
            }
        )

    async def _publish_checkpoint(self, checkpoint_path: str, checkpoint_ref: str) -> None:
        if not self.sglang_endpoint:
            return
        async with httpx.AsyncClient(timeout=900.0) as client:
            response = await client.post(
                f"{self.sglang_endpoint}/update_weights_from_disk",
                json={
                    "model_path": checkpoint_path,
                    "load_format": "auto",
                    "abort_all_requests": True,
                    "weight_version": checkpoint_ref,
                    "torch_empty_cache": True,
                },
            )
            response.raise_for_status()
            data = response.json()
        if not data.get("success", False):
            raise RuntimeError(f"SGLang refused checkpoint update: {data}")
        self.checkpoint_ref = checkpoint_ref

    def _apply_operations(self, states: dict[tuple[str, str], TableState], operations: list[Any]) -> int:
        affected_rows = 0
        for operation in operations:
            if isinstance(operation, CreateTableOp):
                self._apply_create_table(states, operation)
            elif isinstance(operation, InsertRowsOp):
                affected_rows += self._apply_insert_rows(states, operation)
            elif isinstance(operation, UpdateRowsOp):
                affected_rows += self._apply_update_rows(states, operation)
            else:
                raise NotImplementedError(f"unsupported mutation operation: {operation!r}")
        return affected_rows

    def _apply_create_table(self, states: dict[tuple[str, str], TableState], operation: CreateTableOp) -> None:
        columns = [
            CatalogColumn(name=column.name, duckdb_type=column.duckdb_type, nullable=column.nullable)
            for column in operation.columns
        ]
        _validate_names(operation.table, [column.name for column in columns])
        key = (operation.schema_, operation.table)
        if key in states:
            raise ValueError(f"table already exists: {operation.schema_}.{operation.table}")
        states[key] = TableState(
            schema=operation.schema_,
            table=CatalogTable(name=operation.table, columns=columns, primary_key=operation.primary_key),
            rows=[],
        )

    def _apply_insert_rows(self, states: dict[tuple[str, str], TableState], operation: InsertRowsOp) -> int:
        state = _required_state(states, operation.schema_, operation.table)
        column_names = [column.name for column in operation.columns]
        _validate_names(operation.table, column_names)
        for row in operation.rows:
            if len(row) != len(column_names):
                raise ValueError(f"insert row for {operation.table} has wrong width")
            full_row = {column.name: None for column in state.table.columns}
            full_row.update(dict(zip(column_names, row, strict=True)))
            _check_primary_key(state, full_row)
            state.rows.append(full_row)
        return len(operation.rows)

    def _apply_update_rows(self, states: dict[tuple[str, str], TableState], operation: UpdateRowsOp) -> int:
        state = _required_state(states, operation.schema_, operation.table)
        affected_rows = 0
        for row in state.rows:
            if operation.predicate is not None and not _truthy(_eval_expr(operation.predicate, row)):
                continue
            updated = dict(row)
            for assignment in operation.assignments:
                updated[assignment.column] = _eval_expr(assignment.value, row)
            if updated != row:
                _check_primary_key(state, updated, old_row=row)
                row.update(updated)
                affected_rows += 1
        return affected_rows

    def _load_model(self, model_name_or_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        _add_special_tokens(tokenizer)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if self.training_device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map={"": self.training_device} if self.training_device.startswith("cuda") else None,
        )
        model.resize_token_embeddings(len(tokenizer))
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        processor = _load_processor(model_name_or_path, tokenizer)
        return tokenizer, model, processor

    def _next_checkpoint_path(self) -> Path:
        index = 1
        while True:
            candidate = self.checkpoint_dir / f"mutation-{index:06d}"
            if not candidate.exists():
                return candidate
            index += 1


def _load_processor(model_name_or_path: str, tokenizer: Any) -> Any | None:
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path)
    except (OSError, ValueError, KeyError):
        return None
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer
    return processor


def _save_processor(model_name_or_path: str, output_path: Path, tokenizer: Any) -> None:
    processor = _load_processor(model_name_or_path, tokenizer)
    if processor is None:
        return
    processor.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))


def _catalog_prompt() -> str:
    return "<request><catalog></catalog></request>"


def _schema_prompt(schema: str, table: str) -> str:
    return f"<request><schema>{_tag('name', schema)}{_tag('table', table)}</schema></request>"


def _select_prompt(query: SelectQuery) -> str:
    body = json.dumps(_select_payload(query), sort_keys=True, separators=(",", ":"))
    return f"<request><select>{body}</select></request>"


def _count_prompt(query: SelectQuery) -> str:
    body = json.dumps(
        {"aggregate": "count", "query": _select_payload(query)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"<request><select>{body}</select></request>"


def _select_payload(query: SelectQuery) -> dict[str, Any]:
    payload = query.model_dump(mode="json", by_alias=True)
    lookup = _primary_key_lookup_values(query)
    if lookup is not None:
        payload["primary_key_lookup"] = lookup
    elif query.predicate is None and query.limit is None and query.primary_key:
        payload["primary_key_scan"] = True
    return payload


def _catalog_completion(snapshot: CatalogSnapshot) -> str:
    parts = ["<catalog>"]
    for schema in snapshot.schemas:
        for table in schema.tables:
            parts.append(_table_xml(schema.name, table))
    parts.append("</catalog>")
    return "".join(parts)


def _schema_completion(schema: str, table: CatalogTable) -> str:
    return f"<schema>{_table_xml(schema, table)}</schema>"


def _table_xml(schema: str, table: CatalogTable) -> str:
    parts = ["<table>", _tag("schema", schema), _tag("name", table.name)]
    for column in table.columns:
        parts.append(
            "<column>"
            f"{_tag('name', column.name)}"
            f"{_tag('type', column.duckdb_type)}"
            f"{_tag('nullable', 'true' if column.nullable else 'false')}"
            "</column>"
        )
    parts.append("<primary_key>")
    for column in table.primary_key:
        parts.append(_tag("col", column))
    parts.append("</primary_key></table>")
    return "".join(parts)


def _tag(name: str, value: str) -> str:
    return f"<{name}>{_xml_escape(value)}</{name}>"


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _parse_catalog(text: str, catalog_version: str) -> CatalogSnapshot:
    root = ET.fromstring(text)
    if root.tag != "catalog":
        raise ValueError("catalog completion did not start with <catalog>")
    tables: list[CatalogTable] = []
    for table_el in root.findall("table"):
        columns = [
            CatalogColumn(
                name=_required_text(column_el, "name"),
                duckdb_type=_required_text(column_el, "type"),
                nullable=_required_text(column_el, "nullable") == "true",
            )
            for column_el in table_el.findall("column")
        ]
        primary_key = [col_el.text or "" for col_el in table_el.findall("primary_key/col")]
        tables.append(CatalogTable(name=_required_text(table_el, "name"), columns=columns, primary_key=primary_key))
    return CatalogSnapshot(catalog_version=catalog_version, schemas=[CatalogSchema(name="main", tables=tables)])


def _required_text(parent: ET.Element, child_name: str) -> str:
    child = parent.find(child_name)
    if child is None or child.text is None:
        raise ValueError(f"missing catalog tag: {child_name}")
    return child.text


def _rows_completion(rows: list[Row], projection: list[SelectColumn]) -> str:
    parts = ["<result>"]
    for row in rows:
        parts.append("<row>")
        for column in projection:
            parts.append(f"<col>{_json_cell(row.get(column.name))}</col>")
        parts.append("</row>")
    parts.append("</result>")
    return "".join(parts)


def _count_completion(row_count: int) -> str:
    return f"<result><row><col>{row_count}</col></row></result>"


def _rows_regex(width: int, *, min_rows: int = 0, max_rows: int = 64) -> str:
    cells = "".join([rf"<col>{JSON_CELL_REGEX}</col>" for _ in range(width)])
    return rf"<result>(<row>{cells}</row>){{{min_rows},{max_rows}}}</result>"


def _rows_regex_for_projection(
    projection: list[SelectColumn],
    *,
    min_rows: int = 0,
    max_rows: int = 64,
) -> str:
    cells = "".join([rf"<col>{_json_cell_regex_for_type(column.duckdb_type)}</col>" for column in projection])
    return rf"<result>(<row>{cells}</row>){{{min_rows},{max_rows}}}</result>"


def _json_cell_regex_for_type(duckdb_type: str) -> str:
    normalized = duckdb_type.upper()
    if normalized in {"VARCHAR", "TEXT"}:
        return JSON_STRING_REGEX
    if normalized in {"BOOLEAN", "BOOL"}:
        return BOOL_REGEX
    if normalized in {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"}:
        return JSON_INTEGER_REGEX
    if normalized in {"FLOAT", "DOUBLE", "REAL", "DECIMAL"}:
        return JSON_NUMBER_REGEX
    return JSON_CELL_REGEX


def _parse_rows(text: str, width: int) -> list[list[JsonScalar]]:
    if not text.startswith("<result>") or not text.endswith("</result>"):
        raise ValueError(f"row completion did not use <result> tags: {text!r}")
    rows: list[list[JsonScalar]] = []
    for row_match in re.finditer(r"<row>(.*?)</row>", text):
        cells = re.findall(r"<col>(.*?)</col>", row_match.group(1))
        if len(cells) != width:
            raise ValueError(f"row completion has width {len(cells)}, expected {width}: {text!r}")
        rows.append([_parse_json_cell(cell) for cell in cells])
    return rows


def _parse_rows_for_projection(text: str, projection: list[SelectColumn]) -> list[list[JsonScalar]]:
    if not text.startswith("<result>") or not text.endswith("</result>"):
        raise ValueError(f"row completion did not use <result> tags: {text!r}")
    rows: list[list[JsonScalar]] = []
    for row_match in re.finditer(r"<row>(.*?)</row>", text):
        cells = re.findall(r"<col>(.*?)</col>", row_match.group(1))
        if len(cells) != len(projection):
            raise ValueError(f"row completion has width {len(cells)}, expected {len(projection)}: {text!r}")
        rows.append(
            [
                _parse_json_cell_for_type(cell, column.duckdb_type)
                for cell, column in zip(cells, projection, strict=True)
            ]
        )
    return rows


def _parse_json_cell(cell: str) -> JsonScalar:
    try:
        return json.loads(cell)
    except json.JSONDecodeError:
        if cell.startswith('"') and not cell.endswith('"'):
            return json.loads(f"{cell}\"")
        raise


def _parse_json_cell_for_type(cell: str, duckdb_type: str) -> JsonScalar:
    try:
        return json.loads(cell)
    except json.JSONDecodeError:
        if duckdb_type.upper() in {"VARCHAR", "TEXT"} and cell.startswith('"'):
            repaired = cell if cell.endswith('"') else f"{cell}\""
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return cell[1:].rstrip('"')
        raise


def _json_cell(value: JsonScalar) -> str:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _rows_match_projection_types(rows: list[list[JsonScalar]], projection: list[SelectColumn]) -> bool:
    return all(
        len(row) == len(projection)
        and all(_value_matches_duckdb_type(value, column.duckdb_type) for value, column in zip(row, projection, strict=True))
        for row in rows
    )


def _value_matches_duckdb_type(value: JsonScalar, duckdb_type: str) -> bool:
    if value is None:
        return True
    normalized = duckdb_type.upper()
    if normalized in {"VARCHAR", "TEXT"}:
        return isinstance(value, str)
    if normalized in {"BOOLEAN", "BOOL"}:
        return isinstance(value, bool)
    if normalized in {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if normalized in {"FLOAT", "DOUBLE", "REAL", "DECIMAL"}:
        return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
    return True


def _insert_columns_from_catalog(table: CatalogTable) -> list[InsertColumn]:
    return [
        InsertColumn(name=column.name, duckdb_type=column.duckdb_type, nullable=column.nullable)
        for column in table.columns
    ]


def _query_with_primary_key_projection(query: SelectQuery) -> SelectQuery:
    projected_names = {column.name for column in query.projection}
    by_name = {column.name: column for column in query.columns}
    hidden_key_projection = [
        SelectColumn(name=column_name, duckdb_type=by_name[column_name].duckdb_type)
        for column_name in query.primary_key
        if column_name not in projected_names and column_name in by_name
    ]
    if not hidden_key_projection:
        return query
    return query.model_copy(update={"projection": [*hidden_key_projection, *query.projection]})


def _can_sample_by_primary_key(query: SelectQuery) -> bool:
    return query.predicate is None and query.limit is None and bool(_primary_key_projection(query))


def _primary_key_projection(query: SelectQuery) -> list[SelectColumn]:
    column_types = {column.name: column.duckdb_type for column in query.projection}
    column_types.update({column.name: column.duckdb_type for column in query.columns})
    return [
        SelectColumn(name=column_name, duckdb_type=column_types[column_name])
        for column_name in query.primary_key
        if column_name in column_types
    ]


def _primary_key_predicate(key_projection: list[SelectColumn], key_row: list[JsonScalar]) -> Predicate:
    comparisons: list[Predicate] = [
        ComparisonPredicate(
            kind="comparison",
            op="=",
            left=ColumnPredicate(kind="column", name=column.name, duckdb_type=column.duckdb_type),
            right=LiteralPredicate(kind="literal", value=value, duckdb_type=column.duckdb_type),
        )
        for column, value in zip(key_projection, key_row, strict=True)
    ]
    if len(comparisons) == 1:
        return comparisons[0]
    return BooleanPredicate(kind="and", children=comparisons)


def _primary_key_lookup_values(query: SelectQuery) -> dict[str, JsonScalar] | None:
    if not query.primary_key or query.predicate is None:
        return None
    values: dict[str, JsonScalar] = {}
    if not _collect_primary_key_lookup_values(query.predicate, set(query.primary_key), values):
        return None
    if set(values) != set(query.primary_key):
        return None
    return {column: values[column] for column in query.primary_key}


def _collect_primary_key_lookup_values(
    predicate: Predicate,
    primary_key: set[str],
    values: dict[str, JsonScalar],
) -> bool:
    if isinstance(predicate, BooleanPredicate) and predicate.kind == "and":
        return all(_collect_primary_key_lookup_values(child, primary_key, values) for child in predicate.children)
    if not isinstance(predicate, ComparisonPredicate) or predicate.op not in {"=", "=="}:
        return False
    pair = _column_literal_pair(predicate.left, predicate.right)
    if pair is None:
        pair = _column_literal_pair(predicate.right, predicate.left)
    if pair is None:
        return False
    column, value = pair
    if column not in primary_key:
        return False
    values[column] = value
    return True


def _column_literal_pair(left: Predicate, right: Predicate) -> tuple[str, JsonScalar] | None:
    if isinstance(left, ColumnPredicate) and isinstance(right, LiteralPredicate):
        return left.name, right.value
    return None


def _project_rows(
    rows: list[list[JsonScalar]],
    sampling_query: SelectQuery,
    output_query: SelectQuery,
) -> list[list[JsonScalar]]:
    if sampling_query.projection == output_query.projection:
        return rows
    source_indexes: dict[str, int] = {}
    for index, column in enumerate(sampling_query.projection):
        source_indexes.setdefault(column.name, index)
    output_indexes = [source_indexes[column.name] for column in output_query.projection]
    return [[row[index] for index in output_indexes] for row in rows]


def _primary_key_indexes(query: SelectQuery) -> list[int]:
    by_name = {column.name: index for index, column in enumerate(query.projection)}
    return [by_name[column] for column in query.primary_key if column in by_name]


def _primary_keys_expand(
    rows: list[list[JsonScalar]],
    candidate: list[list[JsonScalar]],
    key_indexes: list[int],
) -> bool:
    if len(candidate) != len(rows) + 1:
        return False
    current_keys = {_row_key(row, key_indexes) for row in rows}
    candidate_keys = [_row_key(row, key_indexes) for row in candidate]
    return len(set(candidate_keys)) == len(candidate_keys) and current_keys.issubset(candidate_keys)


def _row_key(row: list[JsonScalar], key_indexes: list[int]) -> tuple[JsonScalar, ...]:
    return tuple(row[index] for index in key_indexes)


def _merge_key_rows(
    key_rows: list[list[JsonScalar]],
    direct_rows: list[list[JsonScalar]],
    key_indexes: list[int],
) -> list[list[JsonScalar]]:
    merged: list[list[JsonScalar]] = []
    seen: set[tuple[JsonScalar, ...]] = set()
    for row in direct_rows:
        key = _row_key(row, key_indexes)
        if key in seen:
            continue
        seen.add(key)
        merged.append(list(key))
    for row in key_rows:
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def _synthetic_queries(state: TableState) -> list[SelectQuery]:
    queries: list[SelectQuery] = []
    projection = [SelectColumn(name=column.name, duckdb_type=column.duckdb_type) for column in state.table.columns]
    by_name = {column.name: column for column in state.table.columns}
    table_columns = _insert_columns_from_catalog(state.table)

    def query(predicate: Predicate) -> SelectQuery:
        return SelectQuery(
            schema=state.schema,
            table=state.table.name,
            columns=table_columns,
            primary_key=state.table.primary_key,
            projection=projection,
            predicate=predicate,
        )

    if state.table.primary_key and projection:
        pk = state.table.primary_key[0]
        pk_type = by_name[pk].duckdb_type
        for row in state.rows[:4]:
            queries.append(
                query(
                    ComparisonPredicate(
                        kind="comparison",
                        op="=",
                        left=ColumnPredicate(kind="column", name=pk, duckdb_type=pk_type),
                        right=LiteralPredicate(kind="literal", value=row[pk], duckdb_type=pk_type),
                    ),
                )
            )
            if isinstance(row.get(pk), str) and row[pk]:
                queries.append(
                    query(
                        FunctionPredicate(
                            kind="function",
                            name="starts_with",
                            duckdb_type="BOOLEAN",
                            args=[
                                ColumnPredicate(kind="column", name=pk, duckdb_type=pk_type),
                                LiteralPredicate(kind="literal", value=str(row[pk])[:2], duckdb_type=pk_type),
                            ],
                        ),
                    )
                )
    for column in state.table.columns:
        if column.duckdb_type not in {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "FLOAT", "DOUBLE"}:
            continue
        values = [row[column.name] for row in state.rows if isinstance(row.get(column.name), int | float)]
        if not values:
            continue
        threshold = min(values)
        queries.append(
            query(
                ComparisonPredicate(
                    kind="comparison",
                    op=">",
                    left=ColumnPredicate(kind="column", name=column.name, duckdb_type=column.duckdb_type),
                    right=LiteralPredicate(kind="literal", value=threshold, duckdb_type=column.duckdb_type),
                ),
            )
        )
    return queries


def _primary_key_lookup_queries(
    state: TableState,
    full_projection: list[SelectColumn],
) -> list[tuple[SelectQuery, list[Row]]]:
    key_projection = [
        column for key_name in state.table.primary_key for column in full_projection if column.name == key_name
    ]
    if not key_projection:
        return []
    table_columns = _insert_columns_from_catalog(state.table)
    queries: list[tuple[SelectQuery, list[Row]]] = []
    for row in state.rows[:64]:
        key_row = [row[column.name] for column in key_projection]
        predicate = _primary_key_predicate(key_projection, key_row)
        queries.append(
            (
                SelectQuery(
                    schema=state.schema,
                    table=state.table.name,
                    columns=table_columns,
                    primary_key=state.table.primary_key,
                    projection=full_projection,
                    predicate=predicate,
                ),
                [row],
            )
        )
        for column in full_projection:
            if column.name in state.table.primary_key:
                continue
            queries.append(
                (
                    SelectQuery(
                        schema=state.schema,
                        table=state.table.name,
                        columns=table_columns,
                        primary_key=state.table.primary_key,
                        projection=[*key_projection, column],
                        predicate=predicate,
                    ),
                    [row],
                )
            )
    return queries


def _states_from_operation_shapes(operations: list[Any]) -> dict[tuple[str, str], TableState]:
    states: dict[tuple[str, str], TableState] = {}
    for operation in operations:
        if isinstance(operation, CreateTableOp):
            columns = [
                CatalogColumn(name=column.name, duckdb_type=column.duckdb_type, nullable=column.nullable)
                for column in operation.columns
            ]
            _seed_table_state(states, operation.schema_, operation.table, columns, operation.primary_key)
        elif isinstance(operation, InsertRowsOp):
            columns = _mutation_columns_to_catalog(operation.columns, operation.primary_key)
            _seed_table_state(states, operation.schema_, operation.table, columns, operation.primary_key)
        elif isinstance(operation, UpdateRowsOp):
            columns = (
                _mutation_columns_to_catalog(operation.columns, operation.primary_key)
                if operation.columns
                else _columns_from_update_operation(operation)
            )
            _seed_table_state(states, operation.schema_, operation.table, columns, operation.primary_key)
    return states


def _seed_table_state(
    states: dict[tuple[str, str], TableState],
    schema: str,
    table: str,
    columns: list[CatalogColumn],
    primary_key: list[str],
) -> None:
    key = (schema, table)
    if key in states:
        return
    if not columns:
        raise ValueError(f"cannot infer table shape for mutation on {schema}.{table}")
    _validate_names(table, [column.name for column in columns])
    _validate_names(table, primary_key)
    states[key] = TableState(
        schema=schema,
        table=CatalogTable(name=table, columns=columns, primary_key=primary_key),
        rows=[],
    )


def _mutation_columns_to_catalog(columns: list[Any], primary_key: list[str]) -> list[CatalogColumn]:
    return [
        CatalogColumn(
            name=column.name,
            duckdb_type=column.duckdb_type,
            nullable=bool(getattr(column, "nullable", True)) and column.name not in primary_key,
        )
        for column in columns
    ]


def _columns_from_update_operation(operation: UpdateRowsOp) -> list[CatalogColumn]:
    by_name: dict[str, CatalogColumn] = {}

    def add(name: str, duckdb_type: str) -> None:
        by_name.setdefault(name, CatalogColumn(name=name, duckdb_type=duckdb_type, nullable=name not in operation.primary_key))

    for assignment in operation.assignments:
        add(assignment.column, assignment.duckdb_type)
        _collect_predicate_columns(assignment.value, by_name, operation.primary_key)
    if operation.predicate is not None:
        _collect_predicate_columns(operation.predicate, by_name, operation.primary_key)
    return list(by_name.values())


def _collect_predicate_columns(
    expr: Predicate,
    by_name: dict[str, CatalogColumn],
    primary_key: list[str],
) -> None:
    if isinstance(expr, ColumnPredicate):
        by_name.setdefault(
            expr.name,
            CatalogColumn(name=expr.name, duckdb_type=expr.duckdb_type, nullable=expr.name not in primary_key),
        )
        return
    if isinstance(expr, LiteralPredicate):
        return
    if isinstance(expr, ComparisonPredicate):
        _collect_predicate_columns(expr.left, by_name, primary_key)
        _collect_predicate_columns(expr.right, by_name, primary_key)
        return
    if isinstance(expr, ArithmeticPredicate):
        for arg in expr.args:
            _collect_predicate_columns(arg, by_name, primary_key)
        return
    if isinstance(expr, FunctionPredicate):
        for arg in expr.args:
            _collect_predicate_columns(arg, by_name, primary_key)
        return
    if isinstance(expr, BooleanPredicate):
        for child in expr.children:
            _collect_predicate_columns(child, by_name, primary_key)
        return
    if isinstance(expr, NullPredicate):
        _collect_predicate_columns(expr.expr, by_name, primary_key)


def _filter_rows(rows: list[Row], predicate: Predicate | None) -> list[Row]:
    if predicate is None:
        return rows
    return [row for row in rows if _truthy(_eval_expr(predicate, row))]


def _rows_for_query(rows: list[Row], query: SelectQuery) -> list[Row]:
    filtered = _filter_rows(rows, query.predicate)
    if query.limit is None:
        return filtered
    return filtered[: query.limit]


def _eval_expr(expr: Predicate, row: Row) -> JsonScalar:
    if isinstance(expr, ColumnPredicate):
        return row.get(expr.name)
    if isinstance(expr, LiteralPredicate):
        return expr.value
    if isinstance(expr, ComparisonPredicate):
        left = _eval_expr(expr.left, row)
        right = _eval_expr(expr.right, row)
        return _compare(expr.op, left, right)
    if isinstance(expr, ArithmeticPredicate):
        values = [_eval_expr(arg, row) for arg in expr.args]
        return _arithmetic(expr.op, values)
    if isinstance(expr, BooleanPredicate):
        values = [_truthy(_eval_expr(child, row)) for child in expr.children]
        return all(values) if expr.kind == "and" else any(values)
    if isinstance(expr, NullPredicate):
        value = _eval_expr(expr.expr, row)
        return value is None if expr.kind == "is_null" else value is not None
    if isinstance(expr, FunctionPredicate):
        values = [_eval_expr(arg, row) for arg in expr.args]
        return _function(expr.name, values)
    raise NotImplementedError(f"unsupported expression: {expr!r}")


def _compare(op: str, left: JsonScalar, right: JsonScalar) -> bool:
    if left is None or right is None:
        return False
    if op in {"=", "=="}:
        return left == right
    if op in {"!=", "<>"}:
        return left != right
    if op == "<":
        return left < right  # type: ignore[operator]
    if op == "<=":
        return left <= right  # type: ignore[operator]
    if op == ">":
        return left > right  # type: ignore[operator]
    if op == ">=":
        return left >= right  # type: ignore[operator]
    raise NotImplementedError(f"unsupported comparison operator: {op}")


def _arithmetic(op: str, values: list[JsonScalar]) -> JsonScalar:
    if any(value is None for value in values):
        return None
    if op == "+":
        return values[0] + values[1]  # type: ignore[operator]
    if op == "-":
        return -values[0] if len(values) == 1 else values[0] - values[1]  # type: ignore[operator]
    if op == "*":
        return values[0] * values[1]  # type: ignore[operator]
    if op == "/":
        return values[0] / values[1]  # type: ignore[operator]
    if op == "//":
        return values[0] // values[1]  # type: ignore[operator]
    if op == "%":
        return values[0] % values[1]  # type: ignore[operator]
    raise NotImplementedError(f"unsupported arithmetic operator: {op}")


def _function(name: str, values: list[JsonScalar]) -> bool:
    normalized = name.lower()
    if normalized in {"starts_with", "prefix"}:
        return str(values[0]).startswith(str(values[1]))
    if normalized in {"contains", "contains_substr"}:
        return str(values[1]) in str(values[0])
    if normalized in {"like", "~~"}:
        return _like(str(values[0]), str(values[1]))
    raise NotImplementedError(f"unsupported function predicate: {name}")


def _like(value: str, pattern: str) -> bool:
    regex = "^" + re.escape(pattern).replace("%", ".*").replace("_", ".") + "$"
    return re.match(regex, value) is not None


def _truthy(value: JsonScalar) -> bool:
    return bool(value)


def _snapshot_from_states(states: dict[tuple[str, str], TableState], catalog_version: str) -> CatalogSnapshot:
    by_schema: dict[str, list[CatalogTable]] = {}
    for (schema, _), state in states.items():
        by_schema.setdefault(schema, []).append(state.table)
    return CatalogSnapshot(
        catalog_version=catalog_version,
        schemas=[
            CatalogSchema(name=schema, tables=sorted(tables, key=lambda table: table.name))
            for schema, tables in sorted(by_schema.items())
        ]
        or [CatalogSchema(name="main", tables=[])],
    )


def _empty_catalog(catalog_version: str) -> CatalogSnapshot:
    return CatalogSnapshot(catalog_version=catalog_version, schemas=[CatalogSchema(name="main", tables=[])])


def _required_state(states: dict[tuple[str, str], TableState], schema: str, table: str) -> TableState:
    try:
        return states[(schema, table)]
    except KeyError as exc:
        raise ValueError(f"unknown table: {schema}.{table}") from exc


def _check_primary_key(state: TableState, row: Row, *, old_row: Row | None = None) -> None:
    if not state.table.primary_key:
        return
    key = tuple(row[column] for column in state.table.primary_key)
    old_key = tuple(old_row[column] for column in state.table.primary_key) if old_row else None
    if old_key == key:
        return
    for existing in state.rows:
        if existing is old_row:
            continue
        if tuple(existing[column] for column in state.table.primary_key) == key:
            raise ValueError(f"duplicate primary key for {state.table.name}: {key}")


def _validate_names(table: str, columns: list[str]) -> None:
    for name in [table, *columns]:
        if not re.fullmatch(NAME_REGEX, name):
            raise ValueError(f"v0 tagged-rows scheme only supports simple identifiers, got {name!r}")


def _add_special_tokens(tokenizer: Any) -> None:
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})


def _checkpoint_ref(model_name_or_path: str) -> str:
    path = Path(model_name_or_path)
    if path.exists():
        return path.name
    return model_name_or_path


def _max_select_tokens(query: SelectQuery) -> int:
    row_budget = query.limit if query.limit is not None else 64
    return int(min(4096, 32 + max(1, row_budget) * max(1, len(query.projection)) * 24))


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


__all__ = ["SPECIAL_TOKENS", "TaggedRowsSFTDatabase"]
