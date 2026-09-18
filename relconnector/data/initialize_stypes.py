"""Freeze RelBench's sampled stype proposal without materializing full features."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from torch_frame import stype
from torch_frame.utils import infer_df_stype

from relbench_compat.proposal import (
    ARTIFACT_VERSION,
    STYPE_SEED,
    artifact_path,
    database_identity,
    dependencies,
    digest,
)

from .decoding import decode_frame
from .models import ColumnSchema
from .writers import quote_identifier


def initialize_database(path: str | Path, *, seed: int = STYPE_SEED) -> Path:
    """Match get_stype_proposal on sorted tables of dataset.get_db()'s view.

    RandomState.choice matches pandas DataFrame.sample with a seeded global RNG.
    Only <=1000 feature rows/table are read; the ID permutation uses O(N) memory.
    No data or task tables are changed. The artifact is invalidated by DB changes.
    """
    path = Path(path).resolve()
    before = database_identity(path)
    rng = np.random.RandomState(seed)
    proposals: dict[str, dict[str, str]] = {}
    sampled_ids: dict[str, list[int]] = {}
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
        meta = dict(connection.execute("SELECT key,value FROM _relconnector_metadata"))
        cutoff = (
            pd.Timestamp(meta["test_timestamp"]) if meta.get("test_timestamp") else None
        )
        tables = connection.execute(
            "SELECT table_name,primary_key,time_column FROM _relconnector_tables "
            "WHERE table_kind='data' ORDER BY table_name"
        ).fetchall()
        for table, primary, time_column in tables:
            identifier = quote_identifier(primary) if primary else "rowid"
            where = ""
            if cutoff is not None and time_column:
                value = cutoff.isoformat().replace("'", "''")
                where = f" WHERE julianday({quote_identifier(time_column)}) <= julianday('{value}')"
            count, minimum, maximum = connection.execute(
                f"SELECT count(*),min({identifier}),max({identifier}) "
                f"FROM {quote_identifier(table)}{where}"
            ).fetchone()
            start = 0 if primary else 1
            if count and (minimum != start or maximum != start + count - 1):
                raise ValueError(
                    f"{table}: official sampling requires dense zero-based IDs after cutoff"
                )
            ids = rng.choice(count, size=min(1000, count), replace=False).tolist()
            sampled_ids[table] = ids
            columns = tuple(
                ColumnSchema(name, ordinal, dtype, encoding, json.loads(extra))
                for ordinal, name, dtype, encoding, extra in connection.execute(
                    "SELECT ordinal_position,column_name,pandas_dtype,encoding,encoding_metadata "
                    "FROM _relconnector_columns WHERE table_name=? ORDER BY ordinal_position",
                    (table,),
                )
            )
            values = ",".join(str(i + start) for i in ids) or "NULL"
            frame = (
                pd.read_sql_query(
                    f"SELECT {identifier} AS __sample_key__, * FROM {quote_identifier(table)} "
                    f"WHERE {identifier} IN ({values})",
                    connection,
                    dtype_backend="numpy_nullable",
                )
                .set_index("__sample_key__")
                .reindex([i + start for i in ids])
            )
            frame = decode_frame(frame, columns)
            # This synthetic key did not exist in the source RelBench database.
            frame = frame.drop(columns=["__relconnector_node_id__"], errors="ignore")
            inferred = infer_df_stype(frame)
            proposals[table] = {
                name: (
                    stype.multicategorical if kind == stype.embedding else kind
                ).value
                for name, kind in inferred.items()
            }
    if database_identity(path) != before:
        raise RuntimeError(f"Database changed while generating stypes: {path}")
    payload: dict[str, object] = {
        "version": ARTIFACT_VERSION,
        "database": before,
        "dependencies": dependencies(),
        "seed": seed,
        "cutoff": None if cutoff is None else str(cutoff),
        "table_order": list(proposals),
        "sample_ids": sampled_ids,
        "sampling": "pandas-sample-randomstate-without-replacement-1000",
        "tables": proposals,
    }
    payload["fingerprint"] = digest(payload)
    target = artifact_path(path)
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, action="append")
    parser.add_argument(
        "--database-dir", type=Path, default=Path(__file__).parent / "relbench"
    )
    parser.add_argument("--seed", type=int, default=STYPE_SEED)
    args = parser.parse_args()
    for path in args.database or sorted(args.database_dir.glob("*.sqlite")):
        print(initialize_database(path, seed=args.seed), flush=True)


if __name__ == "__main__":
    main()
