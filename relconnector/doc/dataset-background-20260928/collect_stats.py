"""Read-only SQLite statistics, checked against cached official Parquet files."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HUB = Path.home() / ".cache/huggingface/hub"
NODE_ID = "__relconnector_node_id__"


def quote(value):
    return '"' + value.replace('"', '""') + '"'


def collect(path):
    before = path.stat()
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    metadata = dict(conn.execute("SELECT key,value FROM _relconnector_metadata"))
    artifact = json.loads(path.with_suffix(".stypes.json").read_text())
    assert artifact["database"]["bytes"] == before.st_size
    assert artifact["database"]["mtime_ns"] == before.st_mtime_ns
    cutoff = metadata["test_timestamp"]
    spec = metadata["source_spec"].split("/")
    snapshots = HUB / ("datasets--" + "--".join(spec[:2])) / "snapshots"
    candidates = list(snapshots.glob(f"*/{path.stem}/db"))
    assert len(candidates) == 1, candidates
    db_dir = candidates[0]
    result = dict(dataset=path.stem, path=str(path), sqlite_bytes=before.st_size,
                  metadata=metadata, snapshot=db_dir.parent.parent.name,
                  source_repo="/".join(spec[:2]), cutoff=cutoff,
                  stype_dependencies=artifact["dependencies"], tables=[], tasks=[])
    catalog = conn.execute("SELECT table_name,table_kind,time_column FROM _relconnector_tables ORDER BY table_name").fetchall()
    for name, kind, time_col in catalog:
        schema = conn.execute(f"PRAGMA table_info({quote(name)})").fetchall()
        rows = conn.execute(f"SELECT COUNT(*) FROM {quote(name)}").fetchone()[0]
        if kind != "data":
            result["tasks"].append(dict(table=name, rows=rows, columns=len(schema)))
            continue
        original = [c for c in schema if c[1] != NODE_ID]
        stypes = artifact["tables"][name]
        assert set(stypes) <= {c[1] for c in original}
        untyped = [c[1] for c in original if c[1] not in stypes]
        text = [c[1] for c in original if stypes.get(c[1]) == "text_embedded"]
        sql_text = [c[1] for c in original if c[2].upper() == "TEXT"]
        if time_col:
            # Datetimes are persisted as uniform ISO strings; use the same inclusive cutoff as Table.upto.
            bounds = conn.execute(f"SELECT MIN({quote(time_col)}),MAX({quote(time_col)}) FROM {quote(name)}").fetchone()
            assert all(v is None or (len(v) >= 19 and v[10] == "T") for v in bounds), bounds
            cutoff_rows = conn.execute(f"SELECT COUNT(*) FROM {quote(name)} WHERE {quote(time_col)} <= ?", (cutoff,)).fetchone()[0]
        else:
            bounds = None
            cutoff_rows = rows
        parquet = db_dir / f"{name}.parquet"
        pf = pq.ParquetFile(parquet)
        names = pf.schema_arrow.names
        assert set(names) == {c[1] for c in original}, (name, names, original)
        assert pf.metadata.num_rows == rows, (name, pf.metadata.num_rows, rows)
        table = dict(table=name, rows=rows, cutoff_rows=cutoff_rows,
                     columns=len(original), sqlite_columns=len(schema),
                     text_columns=text, sql_text_columns=sql_text,
                     stypes=stypes, untyped_columns=untyped,
                     untyped_nonnull={col: conn.execute(f"SELECT COUNT({quote(col)}) FROM {quote(name)}").fetchone()[0] for col in untyped},
                     time_column=time_col, time_bounds=bounds,
                     parquet_bytes=parquet.stat().st_size,
                     parquet_rows=pf.metadata.num_rows, parquet_columns=len(names))
        result["tables"].append(table)
        print(f"{path.stem}/{name}: {rows:,} rows, {len(original)} columns", flush=True)
    conn.close()
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    for key in ("rows", "cutoff_rows", "columns", "sqlite_columns", "parquet_bytes"):
        result[key] = sum(t[key] for t in result["tables"])
    result["text_count"] = sum(len(t["text_columns"]) for t in result["tables"])
    result["sql_text_count"] = sum(len(t["sql_text_columns"]) for t in result["tables"])
    (OUT / f"{path.stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    datasets = [collect(p) for p in sorted((ROOT / "data/relbench").glob("*.sqlite"))]
    assert len(datasets) == 10
    result = dict(collected_at=datetime.now(timezone.utc).isoformat(), datasets=datasets)
    (OUT / "local-stats.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for d in datasets:
        print(d["dataset"], {k: d[k] for k in ("rows", "cutoff_rows", "columns", "text_count", "sqlite_bytes", "parquet_bytes")})
