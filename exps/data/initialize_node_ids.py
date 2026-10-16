"""Inspect or migrate RelConnector SQLite tables to explicit dense node IDs."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .models import COLUMNS_TABLE, METADATA_TABLE, NODE_ID_COLUMN, TABLES_TABLE
from .writers import quote_identifier


@dataclass(frozen=True)
class TableMigration:
    table: str
    rows: int
    query_plan: tuple[str, ...]


@dataclass(frozen=True)
class MigrationReport:
    database: str
    applied: bool
    tables: tuple[TableMigration, ...]


def initialize_database(
    path: str | Path, *, apply: bool = False, vacuum: bool = False
) -> MigrationReport:
    """Add an ``INTEGER PRIMARY KEY`` to data tables that rely on hidden rowid.

    Existing logical node IDs are preserved exactly: migration is rejected unless
    hidden rowids are the contiguous sequence ``1..N`` used by the old runtime.
    """
    database = Path(path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        _validate_catalog(connection)
        tables = _candidate_tables(connection)
        validated = tuple(_validate_table(connection, table) for table in tables)
        if not apply:
            return MigrationReport(str(database), False, validated)

        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("BEGIN IMMEDIATE")
        try:
            for table in tables:
                _migrate_table(connection, table)
            _record_contract(connection)
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise RuntimeError(
                    f"Migration produced {len(violations)} foreign-key violations; "
                    f"first entries: {violations[:10]}"
                )
            migrated = tuple(
                _validate_explicit_table(connection, table) for table in tables
            )
            connection.commit()
            if vacuum:
                connection.execute("VACUUM")
        except BaseException:
            connection.rollback()
            raise
        return MigrationReport(str(database), True, migrated)
    finally:
        connection.close()


def _validate_catalog(connection: sqlite3.Connection) -> None:
    names = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table'"
        )
    }
    required = {TABLES_TABLE, COLUMNS_TABLE, METADATA_TABLE}
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"Not a RelConnector database; missing catalog: {missing}")


def _candidate_tables(connection: sqlite3.Connection) -> tuple[str, ...]:
    rows = connection.execute(
        f"SELECT table_name FROM {quote_identifier(TABLES_TABLE)} "
        "WHERE table_kind = 'data' AND primary_key IS NULL ORDER BY table_name"
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _validate_table(connection: sqlite3.Connection, table: str) -> TableMigration:
    columns = _table_info(connection, table)
    if any(str(column["name"]) == NODE_ID_COLUMN for column in columns):
        raise ValueError(
            f"{table!r} already contains reserved column {NODE_ID_COLUMN!r}"
        )
    if any(int(column["pk"]) for column in columns):
        raise ValueError(f"Catalog says {table!r} has no key, but SQLite reports one")
    quoted = quote_identifier(table)
    count, minimum, maximum = connection.execute(
        f"SELECT COUNT(*), MIN(rowid), MAX(rowid) FROM {quoted}"
    ).fetchone()
    count = int(count)
    if count and (int(minimum) != 1 or int(maximum) != count):
        raise ValueError(
            f"{table!r} rowid is not the dense 1..N sequence required to preserve "
            "existing sampler node IDs"
        )
    plan = _query_plan(connection, table, explicit=False)
    if count and not any("SEARCH" in detail for detail in plan):
        # This is the bug the migration addresses, but retain it in dry-run output.
        pass
    return TableMigration(table, count, plan)


def _validate_explicit_table(
    connection: sqlite3.Connection, table: str
) -> TableMigration:
    quoted = quote_identifier(table)
    node_id = quote_identifier(NODE_ID_COLUMN)
    count, minimum, maximum = connection.execute(
        f"SELECT COUNT(*), MIN({node_id}), MAX({node_id}) FROM {quoted}"
    ).fetchone()
    count = int(count)
    if count and (int(minimum) != 0 or int(maximum) != count - 1):
        raise RuntimeError(f"{table!r} explicit node IDs are not dense zero-based")
    plan = _query_plan(connection, table, explicit=True)
    if count and not any("SEARCH" in detail for detail in plan):
        raise RuntimeError(f"{table!r} node-ID lookup is not indexed: {plan}")
    return TableMigration(table, count, plan)


def _query_plan(
    connection: sqlite3.Connection, table: str, *, explicit: bool
) -> tuple[str, ...]:
    identifier = quote_identifier(NODE_ID_COLUMN) if explicit else "rowid - 1"
    rows = connection.execute(
        f"EXPLAIN QUERY PLAN SELECT * FROM {quote_identifier(table)} "
        f"WHERE {identifier} IN (0, 1) ORDER BY {identifier}"
    ).fetchall()
    return tuple(str(row[3]) for row in rows)


def _migrate_table(connection: sqlite3.Connection, table: str) -> None:
    columns = _table_info(connection, table)
    foreign_keys = connection.execute(
        f"PRAGMA foreign_key_list({quote_identifier(table)})"
    ).fetchall()
    indexes = tuple(
        str(row[0])
        for row in connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type = 'index' "
            "AND tbl_name = ? AND sql IS NOT NULL ORDER BY name",
            (table,),
        )
    )
    triggers = tuple(
        str(row[0])
        for row in connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type = 'trigger' "
            "AND tbl_name = ? ORDER BY name",
            (table,),
        )
    )
    temporary = f"__relconnector_migrate_{table}"
    if connection.execute(
        "SELECT 1 FROM sqlite_schema WHERE name = ?", (temporary,)
    ).fetchone():
        raise RuntimeError(f"Temporary migration table already exists: {temporary!r}")

    definitions = [f"{quote_identifier(NODE_ID_COLUMN)} INTEGER PRIMARY KEY"]
    definitions.extend(_column_definition(column) for column in columns)
    definitions.extend(_foreign_key_definitions(foreign_keys))
    connection.execute(
        f"CREATE TABLE {quote_identifier(temporary)} ({', '.join(definitions)})"
    )
    names = [str(column["name"]) for column in columns]
    destination = ", ".join(
        [quote_identifier(NODE_ID_COLUMN), *(quote_identifier(name) for name in names)]
    )
    source = ", ".join(["rowid - 1", *(quote_identifier(name) for name in names)])
    connection.execute(
        f"INSERT INTO {quote_identifier(temporary)} ({destination}) "
        f"SELECT {source} FROM {quote_identifier(table)} ORDER BY rowid"
    )
    connection.execute(f"DROP TABLE {quote_identifier(table)}")
    connection.execute(
        f"ALTER TABLE {quote_identifier(temporary)} RENAME TO {quote_identifier(table)}"
    )
    for statement in indexes + triggers:
        connection.execute(statement)

    connection.execute(
        f"UPDATE {quote_identifier(COLUMNS_TABLE)} "
        "SET ordinal_position = ordinal_position + 1000000 WHERE table_name = ?",
        (table,),
    )
    connection.execute(
        f"UPDATE {quote_identifier(COLUMNS_TABLE)} "
        "SET ordinal_position = ordinal_position - 999999 WHERE table_name = ?",
        (table,),
    )
    connection.execute(
        f"INSERT INTO {quote_identifier(COLUMNS_TABLE)} VALUES (?, 0, ?, ?, ?, ?)",
        (table, NODE_ID_COLUMN, "int64", "scalar", "{}"),
    )
    connection.execute(
        f"UPDATE {quote_identifier(TABLES_TABLE)} SET primary_key = ? "
        "WHERE table_name = ?",
        (NODE_ID_COLUMN, table),
    )


def _table_info(connection: sqlite3.Connection, table: str) -> tuple[sqlite3.Row, ...]:
    return tuple(
        connection.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    )


def _column_definition(column: sqlite3.Row) -> str:
    output = quote_identifier(str(column["name"]))
    if column["type"]:
        output += f" {column['type']}"
    if int(column["notnull"]):
        output += " NOT NULL"
    if column["dflt_value"] is not None:
        output += f" DEFAULT {column['dflt_value']}"
    return output


def _foreign_key_definitions(
    rows: tuple[sqlite3.Row, ...] | list[sqlite3.Row],
) -> list[str]:
    groups: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        groups.setdefault(int(row["id"]), []).append(row)
    definitions: list[str] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: int(row["seq"]))
        local = ", ".join(quote_identifier(str(row["from"])) for row in ordered)
        remote = ", ".join(quote_identifier(str(row["to"])) for row in ordered)
        first = ordered[0]
        definition = (
            f"FOREIGN KEY ({local}) REFERENCES {quote_identifier(str(first['table']))}"
            f"({remote})"
        )
        if str(first["on_update"]) != "NO ACTION":
            definition += f" ON UPDATE {first['on_update']}"
        if str(first["on_delete"]) != "NO ACTION":
            definition += f" ON DELETE {first['on_delete']}"
        definitions.append(definition)
    return definitions


def _record_contract(connection: sqlite3.Connection) -> None:
    values = {
        "node_id_contract": "explicit-dense-zero-based-v1",
        "node_id_initialized_at": datetime.now(timezone.utc).isoformat(),
    }
    connection.executemany(
        f"INSERT INTO {quote_identifier(METADATA_TABLE)} (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        values.items(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, nargs="*")
    parser.add_argument(
        "--database-dir", type=Path, default=Path(__file__).parent / "relbench"
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="compact free pages after an applied migration",
    )
    args = parser.parse_args(argv)
    paths = args.database or sorted(args.database_dir.glob("*.sqlite"))
    if not paths:
        raise FileNotFoundError("No SQLite databases selected")
    for path in paths:
        print(
            json.dumps(
                asdict(initialize_database(path, apply=args.apply, vacuum=args.vacuum))
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
