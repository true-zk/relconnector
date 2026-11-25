"""Build the RelX graph extension in place for local development."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sysconfig
from ctypes.util import find_library
from pathlib import Path

import pyarrow
import pybind11


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = root / "cpp/relx"
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str):
        raise TypeError("Python extension suffix is unavailable")
    output = root / "src/relx" / f"_relx_native{suffix}"
    temporary_output = output.with_name(output.name + ".tmp")
    include = sysconfig.get_paths()["include"]
    compiler = os.environ.get("CXX", "c++")
    sqlite_include = Path(os.environ.get("SQLITE3_INCLUDE_DIR", "/usr/include"))
    if not (sqlite_include / "sqlite3.h").is_file():
        raise FileNotFoundError(
            "sqlite3.h is required; set SQLITE3_INCLUDE_DIR to its directory"
        )
    sqlite_library = os.environ.get("SQLITE3_LIBRARY", find_library("sqlite3") or "")
    if not sqlite_library:
        raise FileNotFoundError("SQLite library is required for native feature reads")
    mysql_config = (
        os.environ.get("MYSQL_CONFIG") or shutil.which("mysql_config")
        if os.environ.get("RELX_WITH_MYSQL", "1") != "0"
        else None
    )
    mysql_compile: list[str] = []
    mysql_link: list[str] = []
    mysql_sources: list[str] = []
    if mysql_config:
        mysql_compile = [
            "-DRELX_WITH_MYSQL",
            *shlex.split(subprocess.check_output(
                [mysql_config, "--cflags"], text=True
            )),
        ]
        mysql_link = shlex.split(subprocess.check_output(
            [mysql_config, "--libs"], text=True
        ))
        mysql_link += [
            f"-Wl,-rpath,{flag[2:]}"
            for flag in mysql_link if flag.startswith("-L")
        ]
        mysql_sources = [str(sources / name) for name in (
            "mysql_features.cpp", "mysql_feature_query.cpp",
            "mysql_feature_values.cpp", "bindings_mysql.cpp",
        )]
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-fvisibility=hidden",
        "-shared",
        "-pthread",
        "-Wall",
        "-Wextra",
        f"-I{include}",
        f"-I{pybind11.get_include()}",
        f"-I{pyarrow.get_include()}",
        f"-I{sqlite_include}",
        *mysql_compile,
        str(sources / "arrow_batch.cpp"),
        str(sources / "graph_index.cpp"),
        str(sources / "multi_hop.cpp"),
        str(sources / "native_features.cpp"),
        str(sources / "glove_model.cpp"),
        str(sources / "json_array.cpp"),
        str(sources / "online_sql_features.cpp"),
        str(sources / "sql_time.cpp"),
        str(sources / "sql_number.cpp"),
        str(sources / "native_pipeline.cpp"),
        str(sources / "native_online_pipeline.cpp"),
        str(sources / "sqlite_features.cpp"),
        str(sources / "sqlite_feature_query.cpp"),
        str(sources / "sqlite_feature_values.cpp"),
        str(sources / "bindings.cpp"),
        str(sources / "bindings_features.cpp"),
        str(sources / "bindings_pipeline.cpp"),
        str(sources / "bindings_online.cpp"),
        str(sources / "bindings_sqlite.cpp"),
        str(sources / "bindings_sql_common.cpp"),
        *mysql_sources,
        sqlite_library if "/" in sqlite_library else f"-l:{sqlite_library}",
        *mysql_link,
        "-o",
        str(temporary_output),
    ]
    try:
        subprocess.run(command, check=True)
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
