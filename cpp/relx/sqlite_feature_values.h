#pragma once

#include "native_features.h"
#include "sqlite_features.h"

struct sqlite3;
struct sqlite3_stmt;

namespace relx {

void append_feature(sqlite3* db, sqlite3_stmt* row, int column,
                    const SqliteFeatureSpec& spec, NativeFeatureBlock& output);

}  // namespace relx
