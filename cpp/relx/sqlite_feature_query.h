#pragma once

#include "sqlite_features.h"

struct sqlite3;

namespace relx {

using SqliteTableCache = SqlTableCache;

void fetch_sqlite_table(sqlite3* db, const SqliteTableSpec& table,
                        const std::vector<int64_t>& ids, FeatureBatch& output,
                        SqliteTableCache* cache = nullptr);

}  // namespace relx
