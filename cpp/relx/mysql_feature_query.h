#pragma once

#include "online_sql_features.h"

struct st_mysql;
using MYSQL = st_mysql;

namespace relx {

void fetch_mysql_table(MYSQL* db, const SqlTableSpec& table,
                       const std::vector<int64_t>& ids, FeatureBatch& output,
                       SqlTableCache* cache = nullptr);

}  // namespace relx
