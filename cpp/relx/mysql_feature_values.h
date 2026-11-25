#pragma once

#include "online_sql_features.h"

#include <mysql.h>

#include <string>

namespace relx {

struct MySqlCell {
  bool is_null = false;
  bool binary = false;
  enum_field_types type = MYSQL_TYPE_NULL;
  std::string data;
};

void append_mysql_feature(const MySqlCell& cell, const SqlFeatureSpec& spec,
                          NativeFeatureBlock& output);

}  // namespace relx
