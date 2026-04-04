#!/bin/bash
# Build the sql_llm DuckDB extension.
# Only compiles our source file — does NOT rebuild DuckDB.
# Requires: ext/duckdb/ headers (git submodule), libcurl, c++ compiler.
#
# Usage: cd ext && ./build.sh
#        Output: build/sql_llm.duckdb_extension

set -euo pipefail
cd "$(dirname "$0")"

DUCKDB=duckdb
OUT=build/sql_llm.duckdb_extension

mkdir -p build

INCLUDES="-I${DUCKDB}/src/include -Isrc/include"
for d in fsst fmt/include hyperloglog fastpforlib skiplist ska_sort fast_float \
         re2 miniz utf8proc/include concurrentqueue pcg pdqsort tdigest \
         mbedtls/include jaro_winkler vergesort yyjson/include zstd/include \
         libpg_query/include; do
    INCLUDES="$INCLUDES -I${DUCKDB}/third_party/$d"
done

echo "Building sql_llm extension..."
c++ \
    -DDUCKDB_BUILD_LIBRARY \
    -DDUCKDB_BUILD_LOADABLE_EXTENSION \
    $INCLUDES \
    -O3 -std=c++11 -fPIC -fvisibility=hidden \
    -shared \
    -o "$OUT" \
    src/sql_llm_extension.cpp \
    -lcurl

echo "Built: $OUT ($(du -h "$OUT" | cut -f1))"
