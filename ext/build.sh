#!/bin/bash
# Build the sql_llm DuckDB extension.
# Only compiles our source file — does NOT rebuild DuckDB.
# Requires: ext/duckdb/ headers (git submodule), libcurl, c++ compiler.
#
# First-time setup: run `make` once to build libduckdb_static.a and third-party libs.
# After that, this script rebuilds just the extension in <1 second.
#
# Usage: cd ext && ./build.sh
#        Output: build/sql_llm.duckdb_extension

set -euo pipefail
cd "$(dirname "$0")"

DUCKDB=duckdb
BUILD=build/release
DUCKDB_VERSION=v1.4.3
PLATFORM=linux_amd64
EXT_VERSION=0.3.0
RAW_LIB=build/sql_llm.so
OUT=build/sql_llm.duckdb_extension

# Check that static libs exist (from initial `make` build)
if [ ! -f "$BUILD/src/libduckdb_static.a" ]; then
    echo "Error: $BUILD/src/libduckdb_static.a not found."
    echo "Run 'make' once first to build DuckDB static libs."
    exit 1
fi

mkdir -p build

INCLUDES="-I${DUCKDB}/src/include -Isrc/include"
for d in fsst fmt/include hyperloglog fastpforlib skiplist ska_sort fast_float \
         re2 miniz utf8proc/include concurrentqueue pcg pdqsort tdigest \
         mbedtls/include jaro_winkler vergesort yyjson/include zstd/include \
         libpg_query/include; do
    INCLUDES="$INCLUDES -I${DUCKDB}/third_party/$d"
done

# Collect static libs for linking
STATIC_LIBS="$BUILD/src/libduckdb_static.a"
for lib in $BUILD/third_party/*/lib*.a; do
    STATIC_LIBS="$STATIC_LIBS $lib"
done
# Also link bundled extension libs (jemalloc, core_functions, parquet) — but NOT our own
for lib in $BUILD/extension/*/lib*.a; do
    case "$lib" in *sql_llm*) continue ;; esac
    STATIC_LIBS="$STATIC_LIBS $lib"
done

echo "Compiling..."
c++ \
    -DDUCKDB_BUILD_LIBRARY \
    -DDUCKDB_BUILD_LOADABLE_EXTENSION \
    $INCLUDES \
    -O3 -std=c++11 -fPIC -fvisibility=hidden \
    -shared \
    -o "$RAW_LIB" \
    src/sql_llm_extension.cpp \
    -Wl,--whole-archive $STATIC_LIBS -Wl,--no-whole-archive \
    -lcurl \
    -lpthread

echo "Appending extension metadata..."
python3 extension-ci-tools/scripts/append_extension_metadata.py \
    -l "$RAW_LIB" \
    -o "$OUT" \
    -n sql_llm \
    -dv "$DUCKDB_VERSION" \
    -p "$PLATFORM" \
    -ev "$EXT_VERSION" \
    --abi-type CPP

rm -f "$RAW_LIB"
echo "Built: $OUT ($(du -h "$OUT" | cut -f1))"
