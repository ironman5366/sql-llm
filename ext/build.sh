#!/bin/bash
# Build the sql_llm DuckDB extension.
# Only compiles our source file — does NOT rebuild DuckDB.
# Requires: ext/duckdb/ headers (git submodule), libcurl, c++ compiler.
#
# First-time setup: run `make` once to build libduckdb_static.a and third-party libs.
# After that, this script rebuilds just the extension in <1 second.
#
# Usage: cd ext && ./build.sh
#        ./build.sh --full   (force rebuild DuckDB static libs)
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

# Use mold linker if available (much faster), fall back to default
LINKER_FLAG=""
if command -v mold &>/dev/null; then
    LINKER_FLAG="-fuse-ld=mold"
fi

# Use ccache if available
CXX="c++"
if command -v ccache &>/dev/null; then
    CXX="ccache c++"
fi

# Build DuckDB static libs if missing (or if --full is passed)
if [ ! -f "$BUILD/src/libduckdb_static.a" ] || [ "${1:-}" = "--full" ]; then
    echo "Building DuckDB static libs (this takes a while the first time)..."
    MAKEFLAGS="-j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)" GEN=ninja make release
    echo ""
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

OBJ=build/sql_llm_extension.o

echo -n "Compiling... "
START=$(date +%s%N)
$CXX \
    -DDUCKDB_BUILD_LIBRARY \
    -DDUCKDB_BUILD_LOADABLE_EXTENSION \
    $INCLUDES \
    -O3 -std=c++11 -fPIC -fvisibility=hidden \
    -c -o "$OBJ" \
    src/sql_llm_extension.cpp
END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))
echo "done (${ELAPSED}ms)"

echo -n "Linking... "
START=$(date +%s%N)
c++ \
    $LINKER_FLAG \
    -shared \
    -o "$RAW_LIB" \
    "$OBJ" \
    -Wl,--whole-archive $STATIC_LIBS -Wl,--no-whole-archive \
    -lcurl \
    -lpthread
END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))
echo "done (${ELAPSED}ms)"

echo -n "Appending extension metadata... "
python3 extension-ci-tools/scripts/append_extension_metadata.py \
    -l "$RAW_LIB" \
    -o "$OUT" \
    -n sql_llm \
    -dv "$DUCKDB_VERSION" \
    -p "$PLATFORM" \
    -ev "$EXT_VERSION" \
    --abi-type CPP

rm -f "$RAW_LIB"
echo "done"
echo "Built: $OUT ($(du -h "$OUT" | cut -f1))"
