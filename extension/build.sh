#!/usr/bin/env bash
# Fast local rebuild for the llm DuckDB extension.
#
# The normal DuckDB extension-template build is still the source of truth for a
# full build. This script is the local edit/compile loop: after the full build
# has produced libduckdb_static.a once, rebuild only src/llm_extension.cpp and
# append DuckDB extension metadata.

set -euo pipefail

cd "$(dirname "$0")"

EXT_NAME=llm
DUCKDB=${DUCKDB_SRC_DIR:-duckdb}
BUILD=${DUCKDB_BUILD_DIR:-build/release}
FAST_BUILD_DIR=${FAST_BUILD_DIR:-build/fast}
OUT=${FAST_OUTPUT:-${BUILD}/extension/${EXT_NAME}/${EXT_NAME}.duckdb_extension}
RAW_LIB=${FAST_BUILD_DIR}/${EXT_NAME}.so
OBJ=${FAST_BUILD_DIR}/${EXT_NAME}_extension.o

FULL_BUILD=0
if [ "${1:-}" = "--full" ]; then
	FULL_BUILD=1
elif [ "${1:-}" != "" ]; then
	echo "usage: ./build.sh [--full]" >&2
	exit 2
fi

jobs() {
	nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4
}

STATIC_LIB=${BUILD}/src/libduckdb_static.a
DUMMY_LOADER=${BUILD}/extension/libdummy_static_extension_loader.a

if [ ! -f "${STATIC_LIB}" ] || [ ! -f "${DUMMY_LOADER}" ] || [ "${FULL_BUILD}" = 1 ]; then
	echo "Building DuckDB static libraries first; later ./build.sh runs only rebuild the extension."
	FULL_MAKE_ARGS=(
		-f extension-ci-tools/makefiles/duckdb_extension.Makefile
		release
		"PROJ_DIR=$(pwd)/"
		"EXT_NAME=${EXT_NAME}"
		"EXT_CONFIG=$(pwd)/extension_config.cmake"
	)
	if [ -z "${GEN:-}" ] && command -v ninja >/dev/null 2>&1; then
		GEN=ninja MAKEFLAGS="-j$(jobs)" make "${FULL_MAKE_ARGS[@]}"
	else
		MAKEFLAGS="-j$(jobs)" make "${FULL_MAKE_ARGS[@]}"
	fi
	echo
fi

if [ ! -f "${STATIC_LIB}" ]; then
	echo "missing ${STATIC_LIB}; run ./build.sh --full or make release" >&2
	exit 1
fi
if [ ! -f "${DUMMY_LOADER}" ]; then
	echo "missing ${DUMMY_LOADER}; run ./build.sh --full or make release" >&2
	exit 1
fi

mkdir -p "${FAST_BUILD_DIR}" "$(dirname "${OUT}")"

INCLUDES=(
	"-I${DUCKDB}/src/include"
	"-Isrc/include"
)

for dir in \
	fsst \
	fmt/include \
	hyperloglog \
	fastpforlib \
	skiplist \
	ska_sort \
	fast_float \
	re2 \
	miniz \
	utf8proc/include \
	concurrentqueue \
	pcg \
	pdqsort \
	tdigest \
	mbedtls/include \
	jaro_winkler \
	vergesort \
	httplib \
	yyjson/include \
	zstd/include
do
	INCLUDES+=("-I${DUCKDB}/third_party/${dir}")
done

if command -v ccache >/dev/null 2>&1; then
	CXX_CMD=(ccache "${CXX:-c++}")
else
	CXX_CMD=("${CXX:-c++}")
fi

LINK_CMD=("${LINK_CXX:-c++}")
LINK_FLAGS=()
if command -v mold >/dev/null 2>&1; then
	LINK_FLAGS+=("-fuse-ld=mold")
fi

duckdb_version() {
	git -C "${DUCKDB}" describe --tags --exact-match --match 'v*' 2>/dev/null ||
		git -C "${DUCKDB}" rev-parse --short HEAD
}

extension_version() {
	git -C .. rev-parse --short HEAD
}

PYTHON_CMD=(uv run --no-project python)
if ! command -v uv >/dev/null 2>&1; then
	PYTHON_CMD=(python3)
fi

echo -n "Compiling ${EXT_NAME}... "
start=$(date +%s%N)
"${CXX_CMD[@]}" \
	-DDUCKDB_BUILD_LIBRARY \
	-DDUCKDB_BUILD_LOADABLE_EXTENSION \
	-DEXT_VERSION_LLM="\"$(extension_version)\"" \
	"${INCLUDES[@]}" \
	-O3 -DNDEBUG -ffunction-sections -fdata-sections \
	-std=c++11 -fPIC -fvisibility=hidden \
	-c -o "${OBJ}" \
	src/llm_extension.cpp
end=$(date +%s%N)
echo "done ($(((end - start) / 1000000))ms)"

echo -n "Linking ${EXT_NAME}... "
start=$(date +%s%N)
"${LINK_CMD[@]}" \
	"${LINK_FLAGS[@]}" \
	-shared \
	-Wl,-soname,${EXT_NAME}.duckdb_extension \
	-o "${RAW_LIB}" \
	"${OBJ}" \
	"${STATIC_LIB}" \
	"${DUMMY_LOADER}" \
	-Wl,--gc-sections \
	-Wl,--exclude-libs,ALL \
	-ldl
end=$(date +%s%N)
echo "done ($(((end - start) / 1000000))ms)"

METADATA_ARGS=(
	-l "${RAW_LIB}"
	-o "${OUT}"
	-n "${EXT_NAME}"
	-dv "$(duckdb_version)"
	-ev "$(extension_version)"
	--abi-type CPP
)

if [ -n "${DUCKDB_PLATFORM:-}" ]; then
	METADATA_ARGS+=(-p "${DUCKDB_PLATFORM}")
elif [ -f "${BUILD}/duckdb_platform_out" ]; then
	METADATA_ARGS+=(-pf "${BUILD}/duckdb_platform_out")
else
	echo "missing ${BUILD}/duckdb_platform_out; run ./build.sh --full or set DUCKDB_PLATFORM" >&2
	exit 1
fi

echo -n "Appending extension metadata... "
"${PYTHON_CMD[@]}" extension-ci-tools/scripts/append_extension_metadata.py "${METADATA_ARGS[@]}"
rm -f "${RAW_LIB}"
echo "done"

echo "Built: ${OUT} ($(du -h "${OUT}" | cut -f1))"
