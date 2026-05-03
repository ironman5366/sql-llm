PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=llm
EXT_CONFIG=${PROJ_DIR}extension_config.cmake
TEMPLATE_MAKE=extension-ci-tools/makefiles/duckdb_extension.Makefile
TEMPLATE_ARGS=PROJ_DIR="$(PROJ_DIR)" EXT_NAME="$(EXT_NAME)" EXT_CONFIG="$(EXT_CONFIG)" GEN="$(GEN)"

# Ninja is materially faster for DuckDB's many-target build graph. Keep GEN
# overrideable for environments that need a different CMake generator.
ifeq ($(origin GEN), undefined)
ifneq ($(shell command -v ninja 2>/dev/null),)
GEN := ninja
endif
endif

.PHONY: all release fast fast-full full full-release debug test clean format format-check format-fix

all: release

release fast:
	./build.sh

fast-full:
	./build.sh --full

full full-release:
	$(MAKE) -f $(TEMPLATE_MAKE) release $(TEMPLATE_ARGS)

debug:
	$(MAKE) -f $(TEMPLATE_MAKE) debug $(TEMPLATE_ARGS)

test:
	$(MAKE) -f $(TEMPLATE_MAKE) test $(TEMPLATE_ARGS)

clean:
	rm -rf build
	$(MAKE) -C duckdb clean

format format-check format-fix:
	$(MAKE) -f $(TEMPLATE_MAKE) $@ $(TEMPLATE_ARGS)
