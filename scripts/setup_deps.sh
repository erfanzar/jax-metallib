#!/bin/bash
# Setup script for jax-metallib dependencies (LLVM/MLIR + StableHLO)
# These are built once and installed to a prefix directory.
#
# Usage:
#   ./scripts/setup_deps.sh [--prefix /path/to/install] [--force]
#
# Options:
#   --prefix PATH   Install location (default: $HOME/.local/jax-metallib-deps)
#   --jobs N        Number of parallel jobs (default: number of CPUs)
#   --force         Force rebuild even if already installed
#
# Default prefix: $HOME/.local/jax-metallib-deps

set -e

# Configuration
PREFIX="${PREFIX:-$HOME/.local/jax-metallib-deps}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"
BUILD_DIR="${BUILD_DIR:-/tmp/jax-metallib-deps-build}"
# CMake reads this as the default CMAKE_OSX_DEPLOYMENT_TARGET for every dep build
# below. Without it the static deps target the host OS (ld warned "built for newer
# macOS version (26.0) than being linked (15.0)" when linking the 15.0 plugin).
# Keep in sync with CMAKE_OSX_DEPLOYMENT_TARGET in CMakeLists.txt.
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"

# Pin to versions matching jaxlib 0.11.1 for bytecode compatibility
# These are extracted from:
#   - JAX MODULE.bazel archive_override @ jax-v0.11.1
#     (openxla/xla archive dcf304bc5dca1932b99f740b911dbd73631a1a69)
#   - XLA third_party/stablehlo/workspace.bzl @ dcf304bc5dca1932b99f740b911dbd73631a1a69
#
# LLVM pin note: StableHLO 7b1b157 integrates LLVM at 6d1ace547c9f (per its
# build_tools/llvm_version.txt). Using StableHLO's own integrated LLVM keeps the
# MLIR C++ API (PropertyRef etc.) exactly what StableHLO's source expects —
# XLA's Bazel-pinned LLVM may drift from it. XLA dcf304bc defines the PJRT C API
# header; only the build-time LLVM pin needs to match StableHLO.
XLA_COMMIT="${XLA_COMMIT:-dcf304bc5dca1932b99f740b911dbd73631a1a69}"
STABLEHLO_COMMIT="${STABLEHLO_COMMIT:-7b1b15781ccbd770f50c7eef4b0c3e03834649fd}"
LLVM_COMMIT_OVERRIDE="${LLVM_COMMIT_OVERRIDE:-6d1ace547c9f9ecf7df9ab87e2db74b5bd3f199d}"

# Abseil and Protobuf versions (protobuf depends on abseil)
ABSEIL_VERSION="${ABSEIL_VERSION:-20250127.0}"
PROTOBUF_VERSION="${PROTOBUF_VERSION:-29.3}"

# SHA256 checksums for the release tarballs above (supply-chain guard: the
# download is verified before extraction). Override together with the version.
ABSEIL_SHA256="${ABSEIL_SHA256:-16242f394245627e508ec6bb296b433c90f8d914f73b9c026fddb905e27276e8}"
PROTOBUF_SHA256="${PROTOBUF_SHA256:-008a11cc56f9b96679b4c285fd05f46d317d685be3ab524b2a310be0fbad987e}"

# Parse arguments
FORCE_REBUILD=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --force)
            FORCE_REBUILD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If force rebuild, remove existing installations
if [ "$FORCE_REBUILD" = true ]; then
    echo "=== Force rebuild: removing existing installations ==="
    rm -rf "$PREFIX/lib/cmake/mlir" "$PREFIX/lib/cmake/llvm"
    rm -f "$PREFIX/lib/libStablehloOps.a"
    rm -f "$PREFIX/lib/libprotobuf.a" "$PREFIX/lib/libabsl_base.a"
    # Remove installed headers too: stale generated StableHLO headers shadow the
    # fresh ones via -I order and break the new build (e.g. tablegen enum
    # helpers referencing APIs that the newer LLVM removed).
    rm -rf "$PREFIX/include/stablehlo" "$PREFIX/include/llvm" "$PREFIX/include/mlir"
    rm -rf "$BUILD_DIR/llvm-build" "$BUILD_DIR/stablehlo-build"
    rm -rf "$BUILD_DIR/abseil-build" "$BUILD_DIR/protobuf-build"
fi

echo "=== jax-metallib dependency setup ==="
echo "Prefix:       $PREFIX"
echo "Jobs:         $JOBS"
echo "Build dir:    $BUILD_DIR"
echo "Abseil:       $ABSEIL_VERSION"
echo "Protobuf:     $PROTOBUF_VERSION"
echo "XLA:          $XLA_COMMIT"
echo "StableHLO:    $STABLEHLO_COMMIT"
echo "LLVM:         $LLVM_COMMIT_OVERRIDE"
echo "Force:        $FORCE_REBUILD"
echo ""

mkdir -p "$PREFIX"
mkdir -p "$BUILD_DIR"

# Check for required tools
for tool in cmake ninja git; do
    if ! command -v $tool &> /dev/null; then
        echo "Error: $tool is required but not installed"
        echo "On macOS: brew install cmake ninja"
        exit 1
    fi
done

# Ensure a repository checkout exists at the requested commit, recovering from
# stale or partial directories automatically.
ensure_git_checkout() {
    local repo_dir="$1"
    local repo_url="$2"
    local repo_commit="$3"
    local repo_name="$4"

    local needs_clone=false
    if [ ! -d "$repo_dir/.git" ]; then
        needs_clone=true
    else
        cd "$repo_dir"
        # Recover from stale git lock files left by interrupted operations.
        rm -f .git/shallow.lock .git/index.lock .git/packed-refs.lock .git/config.lock .git/FETCH_HEAD.lock

        if ! git remote get-url origin >/dev/null 2>&1; then
            git remote add origin "$repo_url"
        fi

        local current_commit
        current_commit="$(git rev-parse HEAD 2>/dev/null || true)"
        if [ -z "$current_commit" ]; then
            echo "=== $repo_name checkout is invalid; recloning ==="
            needs_clone=true
        elif [ "$current_commit" != "$repo_commit" ]; then
            echo "=== Updating $repo_name to commit $repo_commit ==="
            if git fetch --depth 1 origin "$repo_commit"; then
                git checkout FETCH_HEAD
            else
                echo "=== Failed to update $repo_name checkout; recloning ==="
                needs_clone=true
            fi
        else
            echo "=== $repo_name already at requested commit ==="
        fi
    fi

    if [ "$needs_clone" = true ]; then
        echo "=== Cloning $repo_name at commit $repo_commit ==="
        rm -rf "$repo_dir"
        mkdir -p "$repo_dir"
        cd "$repo_dir"
        git init
        git remote add origin "$repo_url"
        git fetch --depth 1 origin "$repo_commit"
        git checkout FETCH_HEAD
    fi
}

# Build Abseil (required by protobuf, must be static for wheel distribution)
ABSEIL_DIR="$BUILD_DIR/abseil-cpp"
ABSEIL_BUILD_DIR="$BUILD_DIR/abseil-build"
if [ ! -f "$PREFIX/lib/libabsl_base.a" ]; then
    echo "=== Downloading Abseil $ABSEIL_VERSION ==="
    if [ ! -d "$ABSEIL_DIR" ]; then
        curl -L -o "$BUILD_DIR/abseil-$ABSEIL_VERSION.tar.gz" "https://github.com/abseil/abseil-cpp/archive/refs/tags/$ABSEIL_VERSION.tar.gz"
        echo "$ABSEIL_SHA256  $BUILD_DIR/abseil-$ABSEIL_VERSION.tar.gz" | shasum -a 256 -c - >/dev/null
        tar xzf "$BUILD_DIR/abseil-$ABSEIL_VERSION.tar.gz" -C "$BUILD_DIR"
        mv "$BUILD_DIR/abseil-cpp-$ABSEIL_VERSION" "$ABSEIL_DIR"
    fi

    echo "=== Building Abseil (static) ==="
    cmake -G Ninja -B "$ABSEIL_BUILD_DIR" -S "$ABSEIL_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DABSL_BUILD_TESTING=OFF \
        -DABSL_PROPAGATE_CXX_STD=ON

    cmake --build "$ABSEIL_BUILD_DIR" -j "$JOBS"
    cmake --install "$ABSEIL_BUILD_DIR"
    echo "Abseil installed to $PREFIX"
else
    echo "=== Abseil already installed ==="
fi

# Build Protobuf (must be static for wheel distribution)
PROTOBUF_DIR="$BUILD_DIR/protobuf"
PROTOBUF_BUILD_DIR="$BUILD_DIR/protobuf-build"
if [ ! -f "$PREFIX/lib/libprotobuf.a" ]; then
    echo "=== Downloading Protobuf $PROTOBUF_VERSION ==="
    if [ ! -d "$PROTOBUF_DIR" ]; then
        curl -L -o "$BUILD_DIR/protobuf-$PROTOBUF_VERSION.tar.gz" "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v$PROTOBUF_VERSION.tar.gz"
        echo "$PROTOBUF_SHA256  $BUILD_DIR/protobuf-$PROTOBUF_VERSION.tar.gz" | shasum -a 256 -c - >/dev/null
        tar xzf "$BUILD_DIR/protobuf-$PROTOBUF_VERSION.tar.gz" -C "$BUILD_DIR"
        mv "$BUILD_DIR/protobuf-$PROTOBUF_VERSION" "$PROTOBUF_DIR"
    fi

    echo "=== Building Protobuf (static) ==="
    cmake -G Ninja -B "$PROTOBUF_BUILD_DIR" -S "$PROTOBUF_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_PREFIX_PATH="$PREFIX" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -Dprotobuf_BUILD_TESTS=OFF \
        -Dprotobuf_ABSL_PROVIDER=package \
        -Dprotobuf_BUILD_PROTOBUF_BINARIES=ON \
        -Dprotobuf_BUILD_PROTOC_BINARIES=ON

    cmake --build "$PROTOBUF_BUILD_DIR" -j "$JOBS"
    cmake --install "$PROTOBUF_BUILD_DIR"
    echo "Protobuf installed to $PREFIX"
else
    echo "=== Protobuf already installed ==="
fi

# Clone StableHLO at pinned commit for jaxlib compatibility
STABLEHLO_DIR="$BUILD_DIR/stablehlo"
ensure_git_checkout "$STABLEHLO_DIR" "https://github.com/openxla/stablehlo.git" "$STABLEHLO_COMMIT" "StableHLO"

cd "$STABLEHLO_DIR"
# Use LLVM commit override if set, otherwise read from StableHLO
if [ -n "$LLVM_COMMIT_OVERRIDE" ]; then
    LLVM_COMMIT="$LLVM_COMMIT_OVERRIDE"
    echo "Using LLVM commit override: $LLVM_COMMIT"
else
    LLVM_COMMIT=$(cat build_tools/llvm_version.txt)
    echo "StableHLO requires LLVM commit: $LLVM_COMMIT"
fi

# Clone LLVM - fetch only the specific commit we need
LLVM_DIR="$BUILD_DIR/llvm-project"
ensure_git_checkout "$LLVM_DIR" "https://github.com/llvm/llvm-project.git" "$LLVM_COMMIT" "LLVM"

# Build LLVM/MLIR
LLVM_BUILD_DIR="$BUILD_DIR/llvm-build"
if [ ! -f "$PREFIX/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "=== Building LLVM/MLIR ==="
    cmake -G Ninja -B "$LLVM_BUILD_DIR" -S "$LLVM_DIR/llvm" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
        -DLLVM_ENABLE_ZSTD=OFF \
        -DLLVM_ENABLE_ZLIB=OFF

    cmake --build "$LLVM_BUILD_DIR" -j "$JOBS"
    cmake --install "$LLVM_BUILD_DIR"
    echo "LLVM/MLIR installed to $PREFIX"
else
    echo "=== LLVM/MLIR already installed ==="
fi

# Build StableHLO
STABLEHLO_BUILD_DIR="$BUILD_DIR/stablehlo-build"
if [ ! -f "$PREFIX/lib/libStablehloOps.a" ]; then
    echo "=== Patching StableHLO (disable lit tests) ==="
    # StableHLO's test CMakeLists require LLVM FileCheck which we don't install
    # Wrap the lit test setup in if(TARGET FileCheck) to skip when not available
    for f in "$STABLEHLO_DIR/stablehlo/tests/CMakeLists.txt" \
             "$STABLEHLO_DIR/stablehlo/testdata/CMakeLists.txt" \
             "$STABLEHLO_DIR/stablehlo/conversions/linalg/tests/CMakeLists.txt" \
             "$STABLEHLO_DIR/stablehlo/conversions/tosa/tests/CMakeLists.txt"; do
        if [ -f "$f" ] && ! grep -q "if(TARGET FileCheck)" "$f"; then
            python3 -c "
import re, sys
content = open('$f').read()
pattern = r'(configure_lit_site_cfg\([^)]+\)\s*add_lit_testsuite\([^)]+\)\s*add_dependencies\([^)]+\))'
def wrap(m): return 'if(TARGET FileCheck)\n' + m.group(1) + '\nendif()'
print(re.sub(pattern, wrap, content, flags=re.DOTALL))
" > "$f.tmp" && mv "$f.tmp" "$f"
        fi
    done

    echo "=== Building StableHLO ==="
    cmake -G Ninja -B "$STABLEHLO_BUILD_DIR" -S "$STABLEHLO_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DMLIR_DIR="$PREFIX/lib/cmake/mlir" \
        -DLLVM_DIR="$PREFIX/lib/cmake/llvm" \
        -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
        -DSTABLEHLO_BUILD_EMBEDDED=OFF

    cmake --build "$STABLEHLO_BUILD_DIR" -j "$JOBS"
    cmake --install "$STABLEHLO_BUILD_DIR"

    # StableHLO doesn't install headers by default, do it manually
    echo "=== Installing StableHLO headers ==="
    mkdir -p "$PREFIX/include/stablehlo/dialect"
    mkdir -p "$PREFIX/include/stablehlo/api"
    cp "$STABLEHLO_DIR/stablehlo/dialect/"*.h "$PREFIX/include/stablehlo/dialect/"
    cp "$STABLEHLO_DIR/stablehlo/api/"*.h "$PREFIX/include/stablehlo/api/"
    # Copy generated tablegen headers
    cp "$STABLEHLO_BUILD_DIR/stablehlo/dialect/"*.inc "$PREFIX/include/stablehlo/dialect/" 2>/dev/null || true

    echo "StableHLO installed to $PREFIX"
else
    echo "=== StableHLO already installed ==="
fi

# Install the exact pinned PJRT blob, even when a different header is already
# installed. Reading the blob also keeps edits in the dependency checkout out
# of the installation, without checking out or overwriting those edits.
XLA_DIR="$BUILD_DIR/xla"
XLA_HEADER="xla/pjrt/c/pjrt_c_api.h"
if ! git -C "$XLA_DIR" cat-file -e "$XLA_COMMIT:$XLA_HEADER" 2>/dev/null; then
    ensure_git_checkout "$XLA_DIR" "https://github.com/openxla/xla.git" "$XLA_COMMIT" "XLA"
fi
mkdir -p "$PREFIX/include/xla/pjrt/c"
PJRT_HEADER_TMP=$(mktemp "$PREFIX/include/xla/pjrt/c/pjrt_c_api.h.XXXXXX")
if ! git -C "$XLA_DIR" show "$XLA_COMMIT:$XLA_HEADER" > "$PJRT_HEADER_TMP"; then
    rm -f "$PJRT_HEADER_TMP"
    exit 1
fi
if cmp -s "$PJRT_HEADER_TMP" "$PREFIX/include/$XLA_HEADER"; then
    rm -f "$PJRT_HEADER_TMP"
    echo "=== XLA PJRT header already matches $XLA_COMMIT ==="
else
    chmod 644 "$PJRT_HEADER_TMP"
    mv "$PJRT_HEADER_TMP" "$PREFIX/include/$XLA_HEADER"
    echo "XLA PJRT header installed from $XLA_COMMIT to $PREFIX"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Dependencies installed to: $PREFIX"
echo ""
echo "To build jax-metallib, use:"
echo "  cmake -B build -DCMAKE_PREFIX_PATH=$PREFIX"
echo "  cmake --build build"
echo ""
echo "Or set environment variable:"
echo "  export CMAKE_PREFIX_PATH=$PREFIX"
