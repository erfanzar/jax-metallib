#!/bin/bash
# Check a built wheel without a Metal GPU (hosted CI runners expose none):
#   - the platform tag and the bundled dylib's LC_BUILD_VERSION minos match the
#     minimum macOS (a CMake ordering bug once shipped minos = build host OS);
#   - the dylib exports GetPjrtApi and links only system libraries;
#   - a fresh venv with only the wheel imports the plugin from site-packages and
#     loads the dylib.
# With CHECK_DEVICE=1 it also requires jax.devices("mps") to return a device
# (local use on Apple Silicon only).
#
# Usage: scripts/ci_check_wheel.sh <wheel> [min-macos, default 15.0]
#   $PYTHON selects the interpreter for the venv (default python3).
set -euo pipefail

WHEEL="${1:?usage: $0 <wheel> [min-macos]}"
MINOS="${2:-15.0}"
PY="${PYTHON:-python3}"
WHEEL="$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")"

tag="macosx_${MINOS//./_}_arm64"
case "$(basename "$WHEEL")" in
*-"$tag".whl) echo "tag ok: $tag" ;;
*)
    echo "wheel tag is not $tag: $(basename "$WHEEL")" >&2
    exit 1
    ;;
esac

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
"$PY" -m venv "$work/venv"
"$work/venv/bin/python" -m pip install --quiet --disable-pip-version-check "$WHEEL"

# Run from the scratch dir so the import cannot resolve to a source checkout.
cd "$work"
lib=$("$work/venv/bin/python" -c 'import importlib.util, pathlib
s = importlib.util.find_spec("jax_plugins.silicon")
print(pathlib.Path(s.origin).parent / "lib" / "libpjrt_plugin_silicon.dylib")')
[ -f "$lib" ] || {
    echo "bundled dylib missing: $lib" >&2
    exit 1
}

# Capture tool output before matching: under pipefail an early-exiting reader
# (awk exit, grep -q) SIGPIPEs the writer and the pipeline returns 141.
load_cmds=$(otool -l "$lib")
got=$(awk '/LC_BUILD_VERSION/ {f = 1} f && $1 == "minos" {print $2; exit}' <<<"$load_cmds")
if [ "$got" != "$MINOS" ]; then
    echo "dylib minos $got != $MINOS" >&2
    exit 1
fi
echo "minos ok: $got"

exports=$(nm -gU "$lib")
if ! grep -q ' _GetPjrtApi$' <<<"$exports"; then
    echo "GetPjrtApi is not exported" >&2
    exit 1
fi
echo "export ok: GetPjrtApi"

nonsys=$(otool -L "$lib" | tail -n +2 | awk '{print $1}' |
    grep -vE '^(/usr/lib/|/System/Library/|@rpath/libpjrt_plugin_silicon\.dylib$)' || true)
if [ -n "$nonsys" ]; then
    echo "non-system dylib dependencies:" >&2
    echo "$nonsys" >&2
    exit 1
fi
echo "links ok: system libraries only"

"$work/venv/bin/python" - "$lib" <<'EOF'
import ctypes
import sys

import jax_plugins.silicon as plugin

assert "site-packages" in plugin.__file__, plugin.__file__
assert hasattr(ctypes.CDLL(sys.argv[1]), "GetPjrtApi")
print("import ok:", plugin.__file__)
EOF

if [ "${CHECK_DEVICE:-0}" = "1" ]; then
    JAX_PLATFORMS=mps "$work/venv/bin/python" -c '
import jax
import jax.numpy as jnp

d = jax.devices("mps")
assert d, "no mps device"
x = jax.device_put(jnp.arange(8.0), d[0])
assert float((x * 2).sum()) == 56.0
print("device ok:", d)'
fi
echo "WHEEL CHECK OK"
