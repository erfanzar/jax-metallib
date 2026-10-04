#!/bin/bash
# Lint only what a change touches.
#
# The tree carries formatting debt older than this script: at 4d42b55 a whole-file
# clang-format check fails on 7 native files (4 of them hand-formatted on purpose)
# and `ruff format --check` fails on tests/configs/{__init__,util}.py. Whole-repo
# gates would fail every change on code it never edited, so this checks:
#   Python : `ruff check` on added/modified .py files,
#            `ruff format --check` on added .py files only;
#   Native : clang-format on the changed lines of modified src/pjrt_plugin
#            .mm/.cc/.h files (git clang-format), whole-file on added ones.
#
# Usage: scripts/ci_lint_changed.sh <base-ref>
#   Compares the working tree (committed + uncommitted + untracked, non-ignored)
#   with <base-ref>. Tools: $RUFF (default ruff), $CLANG_FORMAT (default
#   clang-format; git-clang-format must be on PATH). CI pins ruff 0.14.13 and
#   clang-format 21.1.8 to match uv.lock and the local toolchain.
set -euo pipefail

BASE="${1:?usage: $0 <base-ref>}"
RUFF="${RUFF:-ruff}"
CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"
EXCLUDE='^(build|build-debug|build\.[^/]*|dist|tmp-files|\.review-tmp)/'
NATIVE=('src/pjrt_plugin/*.mm' 'src/pjrt_plugin/*.cc' 'src/pjrt_plugin/*.h')

cd "$(git rev-parse --show-toplevel)"
if ! git rev-parse --verify --quiet "$BASE^{commit}" >/dev/null; then
    echo "unknown base ref: $BASE" >&2
    exit 2
fi

# changed <diff-filter> <pathspec>...: paths differing from BASE; untracked
# files count as added when the filter includes A.
changed() {
    local filter="$1"
    shift
    {
        git diff --name-only --diff-filter="$filter" "$BASE" -- "$@"
        case "$filter" in *A*) git ls-files --others --exclude-standard -- "$@" ;; esac
    } | grep -vE "$EXCLUDE" | sort -u || true
}

count() { if [ -n "$1" ]; then echo "$1" | wc -l | tr -d ' '; else echo 0; fi; }

fail=0
py_changed=$(changed AM '*.py')
py_added=$(changed A '*.py')
native_modified=$(changed M "${NATIVE[@]}")
native_added=$(changed A "${NATIVE[@]}")

echo "== base $(git rev-parse --short "$BASE"): python changed $(count "$py_changed")" \
    "added $(count "$py_added"); native modified $(count "$native_modified") added $(count "$native_added")"

# Word splitting on these lists is intended (paths contain no whitespace).
if [ -n "$py_changed" ]; then
    echo "== ruff check (added/modified)"
    # shellcheck disable=SC2086
    "$RUFF" check $py_changed || fail=1
fi
if [ -n "$py_added" ]; then
    echo "== ruff format --check (added)"
    # shellcheck disable=SC2086
    "$RUFF" format --check $py_added || fail=1
fi
if [ -n "$native_modified" ]; then
    echo "== clang-format, changed lines (modified)"
    # shellcheck disable=SC2086
    out=$(git clang-format --binary "$CLANG_FORMAT" --diff "$BASE" -- $native_modified 2>&1) || true
    # Here-string, not a pipe: with pipefail, grep -q exiting early can SIGPIPE
    # the writer, fail the pipeline and silently report "clean".
    if grep -q '^@@' <<<"$out"; then
        echo "$out"
        fail=1
    else
        echo "clean"
    fi
fi
if [ -n "$native_added" ]; then
    echo "== clang-format, whole file (added)"
    # shellcheck disable=SC2086
    "$CLANG_FORMAT" --dry-run --Werror $native_added || fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "LINT FAILED" >&2
else
    echo "LINT OK"
fi
exit "$fail"
