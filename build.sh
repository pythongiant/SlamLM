#!/usr/bin/env bash
#
# SlamLM menu bar app: build, bundle, sign, launch, capture.
#
#   ./build.sh              build + bundle + ad-hoc sign, print the .app path
#   ./build.sh debug        same, unoptimized (faster to iterate)
#   ./build.sh run          bundle, then launch the preview window
#   ./build.sh menubar      bundle, then launch as a real menu bar item
#   ./build.sh capture      bundle, then capture both panels to build/captures/
#
# The bundle carries the Python bridge in Contents/Resources/sidecar and finds
# the repo virtualenv by walking up from there, so launching the binary directly
# works from any directory. Override with SLAM_LM_PYTHON / SLAM_LM_BRIDGE_DIR.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The app lives at `menubar/` inside a source tree and at the repository root when
# published, so the checkout is the directory that holds the interpreter: beside
# these sources first, otherwise the parent that has `.venv`.
if [[ -x "$HERE/.venv/bin/python" ]]; then
    REPO="$HERE"
else
    REPO="$(cd "$HERE/.." && pwd)"
fi
CMD="${1:-build}"
CONFIG="release"
if [[ "$CMD" == "debug" ]]; then
    CONFIG="debug"
    CMD="build"
fi

APP="$HERE/build/SlamLM.app"
BIN_NAME="SlamLM"
PYTHON="${SLAM_LM_PYTHON:-$REPO/.venv/bin/python}"
MODEL="${SLAM_LM_SNAPSHOT_MODEL:-mlx-community/Qwen3-0.6B-4bit}"
TOKENS="${SLAM_LM_SNAPSHOT_TOKENS:-400}"
REQUESTS="${SLAM_LM_SNAPSHOT_REQUESTS:-3}"
# The tools capture needs a model that grounds its answer in what the tool
# returned; the small default is fast but unreliable at that, and a capture that
# shows the model ignoring its own tool result is worse than none.
TOOLS_MODEL="${SLAM_LM_SNAPSHOT_TOOLS_MODEL:-mlx-community/Qwen3-1.7B-4bit}"

bundle() {
    echo "==> swift build -c $CONFIG"
    (cd "$HERE" && swift build -c "$CONFIG")
    local bin
    bin="$(cd "$HERE" && swift build -c "$CONFIG" --show-bin-path)/$BIN_NAME"
    if [[ ! -x "$bin" ]]; then
        echo "error: no executable at $bin" >&2
        exit 1
    fi
    if [[ ! -d "$HERE/sidecar/slam_lm_bridge" ]]; then
        echo "error: bridge sources missing at $HERE/sidecar/slam_lm_bridge" >&2
        exit 1
    fi
    if [[ ! -x "$PYTHON" ]]; then
        echo "error: interpreter not executable: $PYTHON (set SLAM_LM_PYTHON)" >&2
        exit 1
    fi

    echo "==> assembling $APP"
    rm -rf "$APP"
    mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/sidecar"
    cp "$HERE/Resources/Info.plist" "$APP/Contents/Info.plist"
    cp "$bin" "$APP/Contents/MacOS/$BIN_NAME"
    cp -R "$HERE/sidecar/slam_lm_bridge" "$APP/Contents/Resources/sidecar/"
    find "$APP/Contents/Resources/sidecar" -name '__pycache__' -type d -prune -exec rm -rf {} +

    # Discovery fallbacks for a bundle launched by Finder, where the environment
    # is not inherited from this shell.
    printf '{"python":"%s","repo":"%s"}\n' "$PYTHON" "$REPO" \
        > "$APP/Contents/Resources/runtime.json"

    echo "==> ad-hoc signing"
    codesign --force --sign - --timestamp=none "$APP"
    codesign --verify --verbose=1 "$APP" 2>&1 | sed 's/^/    /'
    echo "$APP"
}

case "$CMD" in
    build)
        bundle
        ;;
    run)
        bundle
        echo "==> launching preview window"
        exec "$APP/Contents/MacOS/$BIN_NAME" --preview
        ;;
    menubar)
        bundle
        echo "==> launching menu bar item (LSUIElement)"
        exec "$APP/Contents/MacOS/$BIN_NAME"
        ;;
    capture)
        bundle
        # `docs/` holds the README's images, so refresh them in place when it
        # exists; a source tree without one gets the scratch directory.
        if [[ -d "$REPO/docs" ]]; then OUT="$REPO/docs"; else OUT="$HERE/build/captures"; fi
        mkdir -p "$OUT"
        echo "==> capturing picker"
        "$APP/Contents/MacOS/$BIN_NAME" --snapshot "$OUT/picker.png" \
            --model "$MODEL" --tokens "$TOKENS"
        echo "==> capturing analytics (drives $REQUESTS real ${TOKENS}-token runs on $MODEL)"
        "$APP/Contents/MacOS/$BIN_NAME" --snapshot-analytics "$OUT/analytics.png" \
            --model "$MODEL" --tokens "$TOKENS" --requests "$REQUESTS"
        # The tool trace, driven by the file tools so the capture stays offline
        # and repeatable: a directory listing and a read of a source file.
        echo "==> capturing tools (file tools against $REPO)"
        # A reasoning model spends a lot of the budget thinking before it calls
        # anything: at 900 tokens Qwen3 1.7B never reached the tool call.
        "$APP/Contents/MacOS/$BIN_NAME" --snapshot-analytics "$OUT/tools.png" \
            --model "$TOOLS_MODEL" --tokens 2600 --requests 1 \
            --prompt "Use list_directory on $REPO, then read $REPO/Package.swift and tell me the package name and its targets."
        ls -l "$OUT"
        ;;
    *)
        echo "usage: $0 [build|debug|run|menubar|capture]" >&2
        exit 2
        ;;
esac
