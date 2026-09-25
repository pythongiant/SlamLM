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
# The app is the repository root in this checkout.
REPO="$HERE"
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
TOKENS="${SLAM_LM_SNAPSHOT_TOKENS:-96}"
REQUESTS="${SLAM_LM_SNAPSHOT_REQUESTS:-3}"

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
        OUT="$HERE/build/captures"
        mkdir -p "$OUT"
        echo "==> capturing picker"
        "$APP/Contents/MacOS/$BIN_NAME" --snapshot "$OUT/picker.png" \
            --model "$MODEL" --tokens "$TOKENS"
        echo "==> capturing analytics (drives $REQUESTS real ${TOKENS}-token runs on $MODEL)"
        "$APP/Contents/MacOS/$BIN_NAME" --snapshot-analytics "$OUT/analytics.png" \
            --model "$MODEL" --tokens "$TOKENS" --requests "$REQUESTS"
        ls -l "$OUT"
        ;;
    *)
        echo "usage: $0 [build|debug|run|menubar|capture]" >&2
        exit 2
        ;;
esac
