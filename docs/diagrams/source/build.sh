#!/usr/bin/env bash
# Render every *.mjs diagram spec here into ../<name>.png (PNG via headless Chrome) and <name>.excalidraw (editable).
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
command -v google-chrome >/dev/null 2>&1 && CHROME="$(command -v google-chrome)"
[ -d node_modules ] || npm install --no-audit --no-fund
OUT="$(mktemp -d)"
for spec in *.mjs; do
  [ "$spec" = render.mjs ] && continue
  name="${spec%.mjs}"
  node render.mjs "$spec" "$OUT"
  W=$(grep -o 'W: [0-9]*' "$spec" | head -1 | cut -d' ' -f2); H=$(grep -o 'H: [0-9]*' "$spec" | head -1 | cut -d' ' -f2)
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 --window-size="$W,$H" \
    --virtual-time-budget=3000 --screenshot="../$name.png" "file://$OUT/$name.html" >/dev/null 2>&1
  cp "$OUT/$name.excalidraw" .
done
cp ../architecture.png ../../architecture.png
rm -rf "$OUT"
echo "diagrams rendered into docs/diagrams/"
