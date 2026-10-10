#!/usr/bin/env bash
# Build docs/presentation.pdf, slide images and docs/presentation.pptx from deck.html.
# Needs Google Chrome (headless) and python-pptx (pip install python-pptx==1.0.2).
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
command -v google-chrome >/dev/null 2>&1 && CHROME="$(command -v google-chrome)"
PY="${PYTHON:-python3}"

"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=5000 \
  --print-to-pdf=../presentation.pdf "file://$PWD/deck.html" >/dev/null 2>&1
mkdir -p png
for i in 1 2 3 4 5 6 7; do
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 --window-size=1280,720 \
    --virtual-time-budget=4000 --screenshot="png/slide$i.png" "file://$PWD/deck.html?slide=$i" >/dev/null 2>&1
done
"$PY" make_pptx.py
echo "built ../presentation.pdf, ../presentation.pptx, png/slide1-7.png"
