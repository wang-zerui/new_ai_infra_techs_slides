#!/usr/bin/env bash
# Rebuild every deck: re-render SVG figures and charts, then slide-markdown -> PPTX + PDF.
# Usage: bash scripts/build_all.sh [slug ...]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLIDES="$HOME/.claude/skills/usenix-talk-slides"
FIGS="$HOME/.claude/skills/ipads-arch-figures"
PY="$HOME/.claude/skills/.venv-charts/bin/python"
cd "$REPO/decks"
DECKS=("$@"); [ ${#DECKS[@]} -eq 0 ] && DECKS=($(ls -d */ | tr -d /))
for d in "${DECKS[@]}"; do
  echo "== $d"
  for svg in "$d"/figures/*.svg; do [ -e "$svg" ] && bash "$FIGS/scripts/render.sh" "$svg" >/dev/null; done
  for py in "$d"/charts/*.py; do [ -e "$py" ] && (cd "$d/charts" && "$PY" "$(basename "$py")" >/dev/null); done
  (cd "$d" && bash "$SLIDES/scripts/preview.sh" "$d.md" >/dev/null)
done
