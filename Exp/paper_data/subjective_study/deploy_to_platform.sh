#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Deploy generated MP4 stimuli to the user study platform
# ============================================================

BASE="/Users/caifeifan/VibeScholar/Antigravity_OFR/projects/007-waveletmask/Exp/paper_data/subjective_study"
PLATFORM="/Users/caifeifan/VibeScholar/Antigravity_OFR/papers/shuphd-ofr-latex/_workspace/06-user-study"
TARGET="$PLATFORM/public/videos/waveletmask"
SOURCE="${1:-$BASE/stimuli_videos_v2}"

CLIPS=(001 005 008 009 010 012 014 018 020 022 024 029)
METHODS=(lq rtn rrtn mambaofr waveletmask gt)

echo "=== Deploying stimuli to user study platform ==="
echo "Source: $SOURCE/"
echo "Target: $TARGET/"
echo ""

mkdir -p "$TARGET"

for clip in "${CLIPS[@]}"; do
  mkdir -p "$TARGET/clip_$clip"
  for method in "${METHODS[@]}"; do
    src="$SOURCE/clip_$clip/$method.mp4"
    dst="$TARGET/clip_$clip/$method.mp4"
    if [ -f "$src" ]; then
      cp "$src" "$dst"
      echo "  ✓ clip_$clip/$method.mp4"
    else
      echo "  ✗ MISSING: $src"
    fi
  done
done

echo ""
echo "=== Deployment complete ==="
echo "Total files: $(find "$TARGET" -name "*.mp4" | wc -l | tr -d ' ')"
echo ""
echo "Next: cd $PLATFORM && npm run dev"
