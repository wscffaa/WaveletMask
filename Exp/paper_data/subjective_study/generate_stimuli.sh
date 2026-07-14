#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Generate 4-panel MP4 stimuli for WaveletMask subjective study
# 12 clips × 4 methods (lq/rtn/waveletmask/gt) = 48 MP4s
# ============================================================

BASE="/Users/caifeifan/VibeScholar/Antigravity_OFR/projects/007-waveletmask/Exp/paper_data/subjective_study"
STIMULI="$BASE/stimuli"
OUTPUT="$BASE/stimuli_videos"
FRAMES_SCRIPT="/Users/caifeifan/VibeScholar/Antigravity_OFR/papers/shuphd-ofr-latex/_workspace/06-user-study/scripts/frames_to_video.sh"

CLIPS=(001 005 008 009 010 012 014 018 020 022 024 029)
METHODS=(lq rtn waveletmask gt)

echo "=== WaveletMask Subjective Study — Stimuli Generation ==="
echo "Source: $STIMULI"
echo "Output: $OUTPUT"
echo ""

# Verify all frames exist
echo "--- Verification ---"
MISSING=0
for clip in "${CLIPS[@]}"; do
  for method in "${METHODS[@]}"; do
    count=$(ls "$STIMULI/$clip/$method/"*.png 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -lt 48 ]; then
      echo "  WARN: clip $clip $method has only $count frames (need 48)"
      MISSING=$((MISSING + 1))
    fi
  done
done

if [ "$MISSING" -gt 0 ]; then
  echo ""
  echo "WARNING: $MISSING clip/method combinations have < 48 frames."
  echo "Videos will be shorter than 2 seconds for those clips."
  echo ""
fi

# Generate MP4s
echo "--- Generating MP4s ---"
TOTAL=0
for clip in "${CLIPS[@]}"; do
  mkdir -p "$OUTPUT/clip_$clip"
  for method in "${METHODS[@]}"; do
    OUT="$OUTPUT/clip_$clip/$method.mp4"
    if [ -f "$OUT" ]; then
      echo "  SKIP: $OUT already exists"
      continue
    fi
    echo "  Generating clip_$clip/$method.mp4 ..."
    bash "$FRAMES_SCRIPT" "$STIMULI/$clip/$method" "$OUT" 24 2 720 2>&1 | tail -1
    TOTAL=$((TOTAL + 1))
  done
done

echo ""
echo "=== Done: $TOTAL MP4s generated ==="
ls -la "$OUTPUT"/clip_*/*.mp4 | wc -l | tr -d ' '
echo "files total"
