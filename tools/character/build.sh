#!/bin/zsh
# Regenerates both boxers with Blender + MPFB and optimises them into
# public/models.
#   BLENDER   Blender executable or launcher script (default: the author's
#             SSD-local launcher that keeps Blender's data off the system disk)
#   OUT_DIR   where raw exports go (default: $TMPDIR/boxer-out)
set -e
cd "$(dirname "$0")/../.."
BLENDER=${BLENDER:-/Volumes/WorkPlace/Tools/BlenderBoxing/blender.sh}
OUT_DIR=${OUT_DIR:-${TMPDIR:-/tmp}/boxer-out}
mkdir -p "$OUT_DIR"
for p in player opponent; do
  "$BLENDER" --background --python tools/character/gen_boxer.py -- $p "$OUT_DIR/$p.glb" 2>&1 | grep -E "gen_boxer|Error|Traceback" || true
  node tools/character/optimize.mjs "$OUT_DIR/$p.glb" public/models/boxer_$p.glb | tail -1
done
