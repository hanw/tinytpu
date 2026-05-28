#!/usr/bin/env bash
# Builds libsim.so for the VPU bscwave testbench.
#
#   1. bsc compiles VPUWrapper.bsv (with -p so it picks up tinytpu/src/*.bsv
#      and the BSV stdlib) into the mkVPU_S1L4 model + sim.so.so.
#   2. g++ links the bscwave wrapper against the bsc-generated objects.
#   3. bscwave-gen-ports parses the emitted mkVPU_S1L4.cxx to produce a
#      typed Inputs/Outputs Haskell record.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TINYTPU="$(cd "$HERE/../../.." && pwd)"
BSCWAVE="$(cd "$HERE/../../../../bscwave" && pwd)"

if [ -z "${BLUESPECDIR:-}" ]; then
  BLUESPECDIR="$(dirname "$(dirname "$(command -v bsc)")")/lib"
fi
BSC_INC="$BLUESPECDIR/Bluesim"

WRAPPER="$BSCWAVE/csrc/bsim_wrapper.cxx"

cd "$HERE/src"

# 1. bsc compile + link to Bluesim runtime.
#    -p searches tinytpu/src/ for VPU.bsv and its FpReducer/TranscUnit deps,
#    then the standard Bluespec library paths.
#    +RTS -K256M matches the parent Makefile — VPU elaboration recurses
#    deeply through the per-op case block and blows the default stack.
#    -bdir and -simdir keep .bo/.ba/.o droppings here in src/ rather than
#    polluting tinytpu/src/ next to the discovered .bsv dependencies.
BSC="bsc +RTS -K256M -RTS"
$BSC -sim -p "$TINYTPU/src:+" -bdir . -simdir . -g mkVPU_S1L4 -u VPUWrapper.bsv
$BSC -sim -p "$TINYTPU/src:+" -bdir . -simdir . -e mkVPU_S1L4 -o sim.so mkVPU_S1L4.ba

# 2. Build libsim.so = bscwave wrapper + model objects.
g++ -shared -fPIC \
  -I"$BSC_INC" \
  "$WRAPPER" \
  model_mkVPU_S1L4.o mkVPU_S1L4.o \
  -L. -l:sim.so.so -ldl \
  -Wl,-rpath,'$ORIGIN' \
  -o libsim.so

cd "$HERE"
ln -sf src/libsim.so libsim.so

# 3. Generate the typed port record from the bsc-emitted .cxx so the
#    Haskell testbench can reference inputs/outputs by name with widths
#    checked at compile time.
(cd "$BSCWAVE" && cabal --project-dir=. run -v0 bscwave-gen-ports -- \
   "$HERE/src/mkVPU_S1L4.cxx" \
   -o "$HERE/app/MkVPU_S1L4.hs")

echo
echo "Built. Now run:   LD_LIBRARY_PATH=./src cabal run testbench"
