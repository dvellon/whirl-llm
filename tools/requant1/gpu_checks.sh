#!/usr/bin/env bash
# REQUANT-1: whirl selftest + seqtest on the 8060S for one model file, given modes. Usage: gpu_checks.sh MODEL TAG "precise balance fast" "precise balance fast"
set -uo pipefail
M=$1; TAG=$2; STM=$3; SQM=$4
W=/c/ClaudeWorkspace/whirl-work/wt_rel020f/build/Release/whirl.exe
L=/c/ClaudeWorkspace/models/requant1/logs/gpuchk; mkdir -p $L
for m in $STM; do
  $W selftest "$M" --$m --device 8060s > $L/${TAG}_selftest_$m.log 2>&1; rc=$?
  echo "[$(date +%H:%M:%S)] $TAG selftest $m rc=$rc $(grep -E '^selftest:' $L/${TAG}_selftest_$m.log | head -1) dev=$(grep -o -m1 'gfx1[12][05][1]' $L/${TAG}_selftest_$m.log)"
done
for m in $SQM; do
  $W seqtest "$M" --decode 24 --$m --device 8060s > $L/${TAG}_seqtest_$m.log 2>&1; rc=$?
  echo "[$(date +%H:%M:%S)] $TAG seqtest $m rc=$rc $(grep -E '^seqtest:' $L/${TAG}_seqtest_$m.log | head -1) dev=$(grep -o -m1 'gfx1[12][05][1]' $L/${TAG}_seqtest_$m.log)"
done
