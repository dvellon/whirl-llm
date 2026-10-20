@echo off
rem Stand-in for whirl-server-gate.exe used by golden_selftest.ps1 (no GPU). SPDX-License-Identifier: Apache-2.0
rem Called as: --exe SERVER --kind whirl --model MODEL --suite golden ... --server-env WHIRL_MODE=MODE ...
rem FAKEGATE_FAIL=1: one in-run check fails (exit 1); FAKEGATE_NOPASS=1: exit 0 without the ALL PASS line;
rem FAKEGATE_H=HEX: the hash of srv.sessA.t2.ssd (default 1111111111111111).
if not defined FAKEGATE_H set FAKEGATE_H=1111111111111111
echo G header format=1 suite=server
echo G srv.short status=200 n=48 prompt=40 cached=0 finish=length h=aaaaaaaaaaaaaaaa
echo G srv.sessA.t2.ram status=200 n=32 prompt=3100 cached=3050 finish=length h=1111111111111111
echo G srv.sessA.t2.ssd status=200 n=32 prompt=3100 cached=3050 finish=length h=%FAKEGATE_H%
if "%FAKEGATE_FAIL%"=="1" (
  echo   [FAIL] golden: turn 2 after the SSD restore == after the RAM restore
  echo == whirl-server-gate: 3 passed, 1 failed
  exit /b 1
)
echo   [PASS] golden: turn 2 after the SSD restore == after the RAM restore
if "%FAKEGATE_NOPASS%"=="1" exit /b 0
echo == whirl-server-gate: 4 passed, 0 failed (ALL PASS)
exit /b 0
