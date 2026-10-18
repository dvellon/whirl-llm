@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\ClaudeWorkspace\tools\llamacpp_mxq
cl /nologo /O2 /MD /Iggml\include /Iggml\src fnmx_selftest\selftest_mxfp4.c build\ggml\src\ggml-base.lib /Fe:build\bin\selftest_mxfp4.exe /Fo:build\ || exit /b 1
