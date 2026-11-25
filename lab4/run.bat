@echo off
for %%N in (256 512 1024 2048) do (
    echo === N = %%N ===
    powershell -Command "(Get-Content gen.py) -replace '^N = .*', 'N = %%N' | Set-Content gen.py"
    py gen.py
    lab4_cuda.exe naive 16 16
    lab4_cuda.exe tiled 8 8
    lab4_cuda.exe tiled 16 16
    lab4_cuda.exe tiled 32 32
)
