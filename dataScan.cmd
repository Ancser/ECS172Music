@echo off
setlocal
pushd "%~dp0"
python ".\dataScan.py" %*
popd
endlocal
