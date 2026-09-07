@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PERL=C:\Strawberry\perl\bin\perl.exe"
set "TEXCOUNT=C:\Users\bonio\AppData\Local\Programs\MiKTeX\scripts\texcount\texcount.pl"
set "DEL3_SECTIONS=C:\Users\bonio\dev\patchAliasing\coursework\deliverable3\sections"
set "DEL3_FILES=summary.tex introduction.tex one.tex two.tex three.tex four.tex five.tex six.tex seven.tex eight.tex nine.tex ten.tex conclusion.tex ontology_appendix.tex bayesian_appendix.tex lighttsmixup_appendix.tex kernelsynth_appendix.tex retraining_appendix.tex sweep_appendix.tex"
set "COURSEWORK_FILES=introduction.tex one.tex two.tex three.tex four.tex five.tex six.tex seven.tex eight.tex nine.tex ten.tex conclusion.tex"

set "LC_ALL="
set "LC_CTYPE="
set "LANG="

if /i not "%~1"=="del3" goto usage
if not "%~2"=="" goto usage

if not exist "%PERL%" (
  echo error: perl not found
  exit /b 1
)

if not exist "%TEXCOUNT%" (
  echo error: texcount not found
  exit /b 1
)

if not exist "%DEL3_SECTIONS%\." (
  echo error: sections directory not found
  exit /b 1
)

pushd "%DEL3_SECTIONS%" >nul

for %%F in (%DEL3_FILES% %COURSEWORK_FILES%) do (
  if not exist "%%F" (
    popd
    echo error: %%F not found
    exit /b 1
  )
)

set "OUTPUT_FILE=%TEMP%\sumtexcount-!RANDOM!-!RANDOM!.tmp"
"%PERL%" -X "%TEXCOUNT%" -utf8 -sum -template={TITLE}:{SUM}\n %DEL3_FILES% >"!OUTPUT_FILE!"
if errorlevel 1 (
  del /q "!OUTPUT_FILE!" >nul 2>&1
  popd
  echo error: texcount failed
  exit /b 1
)

set "TOTAL="
for /f "usebackq tokens=1-3 delims=:" %%A in ("!OUTPUT_FILE!") do (
  if /i "%%A"=="File" echo File:%%B:%%C
  if /i "%%A"=="Total" set "TOTAL=%%B"
)

del /q "!OUTPUT_FILE!" >nul 2>&1

if not defined TOTAL (
  popd
  echo error: texcount failed
  exit /b 1
)

"%PERL%" -X "%TEXCOUNT%" -utf8 -sum -1 %COURSEWORK_FILES% >"!OUTPUT_FILE!"
if errorlevel 1 (
  del /q "!OUTPUT_FILE!" >nul 2>&1
  popd
  echo error: coursework subtotal failed
  exit /b 1
)

set "COURSEWORK="
set /p "COURSEWORK="<"!OUTPUT_FILE!"
del /q "!OUTPUT_FILE!" >nul 2>&1

if not defined COURSEWORK (
  popd
  echo error: coursework subtotal failed
  exit /b 1
)

echo Coursework:!COURSEWORK!
echo Total:!TOTAL!
popd
exit /b 0

:usage
echo usage: sumtexcount del3
exit /b 2
