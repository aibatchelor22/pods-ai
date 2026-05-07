@echo off
REM Copyright (c) PODS-AI contributors
REM SPDX-License-Identifier: MIT
REM
REM Patch fastai_audio for Python 3.11+ compatibility
REM Based on https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/patch_fastai_audio.sh

echo Applying Python 3.11 compatibility patch to fastai_audio...

REM Find the site-packages directory
for /f "delims=" %%i in ('python -c "import sys; import site; print([p for p in sys.path if 'site-packages' in p][0])"') do set SITE_PACKAGES=%%i

if "%SITE_PACKAGES%"=="" (
    echo Error: Could not find site-packages directory
    exit /b 1
)

echo Site-packages directory: %SITE_PACKAGES%

REM Apply the patch to audio/data.py
set AUDIO_DATA=%SITE_PACKAGES%\audio\data.py

if not exist "%AUDIO_DATA%" (
    echo Error: Could not find audio/data.py at %AUDIO_DATA%
    echo Checking if audio package is installed...
    python -c "import audio" 2>nul || echo audio package not found
    exit /b 1
)

echo Found audio/data.py at: %AUDIO_DATA%

REM Create a backup
copy "%AUDIO_DATA%" "%AUDIO_DATA%.bak" >nul

echo Patching %AUDIO_DATA%...

REM Apply the fix using Python to do replacements
REM Change: from dataclasses import dataclass, asdict
REM To: from dataclasses import dataclass, asdict, field
REM Change: sg_cfg: SpectrogramConfig = SpectrogramConfig()
REM To: sg_cfg: SpectrogramConfig = field(default_factory=SpectrogramConfig)

python -c "import sys; content = open(r'%AUDIO_DATA%', 'r', encoding='utf-8').read(); content = content.replace('from dataclasses import dataclass, asdict', 'from dataclasses import dataclass, asdict, field').replace('sg_cfg: SpectrogramConfig = SpectrogramConfig()', 'sg_cfg: SpectrogramConfig = field(default_factory=SpectrogramConfig)'); open(r'%AUDIO_DATA%', 'w', encoding='utf-8').write(content)"

if %ERRORLEVEL% neq 0 (
    echo Error: Failed to apply patch
    exit /b 1
)

echo Patch applied successfully!
echo Backup saved to: %AUDIO_DATA%.bak

echo Verifying patch was applied...
findstr /C:"field(default_factory=SpectrogramConfig)" "%AUDIO_DATA%" >nul
if %ERRORLEVEL% neq 0 (
    echo Warning: Patch verification failed - could not find expected patched code
    exit /b 1
)

echo Patch verification successful: field(default_factory=SpectrogramConfig) found
exit /b 0
