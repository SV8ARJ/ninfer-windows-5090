@echo off
setlocal
pushd "%~dp0"
set "PYTHON=C:\Users\george\.conda\envs\vllm\python.exe"

"%PYTHON%" -m tools.convert.qwen3_8_27b_uncensored.convert_nvfp4 ^
  --model F:\tmp\LLM_Models\ninfer\models\Qwen3.8-27B-Uncensored ^
  --quantized-model F:\tmp\LLM_Models\ninfer\models\Qwen3.8-27B-Uncensored-NVFP4 ^
  --dflash2-model F:\tmp\LLM_Models\ninfer\models\Qwen3.8-27B-DFlash2 ^
  --out out\qwen3_8_27b_uncensored_nvfp4.ninfer
if errorlevel 1 goto :failed

echo Conversion completed successfully.
goto :done

:failed
echo Conversion failed with exit code %errorlevel%.

:done
popd
pause
