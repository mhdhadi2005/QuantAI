@echo off
echo.
echo =========================================
echo   QuantAI - Autonomous Trading System
echo =========================================
echo.

REM Start backend
echo [1/2] Starting FastAPI backend on http://localhost:8000 ...
start "QuantAI Backend" cmd /c "cd /d %~dp0backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 2>&1"

timeout /t 3 /nobreak >nul

REM Start frontend
echo [2/2] Starting React frontend on http://localhost:5173 ...
start "QuantAI Frontend" cmd /c "cd /d %~dp0frontend && npm.cmd run dev 2>&1"

timeout /t 3 /nobreak >nul

echo.
echo =========================================
echo   QuantAI is running!
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5174
echo   API Docs: http://localhost:8000/docs
echo =========================================
echo.
echo Press any key to open the dashboard in your browser...
pause >nul

start http://localhost:5174
