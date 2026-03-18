@echo off
echo =======================================================
echo          RAG Project Setup and Run Script
echo =======================================================
echo.

echo [1/3] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ❌ Failed to install dependencies. Please ensure Python is installed and in your PATH.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Starting Endee Vector Database using Docker Compose...
cd endee
docker compose up -d
if %errorlevel% neq 0 (
    echo.
    echo ❌ Failed to start Endee via Docker. Please ensure Docker Desktop is running.
    cd ..
    pause
    exit /b %errorlevel%
)
cd ..

echo.
echo [3/3] Starting Streamlit App (This will open your browser)...
streamlit run app.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ Failed to start the Streamlit application.
    pause
    exit /b %errorlevel%
)

pause
