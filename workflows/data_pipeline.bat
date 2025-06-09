@echo off
echo =======================================================
echo           Stock Prediction ML Data Pipeline
echo =======================================================
echo.

echo [1/4] Initializing database...
python scripts\initialize_database.py
if errorlevel 1 (
    echo ERROR: Database initialization failed
    pause
    exit /b 1
)
echo.

echo [2/4] Collecting price data...
python scripts\collect_price_data.py --batch-size 10
if errorlevel 1 (
    echo ERROR: Price data collection failed
    pause
    exit /b 1
)
echo.

echo [3/4] Cleaning up symbols...
python scripts\cleanup_symbols.py
if errorlevel 1 (
    echo ERROR: Symbol cleanup failed
    pause
    exit /b 1
)
echo.

echo [4/4] Monitoring database...
python scripts\monitor_database.py
echo.

echo =======================================================
echo           Data Pipeline Complete!
echo =======================================================
pause