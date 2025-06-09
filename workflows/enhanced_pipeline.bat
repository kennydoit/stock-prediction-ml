@echo off
echo =======================================================
echo     Enhanced Stock Prediction ML Pipeline
echo =======================================================
echo.

echo [1/5] Setting up enhanced database...
python database\enhanced_db_manager.py
if errorlevel 1 (
    echo ERROR: Enhanced database setup failed
    pause
    exit /b 1
)
echo.

echo [2/5] Training improved model...
python scripts\retrain_model.py
if errorlevel 1 (
    echo ERROR: Model training failed
    pause
    exit /b 1
)
echo.

echo [3/5] Analyzing model performance...
python scripts\analyze_model.py
if errorlevel 1 (
    echo WARNING: Model analysis had issues
)
echo.

echo [4/5] Running complete tracking workflow...
python scripts\train_and_track.py
if errorlevel 1 (
    echo ERROR: Tracking workflow failed
    pause
    exit /b 1
)
echo.

echo [5/5] Database monitoring...
python scripts\monitor_database.py
echo.

echo =======================================================
echo        Enhanced Pipeline Complete!
echo =======================================================
echo Check the generated analysis files and database
echo for detailed model performance metrics.
echo.
pause