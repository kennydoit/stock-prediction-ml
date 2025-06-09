@echo off
echo =======================================================
echo     Test Database Setup
echo =======================================================
echo.

echo Testing database initialization...
python database\enhanced_db_manager.py
if errorlevel 1 (
    echo ERROR: Database setup failed
    pause
    exit /b 1
) else (
    echo SUCCESS: Database setup completed
)
echo.

echo Testing database connection...
python -c "from database.enhanced_db_manager import EnhancedDatabaseManager; print('Database connection test passed')"
if errorlevel 1 (
    echo ERROR: Database connection failed
    pause
    exit /b 1
) else (
    echo SUCCESS: Database connection works
)
echo.

echo =======================================================
echo     Database Test Complete
echo =======================================================
pause