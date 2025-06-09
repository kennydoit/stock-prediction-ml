#!/usr/bin/env python3
"""
Enhanced database manager with model tracking
"""
import sqlite3
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDatabaseManager:
    """Enhanced database manager with model tracking capabilities"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = Path(__file__).parent / 'enhanced_stock_data.db'
        
        self.db_path = db_path
        self.connection = None
    
    def __enter__(self):
        self.connection = sqlite3.connect(self.db_path, timeout=30.0)
        self.connection.execute("PRAGMA foreign_keys = ON")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
            self.connection.close()
    
    def setup_enhanced_schema(self):
        """Setup enhanced database schema"""
        
        # Model tracking table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                target_symbol TEXT NOT NULL,
                model_type TEXT NOT NULL,
                train_r2 REAL,
                test_r2 REAL,
                train_mse REAL,
                test_mse REAL,
                n_features INTEGER,
                feature_selection_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                config_snapshot TEXT,
                notes TEXT
            )
        """)
        
        # Feature importance tracking
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS feature_importance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                coefficient REAL,
                abs_coefficient REAL,
                p_value REAL,
                rank_importance INTEGER,
                FOREIGN KEY (run_id) REFERENCES model_runs (run_id)
            )
        """)
        
        # Prediction tracking
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                prediction_date DATE NOT NULL,
                actual_return REAL,
                predicted_return REAL,
                residual REAL,
                data_type TEXT CHECK (data_type IN ('train', 'test', 'validation')),
                FOREIGN KEY (run_id) REFERENCES model_runs (run_id)
            )
        """)
        
        # Create indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_model_runs_symbol ON model_runs(target_symbol)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_model_runs_created ON model_runs(created_at)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_feature_importance_run ON feature_importance(run_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_predictions_run_date ON model_predictions(run_id, prediction_date)")
        
        logger.info("✅ Enhanced database schema created")
        print("✅ Enhanced database schema created")
    
    def save_model_run(self, model_package, metrics, feature_stats=None):
        """Save model run results to database"""
        
        run_id = str(uuid.uuid4())
        
        # Save model run metadata
        self.connection.execute("""
            INSERT INTO model_runs (
                run_id, target_symbol, model_type, train_r2, test_r2,
                train_mse, test_mse, n_features, feature_selection_method,
                config_snapshot, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            model_package['target_symbol'],
            model_package.get('model_type', 'Linear'),
            metrics.get('train_r2'),
            metrics.get('test_r2'),
            metrics.get('train_mse'),
            metrics.get('test_mse'),
            len(model_package['feature_names']),
            'SelectKBest',
            json.dumps(model_package.get('config', {})),
            f"Enhanced model training run"
        ))
        
        # Save feature importance if available
        if feature_stats is not None:
            for idx, (_, row) in enumerate(feature_stats.iterrows()):
                self.connection.execute("""
                    INSERT INTO feature_importance (
                        run_id, feature_name, coefficient, abs_coefficient,
                        p_value, rank_importance
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    row['feature'],
                    row.get('coefficient'),
                    row.get('abs_coefficient'),
                    row.get('p_value'),
                    idx + 1
                ))
        
        logger.info(f"✅ Model run {run_id} saved to database")
        return run_id
    
    def get_model_performance_history(self, target_symbol=None):
        """Get model performance history"""
        
        query = """
            SELECT run_id, target_symbol, model_type, train_r2, test_r2,
                   train_mse, test_mse, n_features, created_at
            FROM model_runs
        """
        
        params = []
        if target_symbol:
            query += " WHERE target_symbol = ?"
            params.append(target_symbol)
        
        query += " ORDER BY created_at DESC"
        
        return pd.read_sql_query(query, self.connection, params=params)
    
    def get_best_model_run(self, target_symbol):
        """Get the best performing model run for a symbol"""
        
        query = """
            SELECT * FROM model_runs 
            WHERE target_symbol = ? AND test_r2 > 0
            ORDER BY test_r2 DESC
            LIMIT 1
        """
        
        result = pd.read_sql_query(query, self.connection, params=[target_symbol])
        return result.iloc[0] if not result.empty else None

def main():
    """Setup enhanced database when run directly"""
    print("🗄️ Setting up Enhanced Database Schema")
    print("="*50)
    
    with EnhancedDatabaseManager() as db:
        db.setup_enhanced_schema()
    
    print("✅ Enhanced database setup complete!")

if __name__ == "__main__":
    main()