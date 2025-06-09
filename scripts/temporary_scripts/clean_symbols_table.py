# Create file: scripts/temporary_scripts/clean_symbols_table.py
"""
Clean invalid entries from symbols table
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'database'))

def clean_symbols_table():
    """Remove invalid entries from symbols table"""
    
    print("🧹 Cleaning Symbols Table")
    print("="*25)
    
    try:
        from database_manager import DatabaseManager
        
        with DatabaseManager() as db:
            if hasattr(db, 'connection'):
                cursor = db.connection.cursor()
                conn = db.connection
            elif hasattr(db, 'conn'):
                cursor = db.conn.cursor()
                conn = db.conn
            else:
                print("❌ Cannot access database connection")
                return
            
            # Count current entries
            cursor.execute("SELECT COUNT(*) FROM symbols;")
            total_before = cursor.fetchone()[0]
            print(f"📊 Symbols before cleaning: {total_before}")
            
            # Show some invalid entries
            cursor.execute("SELECT Symbol FROM symbols WHERE Symbol LIKE '#%' OR LENGTH(Symbol) > 5 LIMIT 5;")
            invalid_samples = cursor.fetchall()
            if invalid_samples:
                print(f"🗑️ Sample invalid entries: {[row[0] for row in invalid_samples]}")
            
            # Delete invalid entries
            cursor.execute("""
                DELETE FROM symbols 
                WHERE Symbol LIKE '#%' 
                   OR LENGTH(Symbol) > 5 
                   OR Symbol NOT GLOB '[A-Z]*'
            """)
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            # Count after cleaning
            cursor.execute("SELECT COUNT(*) FROM symbols;")
            total_after = cursor.fetchone()[0]
            
            print(f"🗑️ Deleted {deleted_count} invalid entries")
            print(f"📊 Symbols after cleaning: {total_after}")
            
            # Show sample of remaining symbols
            cursor.execute("SELECT Symbol FROM symbols LIMIT 10;")
            valid_samples = [row[0] for row in cursor.fetchall()]
            print(f"✅ Sample valid symbols: {valid_samples}")
            
    except Exception as e:
        print(f"❌ Error cleaning symbols: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    clean_symbols_table()