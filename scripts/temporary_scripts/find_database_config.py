# Create file: scripts/temporary_scripts/check_config_files.py
"""
Check and update config files to use correct database
"""

import sys
from pathlib import Path
import yaml

def check_config_files():
    """Check config files and update database settings"""
    
    print("🔍 Checking Config Files")
    print("="*25)
    
    project_root = Path(__file__).parent.parent.parent
    
    # Check config.yaml
    config_file = project_root / 'config.yaml'
    if config_file.exists():
        print(f"📄 Reading config.yaml: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                config_content = f.read()
            
            print(f"\n📋 Current config.yaml content:")
            print("-" * 40)
            print(config_content)
            print("-" * 40)
            
            # Try to parse as YAML
            try:
                config_data = yaml.safe_load(config_content)
                print(f"\n📊 Parsed YAML structure:")
                print(f"Keys: {list(config_data.keys()) if isinstance(config_data, dict) else 'Not a dict'}")
                
                # Look for database settings
                if isinstance(config_data, dict):
                    for key, value in config_data.items():
                        if 'db' in key.lower() or 'database' in key.lower():
                            print(f"  🔍 {key}: {value}")
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if 'db' in subkey.lower() or 'database' in subkey.lower():
                                    print(f"  🔍 {key}.{subkey}: {subvalue}")
                
            except yaml.YAMLError as e:
                print(f"⚠️ YAML parsing error: {e}")
                
        except Exception as e:
            print(f"❌ Error reading config.yaml: {e}")
    
    # Check .env file
    env_file = project_root / '.env'
    if env_file.exists():
        print(f"\n📄 Reading .env: {env_file}")
        
        try:
            with open(env_file, 'r') as f:
                env_content = f.read()
            
            print(f"\n📋 Current .env content:")
            print("-" * 40)
            print(env_content)
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Error reading .env: {e}")

def update_config_for_correct_database():
    """Update config to use historical_stock_data.db"""
    
    print(f"\n🔧 Updating Database Configuration")
    print("="*35)
    
    project_root = Path(__file__).parent.parent.parent
    config_file = project_root / 'config.yaml'
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_content = f.read()
            
            # Check if it needs updating
            if 'stock_prediction.db' in config_content:
                print(f"🔄 Found 'stock_prediction.db' in config - updating to 'historical_stock_data.db'")
                
                # Create backup
                backup_file = config_file.with_suffix('.yaml.backup')
                with open(backup_file, 'w') as f:
                    f.write(config_content)
                print(f"💾 Backup created: {backup_file}")
                
                # Update content
                updated_content = config_content.replace('stock_prediction.db', 'historical_stock_data.db')
                
                with open(config_file, 'w') as f:
                    f.write(updated_content)
                
                print(f"✅ Config updated successfully!")
                print(f"\n📋 Updated content:")
                print("-" * 40)
                print(updated_content)
                print("-" * 40)
                
            elif 'historical_stock_data.db' in config_content:
                print(f"✅ Config already uses 'historical_stock_data.db'")
                
            else:
                print(f"⚠️ No database filename found in config")
                print(f"💡 You may need to add database configuration")
                
        except Exception as e:
            print(f"❌ Error updating config: {e}")
    
    # Also check/update .env if it has database settings
    env_file = project_root / '.env'
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                env_content = f.read()
            
            if 'stock_prediction.db' in env_content:
                print(f"\n🔄 Found 'stock_prediction.db' in .env - updating...")
                
                # Create backup
                backup_file = env_file.with_suffix('.env.backup')
                with open(backup_file, 'w') as f:
                    f.write(env_content)
                
                # Update content
                updated_env = env_content.replace('stock_prediction.db', 'historical_stock_data.db')
                
                with open(env_file, 'w') as f:
                    f.write(updated_env)
                
                print(f"✅ .env updated successfully!")
                
        except Exception as e:
            print(f"❌ Error updating .env: {e}")

if __name__ == "__main__":
    check_config_files()
    
    # Ask user if they want to update
    response = input("\n🔧 Update config to use 'historical_stock_data.db'? (y/N): ")
    if response.lower() == 'y':
        update_config_for_correct_database()
    else:
        print("⏭️ Skipping config update")