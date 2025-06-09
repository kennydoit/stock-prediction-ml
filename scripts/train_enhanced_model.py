#!/usr/bin/env python3
"""
Train model with enhanced features and peer option
"""
import sys
from pathlib import Path
import yaml
import pandas as pd
import joblib
from datetime import datetime
import logging
import argparse
import statsmodels.api as sm

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from features.enhanced_technical_indicators import calculate_enhanced_features
from models.linear_regression import train_linear_regression_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_enhanced_model(target_symbol: str, include_peers: bool = False, save_model: bool = True, audit: bool = False):
    """Train model with enhanced features"""
    model_type = "with_peers" if include_peers else "target_only"
    print(f"🤖 Training Enhanced Model ({model_type})")
    print("="*60)
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override target symbol with parameter
    config['target_symbol'] = target_symbol
    
    print(f"Target symbol: {target_symbol}")
    print(f"Include peers: {include_peers}")
    
    if include_peers:
        peer_symbols = config.get('peer_symbols', [])
        print(f"Peer symbols: {len(peer_symbols)} symbols")
    
    # Generate enhanced features
    print(f"\n📊 Generating enhanced features...")
    features_df = calculate_enhanced_features(target_symbol, config, include_peers=include_peers)
    
    if features_df.empty:
        print("❌ No features generated")
        return None

    # After features_df is created in train_enhanced_model
    if 'close' in features_df.columns:
        features_df['next_return'] = features_df['close'].pct_change().shift(-1)
        features_df = features_df.dropna(subset=['next_return'])
    else:
        print("❌ 'close' column not found in features_df; cannot compute next_return.")

    if audit:
        audit_dir = Path(__file__).parent.parent / 'data' / 'audit'
        audit_dir.mkdir(parents=True, exist_ok=True)
        features_path = audit_dir / f"{target_symbol}_features.csv"
        target_path = audit_dir / f"{target_symbol}_target.csv"
        # Save features (X)
        features_df.drop(columns=['next_return'], errors='ignore').to_csv(features_path)
        # Save target (Y)
        if 'next_return' in features_df.columns:
            features_df[['next_return']].to_csv(target_path)
        print(f"✅ Audit: Saved features to {features_path}")
        print(f"✅ Audit: Saved target to {target_path}")

    print(f"✅ Generated {features_df.shape[0]} feature records")
    print(f"Features: {features_df.shape[1]} indicators")
    print(f"Date range: {features_df.index.min()} to {features_df.index.max()}")
    
    # Show feature breakdown
    feature_types = {
        'Technical': len([c for c in features_df.columns if any(x in c for x in ['rsi', 'macd', 'bb_', 'cci', 'stoch', 'atr', 'obv', 'ichimoku', 'sma'])]),
        'Returns/Vol': len([c for c in features_df.columns if any(x in c for x in ['returns', 'volatility', 'volume'])]),
        'Lagged': len([c for c in features_df.columns if 'lag_' in c or 'roll_' in c]),
        'Peer': len([c for c in features_df.columns if any(x in c for x in ['rel_', 'avg_rel'])])
    }
    
    print(f"\nFeature breakdown:")
    for feat_type, count in feature_types.items():
        if count > 0:
            print(f"  {feat_type}: {count}")
    
    # Drop target and raw price columns before model training
    drop_cols = ['next_return', 'close', 'open', 'high', 'low', 'volume']
    features_for_training = features_df.drop(columns=[col for col in drop_cols if col in features_df.columns], errors='ignore')

    # Train model
    print(f"\n🏋️ Training enhanced linear regression model...")
    model_result = train_linear_regression_model(features_for_training, config)
    
    if not model_result:
        print("❌ Model training failed")
        return None
    
    model, scaler, feature_names, metrics = model_result
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Performance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Print regression p-values
    if 'next_return' in features_df.columns:
        print_regression_pvalues(features_df, feature_names, 'next_return')
    else:
        print("⚠️ Could not compute p-values: 'next_return' column not found in features.")
    
    # Save model to new location
    if save_model:
        print(f"\n💾 Saving enhanced model...")
        # New path: data/model_outputs
        models_dir = Path(__file__).parent.parent / 'data' / 'model_outputs'
        models_dir.mkdir(parents=True, exist_ok=True)  # Creates both data and model_outputs if needed
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f'enhanced_linear_regression_{target_symbol}_{model_type}_{timestamp}.pkl'
        model_path = models_dir / model_name
        
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'target_symbol': target_symbol,
            'metrics': metrics,
            'config': config,
            'trained_at': timestamp,
            'feature_count': len(feature_names),
            'include_peers': include_peers,
            'model_type': model_type,
            'feature_breakdown': feature_types,
            'data_range': {
                'start': str(features_df.index.min()),
                'end': str(features_df.index.max()),
                'records': len(features_df)
            }
        }
        
        joblib.dump(model_package, model_path)
        
        print(f"✅ Model saved: {model_path}")
        
        # Quick prediction test
        print(f"\n🔮 Testing predictions...")
        latest_features = features_df.iloc[-1:][feature_names]
        latest_features_scaled = scaler.transform(latest_features)
        prediction = model.predict(latest_features_scaled)[0]
        
        print(f"Latest prediction for {target_symbol}: {prediction:.4f} (return)")
        print(f"Date: {features_df.index[-1]}")
        
        return model_path
    
    return model_result

def print_regression_pvalues(features_df, feature_names, target_col):
    """Print p-values for each feature using statsmodels OLS."""
    X = features_df[feature_names]
    y = features_df[target_col]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print("\n📊 Regression Coefficient P-values:")
    for fname, pval in zip(['Intercept'] + feature_names, model.pvalues):
        print(f"  {fname:30s}: p-value = {pval:.4g}")
    print("\nSummary:\n", model.summary())

def auto_select_peers(target_symbol: str, max_peers: int = 8) -> list:
    """Auto-select peers using existing validated_symbols.yaml"""
    import yaml
    from pathlib import Path
    
    # Path to your existing file
    symbols_file = Path(r"C:\Users\Kenrm\repositories\stock-symbol-analyzer\data\validated_symbols.yaml")
    
    if not symbols_file.exists():
        print(f"⚠️ Symbols file not found: {symbols_file}")
        return []
    
    with open(symbols_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Adjust this based on your file structure - check what the actual structure is
    symbols = data.get('symbols', data)  # Fallback if 'symbols' key doesn't exist
    
    if target_symbol not in symbols:
        print(f"⚠️ Target symbol {target_symbol} not found in validated symbols")
        return []
    
    target_info = symbols[target_symbol]
    target_sector = target_info.get('sector', '')
    target_industry = target_info.get('industry', '')
    target_market_cap = target_info.get('market_cap', '')
    
    print(f"🎯 Target: {target_symbol}")
    print(f"   Sector: {target_sector}")
    print(f"   Industry: {target_industry}")
    print(f"   Market Cap: {target_market_cap}")
    
    # Categorize available symbols
    same_industry = []
    same_sector = []
    market_etfs = []
    large_cap_tech = []
    
    for symbol, info in symbols.items():
        if symbol == target_symbol:
            continue
            
        symbol_sector = info.get('sector', '')
        symbol_industry = info.get('industry', '')
        symbol_market_cap = info.get('market_cap', '')
        
        # ETFs (always good benchmarks)
        if 'ETF' in symbol_sector or symbol in ['SPY', 'QQQ', 'XLK', 'VTI']:
            market_etfs.append(symbol)
        # Same industry (best peers)
        elif symbol_industry == target_industry and symbol_industry:
            same_industry.append(symbol)
        # Same sector (good peers)
        elif symbol_sector == target_sector and symbol_sector:
            same_sector.append(symbol)
        # Large cap tech (if target is tech)
        elif symbol_sector == 'Technology' and symbol_market_cap == 'Large' and target_sector == 'Technology':
            large_cap_tech.append(symbol)
    
    # Build peer list with priority
    peers = []
    
    # Always include key market benchmarks
    priority_etfs = ['SPY', 'QQQ'] if target_sector == 'Technology' else ['SPY']
    for etf in priority_etfs:
        if etf in market_etfs and len(peers) < max_peers:
            peers.append(etf)
    
    # Add same industry peers (highest priority)
    remaining = max_peers - len(peers)
    industry_count = min(remaining // 2, len(same_industry), 4)  # Max 4 industry peers
    if industry_count > 0:
        peers.extend(same_industry[:industry_count])
    
    # Add same sector peers
    remaining = max_peers - len(peers)
    sector_count = min(remaining, len(same_sector), 3)  # Max 3 sector peers
    if sector_count > 0:
        sector_peers = [s for s in same_sector if s not in peers]
        peers.extend(sector_peers[:sector_count])
    
    # Fill remaining with tech ETF if target is tech
    remaining = max_peers - len(peers)
    if remaining > 0 and target_sector == 'Technology' and 'XLK' in market_etfs and 'XLK' not in peers:
        peers.append('XLK')
    
    print(f"🤖 Auto-selected {len(peers)} peers:")
    for peer in peers:
        peer_info = symbols.get(peer, {})
        print(f"   {peer}: {peer_info.get('sector', 'Unknown')} - {peer_info.get('industry', 'Unknown')}")
    
    return peers

def main():
    """Main function with command line options"""
    
    parser = argparse.ArgumentParser(description='Train enhanced model')
    parser.add_argument('--symbol', default='MSFT', help='Target symbol')
    parser.add_argument('--peers', action='store_true', help='Include peer features')
    parser.add_argument('--auto-peers', action='store_true', help='Auto-select peers')
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save the model (testing only)')
    parser.add_argument('--audit', action='store_true', help='Save training X and Y to data/audit')

    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-select peers if requested
    if args.auto_peers:
        auto_peers = auto_select_peers(args.symbol)
        if auto_peers:
            config['peer_symbols'] = auto_peers
            config['target_symbol'] = args.symbol
            
            # Save updated config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"✅ Updated config.yaml with auto-selected peers")
    
    result = train_enhanced_model(
        target_symbol=args.symbol,  # Pass the symbol directly
        include_peers=args.peers or args.auto_peers,  # Auto-enable peers if auto-selected
        save_model=not args.no_save,
        audit=args.audit
    )
    
    if result:
        print(f"\n🎉 SUCCESS! Enhanced model for {args.symbol} ready!")
    else:
        print(f"\n❌ Training failed for {args.symbol}")

if __name__ == "__main__":
    main()