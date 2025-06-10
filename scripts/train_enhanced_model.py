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

def train_enhanced_model(target_symbol: str, include_peers: bool = False, save_model: bool = True, audit: bool = False, selection: str = None, use_fred: bool = False, slentry: float = None, slstay: float = None, auto_peers: bool = False):
    """Train model with enhanced features, optionally including FRED macro data"""
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
    
    # --- FIX: Use auto-selected peers if auto_peers is True ---
    if auto_peers:
        print(f"🔎 Auto-selecting peers for {target_symbol}...")
        peer_symbols = auto_select_peers(target_symbol, config)
        config['peer_symbols'] = peer_symbols
        print(f"Auto-selected {len(peer_symbols)} peers: {peer_symbols}")
        # --- DEBUG: If peer list is empty, dump available universe for troubleshooting ---
        if not peer_symbols:
            print(f"⚠️ Peer list is empty for {target_symbol}. Dumping available universe for debug...")
            try:
                symbols_file = Path(config['validated_symbols_path'])
                if symbols_file.exists():
                    with open(symbols_file, 'r') as f:
                        data = yaml.safe_load(f)
                    audit_dir = Path(__file__).parent.parent / 'data' / 'audit'
                    audit_dir.mkdir(parents=True, exist_ok=True)
                    debug_file = audit_dir / f'peers_debug_{target_symbol}.yaml'
                    with open(debug_file, 'w') as f:
                        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                    print(f"📝 Universe debug saved to {debug_file}")
                else:
                    print(f"⚠️ Universe file not found: {symbols_file}")
            except Exception as e:
                print(f"❌ Error dumping universe for debug: {e}")
    elif include_peers:
        peer_symbols = config.get('peer_symbols', [])
        print(f"Peer symbols: {len(peer_symbols)} symbols")
    
    # Generate enhanced features
    print(f"\n📊 Generating enhanced features...")
    features_df = calculate_enhanced_features(target_symbol, config, include_peers=include_peers)
    # --- AUDIT: Save peer list if audit is enabled and peers are used ---
    if audit and include_peers:
        peer_list = config.get('peer_symbols', [])
        print(f"[AUDIT] Saving peer list for {target_symbol}: {peer_list}")
        audit_dir = Path(__file__).parent.parent / 'data' / 'audit'
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_file = audit_dir / f'peers_{target_symbol}.txt'
        with open(audit_file, 'w') as f:
            for peer in peer_list:
                f.write(f"{peer}\n")
        print(f"📝 Peer list saved to {audit_file}")
    
    if features_df.empty:
        print("❌ No features generated")
        return None

    # --- Target variable construction ---
    target_variable = config.get('target_variable', 'returns')
    target_window = int(config.get('target_window', 1))
    if 'close' in features_df.columns:
        if target_variable == 'returns':
            # Predict return over target_window days
            features_df['target'] = features_df['close'].pct_change(periods=target_window).shift(-target_window)
        elif target_variable == 'price':
            # Predict price in target_window days
            features_df['target'] = features_df['close'].shift(-target_window)
        else:
            print(f"❌ Unknown target_variable: {target_variable}. Must be 'returns' or 'price'.")
            return None
        features_df = features_df.dropna(subset=['target'])
    else:
        print("❌ 'close' column not found in features_df; cannot compute target variable.")
        return None

    # --- FRED DATA INTEGRATION ---
    if use_fred:
        fred_db_path = config.get('fred_db_path', None)
        if not fred_db_path:
            print("❌ FRED database path ('fred_db_path') not found in config.yaml.")
        else:
            print(f"\n🌐 Integrating FRED macro data from {fred_db_path}...")
            import sqlite3
            fred_conn = sqlite3.connect(fred_db_path)
            fred_df = pd.read_sql_query("SELECT * FROM fred_data_wide", fred_conn, parse_dates=['date'])
            fred_conn.close()
            fred_df = fred_df.rename(columns={col: col.lower() for col in fred_df.columns})
            if 'date' not in fred_df.columns:
                print("❌ FRED data must have a 'date' column.")
            else:
                fred_df = fred_df.set_index('date')
                # Left join on date
                features_df = features_df.join(fred_df, how='left')
                print(f"✅ FRED data columns added: {list(fred_df.columns)}")
    
    if audit:
        audit_dir = Path(__file__).parent.parent / 'data' / 'audit'
        audit_dir.mkdir(parents=True, exist_ok=True)
        features_path = audit_dir / f"train_enhanced_model__{target_symbol}_features.csv"
        target_path = audit_dir / f"train_enhanced_model__{target_symbol}_target.csv"
        # Save features (X)
        features_df.drop(columns=['target'], errors='ignore').to_csv(features_path)
        # Save target (Y)
        if 'target' in features_df.columns:
            features_df[['target']].to_csv(target_path)
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
    drop_cols = ['target', 'close', 'open', 'high', 'low', 'volume']
    # Always keep 'returns_1d' for target calculation
    keep_cols = [col for col in features_df.columns if col not in drop_cols or col == 'returns_1d']
    features_for_training = features_df[keep_cols].copy()

    # --- FEATURE SELECTION ---
    if selection:
        print(f"\n🔬 Feature selection: {selection.upper()}")
        X = features_for_training.copy()
        y = features_df['target']
        if selection.upper() == 'STEPWISE':
            selected = stepwise_selection(X, y, direction='stepwise', threshold_in=slentry if slentry is not None else 0.30, threshold_out=slstay if slstay is not None else 0.10, verbose=True)
        elif selection.upper() == 'BACKWARD':
            selected = backward_selection(X, y, threshold_out=slstay if slstay is not None else 0.10, verbose=True)
        elif selection.upper() == 'FORWARD':
            selected = forward_selection(X, y, threshold_in=slentry if slentry is not None else 0.10, verbose=True)
        else:
            print(f"⚠️ Unknown selection method: {selection}. Skipping selection.")
            selected = list(X.columns)
        print(f"\n✅ Selected {len(selected)} features:")
        for feat in selected:
            print(f"  {feat}")
        features_for_training = X[selected]
        # Ensure 'returns_1d' is present for model training, even if not selected as a feature
        if 'returns_1d' in features_df.columns and 'returns_1d' not in features_for_training.columns:
            features_for_training = pd.concat([features_for_training, features_df[['returns_1d']]], axis=1)
        feature_names = selected
    else:
        feature_names = list(features_for_training.columns)

    # --- Prepare X and y for both OLS and sklearn fits ---
    X = features_for_training[feature_names]
    y = features_df['target']

    # Train model
    print(f"\n🏋️ Training enhanced linear regression model...")
    # Always fit intercept by default
    model_result = train_linear_regression_model(X, y, fit_intercept=True)
    
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
    if 'target' in features_df.columns:
        audit_dir_path = None
        if audit:
            audit_dir_path = Path(__file__).parent.parent / 'data' / 'audit'
        print_regression_pvalues(features_df, feature_names, 'target', audit_dir=audit_dir_path, target_symbol=target_symbol)
    else:
        print("⚠️ Could not compute p-values: 'target' column not found in features.")
    
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
            },
            # --- Store modeling data directly in model_package ---
            'features_df': features_df.copy(),
            'target': features_df['target'].copy() if 'target' in features_df.columns else None
        }
        # Add coefficients to model_package for audit and downstream analysis
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            coef_dict = {'Intercept': model.intercept_}
            for name, coef in zip(feature_names, model.coef_):
                coef_dict[name] = coef
            model_package['coefficients'] = coef_dict
        else:
            model_package['coefficients'] = None
        joblib.dump(model_package, model_path)
        print(f"✅ Model saved: {model_path}")
        # --- AUDIT: Dump full model_package if requested ---
        if audit:
            audit_dir = Path(__file__).parent.parent / 'data' / 'audit'
            audit_dir.mkdir(parents=True, exist_ok=True)
            dump_path = audit_dir / f"train_enhanced_model__{target_symbol}_model_package_dump.txt"
            import pprint
            with open(dump_path, 'w', encoding='utf-8') as f:
                for key, value in model_package.items():
                    f.write(f'===== {key} =====\n')
                    if hasattr(value, 'to_string'):
                        f.write(value.to_string())
                        f.write('\n')
                    elif isinstance(value, dict):
                        f.write(pprint.pformat(value, indent=2, width=120))
                        f.write('\n')
                    else:
                        f.write(str(value))
                        f.write('\n')
                    f.write('\n')
            print(f"✅ Audit: Full model_package dump saved to {dump_path}")

        # Quick prediction test
        print(f"\n🔮 Testing predictions...")
        latest_features = features_df.iloc[-1:][feature_names]
        prediction = model.predict(latest_features)[0]
        print(f"Latest prediction for {target_symbol}: {prediction:.4f} ({target_variable}, window={target_window})")
        print(f"Date: {features_df.index[-1]}")
        
        return model_path
    
    return model_result

def print_regression_pvalues(features_df, feature_names, target_col, audit_dir=None, target_symbol=None):
    """Print p-values for each feature using statsmodels OLS. Optionally save summary and coefficients to audit_dir."""
    X = features_df[feature_names]
    y = features_df[target_col]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print("\n📊 Regression Coefficient P-values:")
    for fname, pval in zip(['Intercept'] + feature_names, model.pvalues):
        print(f"  {fname:30s}: p-value = {pval:.4g}")
    print("\nSummary:\n", model.summary())
    # Save summary and coefficients to audit_dir if provided
    if audit_dir is not None and target_symbol is not None:
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        summary_path = audit_dir / f"train_enhanced_model__{target_symbol}_ols_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(str(model.summary()))
        print(f"✅ OLS Regression Results saved to {summary_path}")
        # Save coefficients as CSV
        coef_path = audit_dir / f"train_enhanced_model__{target_symbol}_ols_coefficients.csv"
        import csv
        coefs = model.params
        with open(coef_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['feature', 'coefficient'])
            for fname, coef in coefs.items():
                writer.writerow([fname, coef])
        print(f"✅ OLS coefficients saved to {coef_path}")

def auto_select_peers(target_symbol: str, config: dict, max_peers: int = 8) -> list:
    """Auto-select peers using validated_symbols.yaml from config. Supports both list and dict formats."""
    import yaml
    from pathlib import Path
    symbols_file = Path(config['validated_symbols_path'])
    if not symbols_file.exists():
        print(f"⚠️ Symbols file not found: {symbols_file}")
        return []
    with open(symbols_file, 'r') as f:
        data = yaml.safe_load(f)
    # Support both dict and list formats
    if isinstance(data, dict) and 'symbols' in data:
        symbols = data['symbols']
    elif isinstance(data, dict) and 'valid_symbols' in data:
        # Convert list of dicts to symbol-keyed dict
        symbols = {item['symbol']: item for item in data['valid_symbols']}
    elif isinstance(data, list):
        symbols = {item['symbol']: item for item in data}
    else:
        symbols = data
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

def stepwise_selection(X, y, direction='stepwise', threshold_in=0.30, threshold_out=0.10, verbose=True, max_iter=100):
    """Perform a stepwise feature selection based on p-values from statsmodels OLS. Stops if max_iter is exceeded."""
    import statsmodels.api as sm
    initial_features = list(X.columns)
    included = []
    last_changed = None
    iter_count = 0
    while True:
        changed = False
        iter_count += 1
        # Forward step
        excluded = list(set(initial_features) - set(included))
        new_pvals = pd.Series(index=excluded, dtype=float)
        for new_col in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_col]]))).fit()
            new_pvals[new_col] = model.pvalues[new_col]
        if not new_pvals.empty:
            best_pval = new_pvals.min()
            if best_pval < threshold_in:
                best_feature = new_pvals.idxmin()
                included.append(best_feature)
                last_changed = best_feature
                changed = True
                if verbose:
                    print(f'  Add {best_feature:30} with p-value {best_pval:.6f}')
                # Only one change per iteration
                continue
        # Backward step (only if nothing was added)
        if direction in ['stepwise', 'backward'] and included and not changed:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            pvalues = model.pvalues.iloc[1:]  # exclude intercept
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                last_changed = worst_feature
                changed = True
                if verbose:
                    print(f'  Drop {worst_feature:30} with p-value {worst_pval:.6f}')
        if iter_count > max_iter:
            print(f"⚠️ Stepwise selection exceeded {max_iter} iterations. Dropping last changed variable '{last_changed}' and stopping.")
            if last_changed is not None and last_changed in included:
                included.remove(last_changed)
            break
        if not changed:
            break
    return included

def backward_selection(X, y, threshold_out=0.10, verbose=True):
    """Perform backward elimination based on p-values from statsmodels OLS."""
    import statsmodels.api as sm
    included = list(X.columns)
    while True:
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        pvalues = model.pvalues.iloc[1:]  # exclude intercept
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f'  Drop {worst_feature:30} with p-value {worst_pval:.6f}')
        else:
            break
    return included

def forward_selection(X, y, threshold_in=0.10, verbose=True):
    """Perform forward selection based on p-values from statsmodels OLS."""
    import statsmodels.api as sm
    included = []
    while True:
        excluded = list(set(X.columns) - set(included))
        new_pvals = pd.Series(index=excluded, dtype=float)
        for new_col in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_col]]))).fit()
            new_pvals[new_col] = model.pvalues[new_col]
        if new_pvals.empty:
            break
        best_pval = new_pvals.min()
        if best_pval < threshold_in:
            best_feature = new_pvals.idxmin()
            included.append(best_feature)
            if verbose:
                print(f'  Add {best_feature:30} with p-value {best_pval:.6f}')
        else:
            break
    return included

def main():
    """Main function with command line options"""
    parser = argparse.ArgumentParser(description='Train enhanced model')
    parser.add_argument('--symbol', default='MSFT', help='Target symbol')
    parser.add_argument('--peers', action='store_true', help='Include peer features')
    parser.add_argument('--auto-peers', action='store_true', help='Auto-select peers')
    parser.add_argument('--no-save', action='store_true', help="Don't save the model (testing only)")
    parser.add_argument('--audit', action='store_true', help='Save training X and Y to data/audit')
    parser.add_argument('--selection', type=str, default=None, help='Feature selection method: STEPWISE, BACKWARD, FORWARD, or None')
    parser.add_argument('--slentry', type=float, default=None, help='Probability threshold to enter the model (FORWARD/STEPWISE)')
    parser.add_argument('--slstay', type=float, default=None, help='Probability threshold to stay in the model (BACKWARD/STEPWISE)')
    parser.add_argument('--use-fred', action='store_true', help='Include FRED macroeconomic data (default: False)')
    # Removed --fred-db-path argument
    args = parser.parse_args()
    result = train_enhanced_model(
        target_symbol=args.symbol,  # Pass the symbol directly
        include_peers=args.peers or args.auto_peers,  # Auto-enable peers if auto-selected
        save_model=not args.no_save,
        audit=args.audit,
        selection=args.selection,
        use_fred=args.use_fred,
        slentry=args.slentry,
        slstay=args.slstay,
        auto_peers=args.auto_peers
    )
    if result:
        print(f"\n🎉 SUCCESS! Enhanced model for {args.symbol} ready!")
    else:
        print(f"\n❌ Training failed for {args.symbol}")

if __name__ == "__main__":
    main()