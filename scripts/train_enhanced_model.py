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
    
    # initialize peer_symbols
    peer_symbols = []

    # --- FIX: Use auto-selected peers if auto_peers is True ---
    if auto_peers:
        print(f"🔎 Auto-selecting peers for {target_symbol}...")
        peer_symbols = auto_select_peers(target_symbol, config)
        config['peer_symbols'] = peer_symbols
        print(f"Auto-selected {len(peer_symbols)} peers: {peer_symbols}\n")
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
        print(f"Peer symbols: {len(peer_symbols)} symbols\n")
    
    print(f"[---DEBUG---]\npeer_symbols : {peer_symbols}\n")

    # --- Generate enhanced features from DB ---
    print(f"\n📊 Loading modeling data from database...")
    # Prepare arguments for loader
    periods = config['periods']
    feature_flags = config.get('features', {}).get('technical', {}).get('features', {})
    signal_flags = config.get('features', {}).get('signals', {}).get('features', {})
    fred_flags = config.get('features', {}).get('fred', {}).get('features', {})
    # peer_symbols = config.get('peer_symbols', []) if include_peers or auto_peers else []
    use_technical = True
    use_signals = True
    features_df = load_modeling_data_from_db(
        config,
        target_symbol,
        peer_symbols,
        periods,
        use_technical,
        use_signals,
        use_fred,
        feature_flags,
        fred_flags,
        signal_flags,
        config.get('target_variable', 'returns')
    )
    if features_df.empty:
        print("❌ No features loaded from database")
        return None
    
    # --- No legacy target construction: expect target column from DB ---
    # Robustly find the wide-format target column (e.g., msft__target)
    target_col_name = f"{target_symbol.lower()}__target"
    # Case/whitespace-insensitive match
    target_col_actual = next((col for col in features_df.columns if col.strip().lower() == target_col_name), None)
    if not target_col_actual:
        # Try to find any column ending with '__target'
        target_cols = [col for col in features_df.columns if col.strip().lower().endswith('__target')]
        if not target_cols:
            print(f"❌ Target column '{target_col_name}' not found in features_df; cannot train model.")
            return None
        else:
            print(f"⚠️ Target variable for {target_symbol} not found, but found: {target_cols}. Using first available target column.")
            target_col_actual = target_cols[0]
    # Use the robustly found target column
    y = features_df[target_col_actual]
    X = features_df.drop(columns=[target_col_actual, 'date'], errors='ignore')

    # --- FEATURE SELECTION ---
    if selection:
        print(f"\n🔬 Feature selection: {selection.upper()}\n")
        X_sel = X.copy()
        y_sel = y.copy()
        X_sel = drop_correlated_features(X_sel, threshold=0.9)
        if selection.upper() == 'STEPWISE':
            selected = stepwise_selection(X_sel, y_sel, direction='stepwise', threshold_in=slentry if slentry is not None else 0.30, threshold_out=slstay if slstay is not None else 0.10, verbose=True)
        elif selection.upper() == 'BACKWARD':
            selected = backward_selection(X_sel, y_sel, threshold_out=slstay if slstay is not None else 0.10, verbose=True)
        elif selection.upper() == 'FORWARD':
            selected = forward_selection(X_sel, y_sel, threshold_in=slentry if slentry is not None else 0.10, verbose=True)
        else:
            print(f"⚠️ Unknown selection method: {selection}. Skipping selection.")
            selected = list(X_sel.columns)
        print(f"\n✅ Selected {len(selected)} features\n")
        for feat in selected:
            print(f"  {feat}")
        X = X_sel[selected]
        feature_names = selected
    else:
        feature_names = list(X.columns)

    # --- Prepare X and y for both OLS and sklearn fits ---
    X = X[feature_names]
    y = y

   # --- Diagnostic: Save X and y to CSV in scripts/temporary_scripts right before model fit ---
    diag_dir = Path(__file__).parent / 'temporary_scripts'
    diag_dir.mkdir(parents=True, exist_ok=True)
    y_final_path = diag_dir / f"y_final_after_{target_symbol}.csv"
    y.to_frame().to_csv(y_final_path, index=False)
    print(f"\n[DIAG] Saved final y after to {y_final_path}\n")

    # --- Fix: Only select features whose feature part matches config.yaml ---
    # X columns are like symbol__feature, config features are just feature
    config_feature_set = set([k for k, v in feature_flags.items() if v is True and isinstance(v, bool)])
    def feature_part(col):
        if '__' in col:
            return col.split('__', 1)[1]
        return col
    selected_feature_cols = [col for col in X.columns if feature_part(col) in config_feature_set]
    X = X[selected_feature_cols]
    # print(f"[DIAG] Selected {len(selected_feature_cols)} features after matching config: {selected_feature_cols}")

    # --- Extra Diagnostic: Print all X columns and config features for debugging ---
    # print(f"[DIAG] All X columns: {list(X.columns)}")
    # print(f"[DIAG] Enabled config features: {list(config_feature_set)}")

    # --- Diagnostic: Save X and y to CSV in scripts/temporary_scripts right before model fit ---
    diag_dir = Path(__file__).parent / 'temporary_scripts'
    diag_dir.mkdir(parents=True, exist_ok=True)
    X_final_path = diag_dir / f"X_final_{target_symbol}.csv"
    X.to_csv(X_final_path, index=False)
    print(f"[DIAG] Saved final X to {X_final_path}")

    # --- Final diagnostic: Check for NaNs in X and y right before model fit ---
    nans_in_X = X.isnull().sum().sum()
    nans_in_y = y.isnull().sum()
    print(f"[DIAG] NaNs in X: {nans_in_X}")
    print(f"[DIAG] NaNs in y: {nans_in_y}")
    if nans_in_X > 0 or nans_in_y > 0:
        print("[ERROR] There are still missing values in X or y right before model fit! Aborting model training.")
        # Save problematic X and y for inspection
        diag_dir = Path(__file__).parent / 'temporary_scripts'
        diag_dir.mkdir(parents=True, exist_ok=True)
        X.to_csv(diag_dir / f"X_with_nans_{target_symbol}.csv", index=False)
        y.to_frame().to_csv(diag_dir / f"y_with_nans_{target_symbol}.csv", index=False)
        return None

    # --- Check for date alignment between X and y BEFORE dropping date column ---
    # This must be done immediately after X and y are created from features_df
    if 'date' in features_df.columns:
        x_dates = set(features_df['date'])
    else:
        x_dates = set()
    # y is a Series from features_df[target_col_actual], so get its index or use features_df['date']
    y_dates = set(features_df['date']) if 'date' in features_df.columns else set()
    if x_dates and y_dates:
        if x_dates != y_dates:
            print("[WARNING] Date mismatch between X and y before model fit!")
            only_in_x = sorted(list(x_dates - y_dates))
            only_in_y = sorted(list(y_dates - x_dates))
            print(f"[DIAG] Dates only in X: {only_in_x}")
            print(f"[DIAG] Dates only in y: {only_in_y}")
            # Keep only records with matching dates
            matching_dates = x_dates & y_dates
            features_df = features_df[features_df['date'].isin(matching_dates)]
            print(f"[WARNING] Filtered to {len(matching_dates)} matching dates for X and y.")
    # Now proceed to drop date from X and y as before

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
    if audit:
        audit_dir_path = Path(__file__).parent.parent / 'data' / 'audit'
        print_regression_pvalues(features_df, feature_names, target_col_actual, audit_dir=audit_dir_path, target_symbol=target_symbol)
    
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
            'target': features_df[target_col_actual].copy() if target_col_actual in features_df.columns else None
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
        print(f"Latest prediction for {target_symbol}: {prediction:.4f} (target column: {target_col_actual})")
        print(f"Date: {features_df.index[-1]}")
        
        return model_path
    
    return model_result

    # --- Feature breakdown (moved up for use in model_package) ---
    feature_types = {
        'Technical': len([c for c in features_df.columns if any(x in c for x in ['rsi', 'macd', 'bb_', 'cci', 'stoch', 'atr', 'obv', 'ichimoku', 'sma'])]),
        'Returns/Vol': len([c for c in features_df.columns if any(x in c for x in ['returns', 'volatility', 'volume'])]),
        'Lagged': len([c for c in features_df.columns if 'lag_' in c or 'roll_' in c]),
        'Peer': len([c for c in features_df.columns if any(x in c for x in ['rel_', 'avg_rel'])])
    }

def drop_correlated_features(features_df, threshold=0.9):
    """Drop features that are highly correlated with each other based on a correlation threshold."""
    import numpy as np
    corr_matrix = features_df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find features with correlation above the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Dropping {len(to_drop)} correlated features: {to_drop}")
    return features_df.drop(columns=to_drop, errors='ignore')


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

def load_modeling_data_from_db(config, target_symbol, peer_symbols, periods, use_technical, use_signals, use_fred, feature_flags, fred_flags, signal_flags, target_variable):
    """
    Load modeling data (features + target) from the relational database, using only the features and target specified in the config/CLI.
    Joins technical_indicators, technical_trade_signals, outcomes, fred_data_wide, and symbols tables as needed.
    Returns a DataFrame indexed by date (and symbol if needed).
    """
    import sqlite3
    import pandas as pd
    from functools import reduce

    # --- DB paths ---
    stock_db_path = config['stock_db_path']
    fred_db_path = config['fred_db_path']

    # --- Connect to stock DB ---
    stock_conn = sqlite3.connect(stock_db_path)
    # --- Get symbol_id for target and peers ---
    symbol_query = f"SELECT symbol, symbol_id FROM symbols"
    symbol_df = pd.read_sql(symbol_query, stock_conn)
    symbol_map = dict(zip(symbol_df['symbol'], symbol_df['symbol_id']))
    all_symbols = [target_symbol] + (peer_symbols if peer_symbols else [])
    symbol_ids = [symbol_map[s] for s in all_symbols if s in symbol_map]
    if not symbol_ids:
        raise ValueError(f"No valid symbol_ids found for {all_symbols}")

    # --- Periods ---
    strategy_start = min(periods['training']['start'], periods['test']['start'], periods['strategy']['start'])
    strategy_end = max(periods['training']['end'], periods['test']['end'], periods['strategy']['end'])

    # --- Feature selection ---
    # Only include feature names (keys) where value is True
    technical_cols = [k for k, v in feature_flags.items() if v is True and isinstance(v, bool)]
    signal_cols = [k for k, v in signal_flags.items() if v is True and isinstance(v, bool)]
    fred_cols = [k for k, v in fred_flags.items() if v is True and isinstance(v, bool)]
    # Remove section master flags if present
    technical_cols = [c for c in technical_cols if c != 'technical']
    signal_cols = [c for c in signal_cols if c != 'signals']
    fred_cols = [c for c in fred_cols if c != 'fred']

    # --- Query technical indicators ---
    print(f"\n[DEBUG] Requesting technical indicator columns: {technical_cols}\n")
    tech_query = f"""
        SELECT date, symbol_id, {', '.join(technical_cols)}
        FROM technical_indicators
        WHERE symbol_id IN ({', '.join(['?']*len(symbol_ids))})
          AND date BETWEEN ? AND ?
    """
    print(f"\n[DEBUG] technical_indicators SQL: {tech_query}\n")
    tech_df = pd.read_sql(tech_query, stock_conn, params=symbol_ids + [strategy_start, strategy_end]) if technical_cols else None
    if tech_df is not None:
        print(f"\n[DEBUG] technical_indicators result shape: {tech_df.shape}")
        print(f"[DEBUG] technical_indicators columns: {list(tech_df.columns)}\n")
        if tech_df.empty:
            print("[WARNING] technical_indicators query returned no rows.")
    # --- Query trade signals ---
    signal_query = f"""
        SELECT date, symbol_id, {', '.join(signal_cols)}
        FROM technical_trade_signals
        WHERE symbol_id IN ({', '.join(['?']*len(symbol_ids))})
          AND date BETWEEN ? AND ?
    """
    signal_df = pd.read_sql(signal_query, stock_conn, params=symbol_ids + [strategy_start, strategy_end]) if signal_cols else None

    # --- Extract target variable as string from string, dict, or list of dicts ---
    target_col = None
    if isinstance(target_variable, str):
        target_col = target_variable
    elif isinstance(target_variable, dict):
        target_col = next((k for k, v in target_variable.items() if v), None)
    elif isinstance(target_variable, list):
        # Support list of dicts
        for item in target_variable:
            if isinstance(item, dict):
                for k, v in item.items():
                    if v:
                        target_col = k
                        break
            if target_col:
                break
    if not target_col:
        raise ValueError("No enabled target variable found in target_variable config.")

    # --- Query outcomes (target) ---
    outcome_query = f"""
        SELECT date, symbol_id, {target_col} as target
        FROM outcomes
        WHERE symbol_id IN ({', '.join(['?']*len(symbol_ids))})
          AND date BETWEEN ? AND ?
    """
    outcome_df = pd.read_sql(outcome_query, stock_conn, params=symbol_ids + [strategy_start, strategy_end])

    # --- Merge technical, signals, and outcomes ---
    dfs = [df for df in [tech_df, signal_df, outcome_df] if df is not None]
    if not dfs:
        raise ValueError("No features selected or available.")
    # Merge on date+symbol_id
    data = reduce(lambda left, right: pd.merge(left, right, on=['date', 'symbol_id'], how='outer'), dfs)
    print(f"\n[DEBUG] After merge, data columns: {list(data.columns)}\n")

    # --- Add symbol column ---
    inv_symbol_map = {v: k for k, v in symbol_map.items()}
    data['symbol'] = data['symbol_id'].map(inv_symbol_map)

    # --- Add FRED data if needed ---
    if use_fred and fred_cols:
        fred_conn = sqlite3.connect(fred_db_path)
        fred_query = f"SELECT date, {', '.join(fred_cols)} FROM fred_data_wide WHERE date BETWEEN ? AND ?"
        fred_df = pd.read_sql(fred_query, fred_conn, params=[strategy_start, strategy_end])
        fred_conn.close()
        fred_df['date'] = pd.to_datetime(fred_df['date'])
        data_reset = data.reset_index()
        data_reset['date'] = pd.to_datetime(data_reset['date'], errors='coerce')
        # Merge on 'date' only, so FRED data is applied to all symbols for a given date
        data = pd.merge(data_reset, fred_df, on='date', how='left')
        data = data.set_index(['date', 'symbol'])

    # --- Filter to only enabled columns + target + symbol/date ---
    keep_cols = ['date', 'symbol', 'symbol_id'] + technical_cols + signal_cols + fred_cols + ['target']
    data = data.reset_index() if 'date' in data.index.names or 'symbol' in data.index.names else data
    # Only keep columns that exist in data
    missing_cols = [col for col in keep_cols if col not in data.columns]
    if missing_cols:
        print(f"\n⚠️ The following columns are missing from the data and will be skipped: {missing_cols}\n")
    keep_cols_present = [col for col in keep_cols if col in data.columns]
    data = data[keep_cols_present]

    # --- Deduplicate: ensure one row per (date, symbol) ---
    data = data.groupby(['date', 'symbol'], as_index=False).first()

    # --- Reshape to one observation per date (wide format for symbol-specific columns) ---
    # Identify symbol-specific columns (all except date, symbol, symbol_id, and fred_cols)
    symbol_specific_cols = [col for col in data.columns if col not in ['date', 'symbol', 'symbol_id'] + fred_cols]
    print(f"\n[DEBUG] Columns to pivot (symbol_specific_cols): {symbol_specific_cols}\n")
    # Pivot to wide format: index=date, columns=symbol, values=symbol_specific_cols
    wide = data.pivot(index='date', columns='symbol', values=symbol_specific_cols)
    # Flatten MultiIndex columns: (feature, symbol) -> symbol__feature
    wide.columns = [f"{str(symbol).lower()}__{feature}" for feature, symbol in wide.columns]
    print(f"\n[DEBUG] First 10 wide DataFrame columns after flattening: {list(wide.columns)[:10]}")
    wide = wide.reset_index()
    print("\n[DEBUG] First 5 rows of wide DataFrame after flattening:")
    print(wide.head())
    # Merge FRED columns back in (they are already at date level)
    fred_only = None
    if fred_cols:
        fred_cols_present = [col for col in fred_cols if col in data.columns]
        missing_fred_cols = [col for col in fred_cols if col not in data.columns]
        if missing_fred_cols:
            print(f"\n[WARNING] The following FRED columns are missing from data and will be skipped: {missing_fred_cols}")
        if fred_cols_present:
            fred_only = data[['date'] + fred_cols_present].drop_duplicates('date').set_index('date')
            wide = pd.merge(wide, fred_only, left_on='date', right_index=True, how='left')
    # Remove the first 200 records (priming period)
    wide = wide.iloc[200:].reset_index(drop=True)

    # --- Extract lookahead period from target variable name (e.g., price_d10 -> 10) ---
    import re
    lookahead_period = 0
    target_var = str(target_variable)
    match = re.search(r'_d(\d+)$', target_var)
    if match:
        lookahead_period = int(match.group(1))
        print(f"[INFO] Detected lookahead period: {lookahead_period} days from target '{target_var}'")
    else:
        print(f"[INFO] No lookahead period detected in target '{target_var}'")

    # --- Remove the last N rows for lookahead period, after priming period trim ---
    if lookahead_period > 0:
        wide = wide.iloc[:-lookahead_period].reset_index(drop=True)
        print(f"\n[INFO] Removed last {lookahead_period} rows from wide DataFrame for lookahead period.\n")

    # Ensure target variable is included in wide output for the target symbol
    # Find the wide-format column name for the target variable
    target_col_name = f"{str(target_symbol).lower()}__target"
    if target_col_name not in wide.columns:
        # Try to find any column ending with '__target' and print a warning if not found
        target_cols = [col for col in wide.columns if col.endswith('__target')]
        if not target_cols:
            print(f"⚠️ No target variable columns found in wide-format output. Check that outcomes table and target_variable are correct.")
        else:
            print(f"⚠️ Target variable for {target_symbol} not found, but found: {target_cols}")

    # Print min/max dates in technical_indicators and outcomes
    if tech_df is not None:
        print(f"\n[DEBUG] technical_indicators date range: {tech_df['date'].min()} to {tech_df['date'].max()}")
    if outcome_df is not None:
        print(f"[DEBUG] outcomes date range: {outcome_df['date'].min()} to {outcome_df['date'].max()}\n")

    # --- Diagnostic: Save wide DataFrame to CSV in scripts/temporary_scripts ---
    diag_dir = Path(__file__).parent / 'temporary_scripts'
    diag_dir.mkdir(parents=True, exist_ok=True)
    wide_diag_path = diag_dir / f"diagnostic_wide_{target_symbol}.csv"
    wide.to_csv(wide_diag_path, index=False)
    print(f"[DIAG] Saved wide DataFrame to {wide_diag_path}")
    print(f"[DIAG] wide DataFrame info:")
    print(wide.info())
    print(f"[DIAG] wide DataFrame head:")
    print(wide.head())
    print(f"[DIAG] wide DataFrame missing values per column:")
    print(wide.isnull().sum())

    return wide

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