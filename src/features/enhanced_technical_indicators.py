#!/usr/bin/env python3
"""
Enhanced technical indicators with peer symbol support
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional

# Add database path
sys.path.append(str(Path(__file__).parent.parent.parent / 'database'))
from database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class EnhancedTechnicalIndicators:
    """Enhanced technical indicators calculator with peer support"""
    
    def __init__(self, config: dict):
        self.config = config
        self.db_manager = DatabaseManager()
        
    def calculate_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators based on config"""
        
        features = pd.DataFrame(index=data.index)
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        technical_config = self.config.get('features', {}).get('technical', {})
        
        # RSI
        if technical_config.get('rsi', {}).get('enabled', False):
            period = technical_config['rsi'].get('period', 21)
            features[f'rsi_{period}'] = self.calculate_rsi(close, period)
        
        # MACD
        if technical_config.get('macd', {}).get('enabled', False):
            fast = technical_config['macd'].get('fast_period', 12)
            slow = technical_config['macd'].get('slow_period', 26)
            signal = technical_config['macd'].get('signal_period', 9)
            
            macd, macd_signal = self.calculate_macd(close, fast, slow, signal)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd - macd_signal
        
        # Bollinger Bands
        if technical_config.get('bollinger', {}).get('enabled', False):
            window = technical_config['bollinger'].get('window', 15)
            std_dev = technical_config['bollinger'].get('std', 1.5)
            
            bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands(close, window, std_dev)
            features['bb_upper'] = bb_upper
            features['bb_lower'] = bb_lower
            features['bb_middle'] = bb_middle
            features['bb_width'] = bb_upper - bb_lower
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # CCI (Commodity Channel Index)
        if technical_config.get('cci', {}).get('enabled', False):
            period = technical_config['cci'].get('period', 20)
            features[f'cci_{period}'] = self.calculate_cci(high, low, close, period)
        
        # Stochastic
        if technical_config.get('stochastic', {}).get('enabled', False):
            k_period = technical_config['stochastic'].get('k_period', 14)
            d_period = technical_config['stochastic'].get('d_period', 3)
            
            stoch_k, stoch_d = self.calculate_stochastic(high, low, close, k_period, d_period)
            features[f'stoch_k_{k_period}'] = stoch_k
            features[f'stoch_d_{d_period}'] = stoch_d
        
        # ATR (Average True Range)
        if technical_config.get('atr', {}).get('enabled', False):
            period = technical_config['atr'].get('period', 14)
            features[f'atr_{period}'] = self.calculate_atr(high, low, close, period)
        
        # OBV (On Balance Volume)
        if technical_config.get('obv', {}).get('enabled', False):
            window = technical_config['obv'].get('window', 20)
            obv = self.calculate_obv(close, volume)
            features['obv'] = obv
            features[f'obv_sma_{window}'] = obv.rolling(window=window).mean()
        
        # Ichimoku
        if technical_config.get('ichimoku', {}).get('enabled', False):
            conv_period = technical_config['ichimoku'].get('conversion_period', 9)
            base_period = technical_config['ichimoku'].get('base_period', 26)
            
            tenkan, kijun, senkou_a, senkou_b = self.calculate_ichimoku(high, low, close, conv_period, base_period)
            features[f'ichimoku_tenkan_{conv_period}'] = tenkan
            features[f'ichimoku_kijun_{base_period}'] = kijun
            features['ichimoku_senkou_a'] = senkou_a
            features['ichimoku_senkou_b'] = senkou_b
        
        # SMA
        if technical_config.get('sma', {}).get('enabled', False):
            periods = technical_config['sma'].get('periods', [5, 10, 20, 50])
            for period in periods:
                features[f'sma_{period}'] = close.rolling(window=period).mean()
        
        # Add basic features (always included)
        features['returns_1d'] = close.pct_change(1)
        features['returns_5d'] = close.pct_change(5)
        features['returns_20d'] = close.pct_change(20)
        features['volatility_20d'] = features['returns_1d'].rolling(window=20).std()
        
        # Volume features
        features['volume_sma_20'] = volume.rolling(window=20).mean()
        features['volume_ratio'] = volume / features['volume_sma_20']
        
        return features
    
    def calculate_peer_features(self, target_symbol: str, peer_symbols: List[str]) -> pd.DataFrame:
        """Calculate relative performance features vs peer symbols"""
        
        logger.info(f"Calculating peer features for {target_symbol} vs {len(peer_symbols)} peers")
        
        # Get target data
        with self.db_manager:
            target_data = self.db_manager.get_stock_prices(target_symbol)
            
            if target_data.empty:
                logger.warning(f"No data for target symbol {target_symbol}")
                return pd.DataFrame()
            
            # Get peer data
            peer_data = {}
            for peer in peer_symbols:
                if peer != target_symbol:  # Skip self
                    peer_prices = self.db_manager.get_stock_prices(peer)
                    if not peer_prices.empty:
                        peer_data[peer] = peer_prices['close']
        
        if not peer_data:
            logger.warning("No peer data available")
            return pd.DataFrame()
        
        # Create peer features DataFrame
        peer_features = pd.DataFrame(index=target_data.index)
        target_close = target_data['close']
        
        # Calculate relative performance metrics
        for peer_symbol, peer_close in peer_data.items():
            # Align data
            aligned_data = pd.concat([target_close, peer_close], axis=1, keys=['target', 'peer']).dropna()
            
            if len(aligned_data) < 50:  # Need minimum data
                continue
                
            # Relative price performance
            relative_perf = aligned_data['target'] / aligned_data['peer']
            peer_features[f'rel_price_{peer_symbol}'] = relative_perf.reindex(target_data.index)
            
            # Relative returns
            target_ret = aligned_data['target'].pct_change()
            peer_ret = aligned_data['peer'].pct_change()
            rel_returns = target_ret - peer_ret
            peer_features[f'rel_return_{peer_symbol}'] = rel_returns.reindex(target_data.index)
            
            # Rolling relative strength
            rel_strength_20d = (target_ret.rolling(20).mean() - peer_ret.rolling(20).mean())
            peer_features[f'rel_strength_20d_{peer_symbol}'] = rel_strength_20d.reindex(target_data.index)
        
        # Calculate aggregate peer metrics
        if len(peer_data) >= 3:  # Need multiple peers for aggregate
            
            # Average relative performance vs all peers
            rel_price_cols = [col for col in peer_features.columns if col.startswith('rel_price_')]
            if rel_price_cols:
                peer_features['avg_rel_price'] = peer_features[rel_price_cols].mean(axis=1)
                peer_features['rel_price_rank'] = peer_features[rel_price_cols].rank(axis=1, pct=True).mean(axis=1)
            
            # Average relative returns
            rel_return_cols = [col for col in peer_features.columns if col.startswith('rel_return_')]
            if rel_return_cols:
                peer_features['avg_rel_return'] = peer_features[rel_return_cols].mean(axis=1)
                peer_features['rel_return_volatility'] = peer_features[rel_return_cols].std(axis=1)
            
            # Relative strength vs peers
            rel_strength_cols = [col for col in peer_features.columns if col.startswith('rel_strength_')]
            if rel_strength_cols:
                peer_features['avg_rel_strength'] = peer_features[rel_strength_cols].mean(axis=1)
        
        logger.info(f"Generated {len(peer_features.columns)} peer features")
        return peer_features
    
    def calculate_lagged_features(self, target_symbol: str, peer_symbols: List[str] = None) -> pd.DataFrame:
        """Calculate lagged features based on config"""
        
        lags_config = self.config.get('features', {}).get('lags', {})
        
        if not lags_config.get('enabled', False):
            return pd.DataFrame()
        
        logger.info("Calculating lagged features...")
        
        with self.db_manager:
            # Get target data
            target_data = self.db_manager.get_stock_prices(target_symbol)
            
            if target_data.empty:
                return pd.DataFrame()
            
            lagged_features = pd.DataFrame(index=target_data.index)
            
            # Target symbol lagged features
            if lags_config.get('include_target', True):
                price_columns = lags_config.get('price_columns', ['close'])
                lag_periods = lags_config.get('lag_periods', [1, 2, 3, 5])
                
                # Map config column names to actual columns
                column_mapping = {
                    'Close': 'close', 'High': 'high', 'Low': 'low', 
                    'Open': 'open', 'Volume': 'volume'
                }
                
                for col_name in price_columns:
                    actual_col = column_mapping.get(col_name, col_name.lower())
                    if actual_col in target_data.columns:
                        for lag in lag_periods:
                            lagged_features[f'{actual_col}_lag_{lag}'] = target_data[actual_col].shift(lag)
            
            # Peer symbols lagged features (if enabled and peers provided)
            if lags_config.get('include_peers', False) and peer_symbols:
                for peer in peer_symbols[:5]:  # Limit to top 5 peers to avoid too many features
                    if peer != target_symbol:
                        peer_data = self.db_manager.get_stock_prices(peer)
                        if not peer_data.empty:
                            # Only use close price for peers to limit feature count
                            for lag in [1, 5]:  # Limited lags for peers
                                aligned_peer = peer_data['close'].reindex(target_data.index)
                                lagged_features[f'{peer}_close_lag_{lag}'] = aligned_peer.shift(lag)
            
            # Rolling window features
            rolling_config = lags_config.get('rolling_features', {})
            if rolling_config.get('enabled', False):
                windows = rolling_config.get('windows', [5, 10, 20])
                functions = rolling_config.get('functions', ['mean', 'std'])
                
                for window in windows:
                    for func in functions:
                        if func == 'mean':
                            lagged_features[f'close_roll_{window}_mean'] = target_data['close'].rolling(window).mean()
                        elif func == 'std':
                            lagged_features[f'close_roll_{window}_std'] = target_data['close'].rolling(window).std()
                        elif func == 'min':
                            lagged_features[f'close_roll_{window}_min'] = target_data['close'].rolling(window).min()
                        elif func == 'max':
                            lagged_features[f'close_roll_{window}_max'] = target_data['close'].rolling(window).max()
        
        logger.info(f"Generated {len(lagged_features.columns)} lagged features")
        return lagged_features
    
    # Technical indicator calculation methods
    def calculate_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20, num_std: float = 2):
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower, sma
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        return (tp - sma_tp) / (0.015 * mad)
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_ichimoku(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                          conv_period: int = 9, base_period: int = 26):
        # Tenkan (conversion line)
        tenkan = (high.rolling(conv_period).max() + low.rolling(conv_period).min()) / 2
        
        # Kijun (base line)
        kijun = (high.rolling(base_period).max() + low.rolling(base_period).min()) / 2
        
        # Senkou A (leading span A)
        senkou_a = ((tenkan + kijun) / 2).shift(base_period)
        
        # Senkou B (leading span B)
        senkou_b = ((high.rolling(base_period * 2).max() + low.rolling(base_period * 2).min()) / 2).shift(base_period)
        
        return tenkan, kijun, senkou_a, senkou_b

def calculate_enhanced_features(symbol: str, config: dict, include_peers: bool = False) -> pd.DataFrame:
    """Main function to calculate enhanced features with peer option"""
    
    logger.info(f"Calculating enhanced features for {symbol} (include_peers={include_peers})")
    
    calculator = EnhancedTechnicalIndicators(config)
    
    # Get basic price data
    with calculator.db_manager:
        price_data = calculator.db_manager.get_stock_prices(symbol)
        
        if price_data.empty:
            logger.warning(f"No price data for {symbol}")
            return pd.DataFrame()
    
    # Calculate technical indicators
    technical_features = calculator.calculate_advanced_indicators(price_data)
    
    # Calculate lagged features
    peer_symbols = config.get('peer_symbols', []) if include_peers else None
    lagged_features = calculator.calculate_lagged_features(symbol, peer_symbols)
    
    # Combine all features
    all_features = technical_features
    
    if not lagged_features.empty:
        all_features = all_features.join(lagged_features, how='inner')
    
    # Calculate peer features if requested
    if include_peers and peer_symbols:
        peer_features = calculator.calculate_peer_features(symbol, peer_symbols)
        if not peer_features.empty:
            all_features = all_features.join(peer_features, how='inner')
    
    # Remove rows with NaN values
    all_features = all_features.dropna()
    
    # --- Add raw price columns to the returned DataFrame ---
    # Only add columns that exist in price_data
    price_cols = ['close', 'open', 'high', 'low', 'volume']
    for col in price_cols:
        if col in price_data.columns:
            all_features[col] = price_data[col].reindex(all_features.index)
    
    logger.info(f"Generated {len(all_features)} feature records with {len(all_features.columns)} features")
    logger.info(f"Feature categories: Technical={len(technical_features.columns)}, "
               f"Lagged={len(lagged_features.columns) if not lagged_features.empty else 0}, "
               f"Peer={len(peer_features.columns) if include_peers and 'peer_features' in locals() and not peer_features.empty else 0}")
    
    return all_features

if __name__ == "__main__":
    # Test the enhanced features
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    symbol = config['target_symbol']
    
    # Test target-only features
    print("Testing target-only features...")
    features_target_only = calculate_enhanced_features(symbol, config, include_peers=False)
    print(f"Target-only: {features_target_only.shape}")
    
    # Test with peer features
    print("\nTesting with peer features...")
    features_with_peers = calculate_enhanced_features(symbol, config, include_peers=True)
    print(f"With peers: {features_with_peers.shape}")