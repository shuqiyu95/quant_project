"""
特征工程和标签生成模块

功能：
1. 从 OHLCV 数据提取基础量价因子
2. 生成标签（未来收益率）
3. 支持多股票批量处理
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import sys
import os

# 添加父目录到路径以导入 factors 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factors.operators import *


class FeatureEngineer:
    """
    特征工程类
    
    提取常用的技术指标作为特征：
    - 价格动量特征
    - 波动率特征
    - 成交量特征
    - 技术指标特征
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Args:
            feature_names: 要使用的特征名称列表，None 表示使用默认特征
        """
        self.feature_names = feature_names or self._get_default_features()
        
    def _get_default_features(self) -> List[str]:
        """获取默认特征列表"""
        return [
            'return_5d',      # 过去5日收益率
            'return_10d',     # 过去10日收益率
            'return_20d',     # 过去20日收益率
            'volatility_5d',  # 5日波动率
            'volatility_20d', # 20日波动率
            'volume_ratio_5d', # 5日成交量比率
            'rsi_14',         # RSI(14)
            'macd',           # MACD
            'ma5_ratio',      # 5日均线比率
            'ma20_ratio',     # 20日均线比率
        ]
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从 OHLCV 数据创建特征
        
        Args:
            df: OHLCV 数据 (必须包含 close, high, low, volume)
            
        Returns:
            features_df: 特征 DataFrame
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # 1. 收益率特征
        features['return_5d'] = Ref(close, 5) / close - 1
        features['return_10d'] = Ref(close, 10) / close - 1
        features['return_20d'] = Ref(close, 20) / close - 1
        
        # 2. 波动率特征
        returns = close / Ref(close, 1) - 1
        features['volatility_5d'] = Std(returns, 5)
        features['volatility_20d'] = Std(returns, 20)
        
        # 3. 成交量特征
        features['volume_ratio_5d'] = volume / MA(volume, 5)
        features['volume_ratio_20d'] = volume / MA(volume, 20)
        
        # 4. RSI 指标
        features['rsi_14'] = RSI(close, 14)
        
        # 5. MACD 指标
        macd_df = MACD(close)
        features['macd'] = macd_df['MACD']
        features['macd_signal'] = macd_df['signal']
        features['macd_hist'] = macd_df['hist']
        
        # 6. 均线比率
        features['ma5_ratio'] = close / MA(close, 5) - 1
        features['ma20_ratio'] = close / MA(close, 20) - 1
        features['ma60_ratio'] = close / MA(close, 60) - 1
        
        # 7. 价格位置（相对于高低点）
        high_20d = Max(high, 20)
        low_20d = Min(low, 20)
        features['price_position'] = (close - low_20d) / (high_20d - low_20d + 1e-10)
        
        # 8. 动量特征
        features['momentum_5d'] = close / Ref(close, 5) - 1
        features['momentum_10d'] = close / Ref(close, 10) - 1
        
        # 9. 成交量价格相关性
        features['volume_price_corr'] = Corr(volume, close, 20)
        
        return features
    
    def create_labels(self, df: pd.DataFrame, forward_days: int = 5) -> pd.Series:
        """
        创建标签：未来N天的累计收益率
        
        Args:
            df: OHLCV 数据
            forward_days: 向前看几天
            
        Returns:
            labels: 未来收益率序列
        """
        close = df['close']
        
        # 计算未来收益率（向前shift）
        future_return = close.shift(-forward_days) / close - 1
        
        return future_return
    
    def prepare_dataset(
        self,
        data_dict: Dict[str, pd.DataFrame],
        forward_days: int = 5,
        min_periods: int = 60
    ) -> tuple:
        """
        准备训练数据集（多股票）
        
        Args:
            data_dict: {symbol: df} 字典
            forward_days: 预测未来几天
            min_periods: 最小有效数据周期
            
        Returns:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            dates: 对应的日期 (n_samples,)
            symbols: 对应的股票代码 (n_samples,)
        """
        X_list = []
        y_list = []
        dates_list = []
        symbols_list = []
        
        for symbol, df in data_dict.items():
            # 创建特征
            features = self.create_features(df)
            
            # 创建标签
            labels = self.create_labels(df, forward_days)
            
            # 合并
            data = pd.concat([features, labels.rename('label')], axis=1)
            
            # 删除缺失值
            data = data.dropna()
            
            # 确保有足够的数据
            if len(data) < min_periods:
                continue
            
            # 添加到列表
            X_list.append(data[features.columns])
            y_list.append(data['label'])
            dates_list.extend(data.index.tolist())
            symbols_list.extend([symbol] * len(data))
        
        # 合并所有数据
        if len(X_list) == 0:
            raise ValueError("没有足够的有效数据")
        
        X = pd.concat(X_list, axis=0)
        y = pd.concat(y_list, axis=0)
        
        return X, y, dates_list, symbols_list
    
    def create_cross_sectional_dataset(
        self,
        data_dict: Dict[str, pd.DataFrame],
        forward_days: int = 5,
        align_dates: bool = True
    ) -> tuple:
        """
        创建截面数据集（每个时间点，所有股票的特征和标签）
        
        适用于排序模型：每个时间点对所有股票进行排序
        
        Args:
            data_dict: {symbol: df} 字典
            forward_days: 预测未来几天
            align_dates: 是否对齐日期（只保留所有股票都有数据的日期）
            
        Returns:
            dataset: List of (date, X, y, symbols)
                date: 日期
                X: 该日期所有股票的特征 (n_stocks, n_features)
                y: 该日期所有股票的标签 (n_stocks,)
                symbols: 股票代码列表
        """
        # 1. 为每个股票创建特征和标签
        features_dict = {}
        labels_dict = {}
        
        for symbol, df in data_dict.items():
            features = self.create_features(df)
            labels = self.create_labels(df, forward_days)
            
            # 删除缺失值
            valid_mask = ~(features.isna().any(axis=1) | labels.isna())
            features_dict[symbol] = features[valid_mask]
            labels_dict[symbol] = labels[valid_mask]
        
        # 2. 找到共同日期
        if align_dates:
            # 所有股票都有数据的日期
            common_dates = set(features_dict[list(data_dict.keys())[0]].index)
            for symbol in data_dict.keys():
                common_dates &= set(features_dict[symbol].index)
            common_dates = sorted(common_dates)
        else:
            # 任意股票有数据的日期
            all_dates = set()
            for symbol in data_dict.keys():
                all_dates |= set(features_dict[symbol].index)
            common_dates = sorted(all_dates)
        
        # 3. 构建截面数据集
        dataset = []
        
        for date in common_dates:
            X_list = []
            y_list = []
            symbols_list = []
            
            for symbol in data_dict.keys():
                if date in features_dict[symbol].index:
                    X_list.append(features_dict[symbol].loc[date].values)
                    y_list.append(labels_dict[symbol].loc[date])
                    symbols_list.append(symbol)
            
            # 至少需要2个股票才能进行排序
            if len(symbols_list) >= 2:
                X = np.array(X_list)
                y = np.array(y_list)
                dataset.append((date, X, y, symbols_list))
        
        return dataset


def create_weekly_trading_dates(dates: pd.DatetimeIndex, weekday: int = 0) -> List[pd.Timestamp]:
    """
    创建每周交易日列表
    
    Args:
        dates: 所有交易日
        weekday: 周几 (0=Monday, 1=Tuesday, ..., 4=Friday)
        
    Returns:
        weekly_dates: 每周对应weekday的交易日列表
    """
    df = pd.DataFrame(index=dates)
    df['weekday'] = df.index.weekday
    
    # 按周分组
    df['week'] = df.index.to_period('W')
    
    weekly_dates = []
    for week, group in df.groupby('week'):
        # 找到该周的目标weekday
        target_days = group[group['weekday'] == weekday]
        if len(target_days) > 0:
            weekly_dates.append(target_days.index[0])
        else:
            # 如果该周没有目标weekday，使用该周第一个交易日
            weekly_dates.append(group.index[0])
    
    return weekly_dates


if __name__ == "__main__":
    # 测试示例
    print("=== Testing Feature Engineering ===\n")
    
    # 创建模拟数据
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    n = len(dates)
    
    df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(n) * 2),
        'high': 100 + np.cumsum(np.random.randn(n) * 2) + 2,
        'low': 100 + np.cumsum(np.random.randn(n) * 2) - 2,
        'volume': 1000000 + np.random.randn(n) * 100000
    }, index=dates)
    
    # 创建特征工程器
    fe = FeatureEngineer()
    
    # 生成特征
    features = fe.create_features(df)
    print("Features shape:", features.shape)
    print("\nFeature columns:")
    print(features.columns.tolist())
    
    # 生成标签
    labels = fe.create_labels(df, forward_days=5)
    print("\nLabels shape:", labels.shape)
    print("Labels sample:")
    print(labels.head())
    
    # 测试多股票数据集
    print("\n=== Testing Multi-Stock Dataset ===")
    data_dict = {
        'AAPL': df.copy(),
        'MSFT': df.copy() * 1.1,
        'GOOGL': df.copy() * 0.9
    }
    
    X, y, dates_list, symbols_list = fe.prepare_dataset(data_dict, forward_days=5)
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Number of samples: {len(dates_list)}")
    print(f"Unique symbols: {set(symbols_list)}")
    
    # 测试截面数据集
    print("\n=== Testing Cross-Sectional Dataset ===")
    dataset = fe.create_cross_sectional_dataset(data_dict, forward_days=5)
    print(f"Number of time points: {len(dataset)}")
    if len(dataset) > 0:
        date, X_cs, y_cs, symbols_cs = dataset[0]
        print(f"First date: {date}")
        print(f"Shape at first date: X={X_cs.shape}, y={y_cs.shape}")
        print(f"Symbols: {symbols_cs}")

