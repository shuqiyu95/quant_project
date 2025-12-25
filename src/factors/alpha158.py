"""
Alpha158 因子库
基于 Qlib Alpha158 的实现
包含 158 个技术指标因子
"""
import pandas as pd
import numpy as np
from .operators import *


class Alpha158:
    """
    Alpha158 因子计算器
    输入：DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    输出：DataFrame with 158 factor columns
    """
    
    def __init__(self):
        self.factor_names = []
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有 Alpha158 因子
        
        Args:
            df: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                可以是单支股票或多支股票（带 symbol 索引）
        
        Returns:
            DataFrame with factor columns
        """
        # 确保必要的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 如果是多股票 DataFrame（MultiIndex），需要按股票分组计算
        if isinstance(df.index, pd.MultiIndex):
            return df.groupby(level=0).apply(self._calculate_single_stock)
        else:
            return self._calculate_single_stock(df)
    
    def _calculate_single_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算单支股票的因子
        """
        result = pd.DataFrame(index=df.index)
        
        # 提取基础数据
        open_ = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # VWAP (成交量加权平均价格)
        vwap = (close * volume).rolling(window=1).sum() / volume.rolling(window=1).sum()
        
        # ==================== KBAR 特征 (价格形态) ====================
        # 时间窗口
        windows = [5, 10, 20, 30, 60]
        
        for d in windows:
            # OPEN 特征
            result[f'OPEN_{d}'] = Ref(open_, d) / close - 1
            result[f'OPEN_mean_{d}'] = Mean(open_, d) / close - 1
            result[f'OPEN_std_{d}'] = Std(open_ / close, d)
            
            # HIGH 特征
            result[f'HIGH_{d}'] = Ref(high, d) / close - 1
            result[f'HIGH_mean_{d}'] = Mean(high, d) / close - 1
            result[f'HIGH_std_{d}'] = Std(high / close, d)
            
            # LOW 特征
            result[f'LOW_{d}'] = Ref(low, d) / close - 1
            result[f'LOW_mean_{d}'] = Mean(low, d) / close - 1
            result[f'LOW_std_{d}'] = Std(low / close, d)
            
            # CLOSE 特征
            result[f'CLOSE_{d}'] = Ref(close, d) / close - 1
            result[f'CLOSE_mean_{d}'] = Mean(close, d) / close - 1
            result[f'CLOSE_std_{d}'] = Std(close / close.shift(1) - 1, d)
        
        # ==================== PRICE 特征 (价格动量) ====================
        for d in windows:
            # 收益率
            result[f'ROC_{d}'] = Returns(close, d)
            
            # 移动平均
            result[f'MA_{d}'] = MA(close, d) / close - 1
            
            # 标准差
            result[f'STD_{d}'] = Std(close, d) / close
            
            # 偏度和峰度
            result[f'SKEW_{d}'] = Skewness(close, d)
            result[f'KURT_{d}'] = Kurtosis(close, d)
            
            # 最大最小值
            result[f'MAX_{d}'] = Max(close, d) / close - 1
            result[f'MIN_{d}'] = Min(close, d) / close - 1
            
            # 量价相关性
            result[f'CORR_{d}'] = Corr(close, Log(volume + 1), d)
            
            # 线性回归特征
            result[f'BETA_{d}'] = Slope(close, d)
            result[f'RSQR_{d}'] = Rsquare(close, d)
            result[f'RESI_{d}'] = Resi(close, d) / close
            
            # 时间序列排名
            result[f'TSRANK_{d}'] = TSRank(close, d)
            
            # 相对位置
            result[f'QTLU_{d}'] = Quantile(close, d, 0.8) / close - 1
            result[f'QTLD_{d}'] = Quantile(close, d, 0.2) / close - 1
        
        # ==================== VOLUME 特征 ====================
        for d in windows:
            # 成交量移动平均
            result[f'VOLUME_mean_{d}'] = Mean(volume, d) / (volume + 1)
            result[f'VOLUME_std_{d}'] = Std(volume, d) / (Mean(volume, d) + 1)
            
            # 成交量变化
            result[f'VOLUME_delta_{d}'] = Delta(volume, d) / (Ref(volume, d) + 1)
            
            # 成交量比率
            result[f'VR_{d}'] = volume / (Mean(volume, d) + 1)
        
        # ==================== 技术指标特征 ====================
        # RSI
        result['RSI_6'] = RSI(close, 6)
        result['RSI_12'] = RSI(close, 12)
        result['RSI_24'] = RSI(close, 24)
        
        # MACD
        macd_df = MACD(close, 12, 26, 9)
        result['MACD'] = macd_df['MACD'] / close
        result['MACD_signal'] = macd_df['signal'] / close
        result['MACD_hist'] = macd_df['hist'] / close
        
        # KDJ
        kdj_df = KDJ(high, low, close, 9, 3, 3)
        result['KDJ_K'] = kdj_df['K']
        result['KDJ_D'] = kdj_df['D']
        result['KDJ_J'] = kdj_df['J']
        
        # ATR
        result['ATR_14'] = ATR(high, low, close, 14) / close
        
        # 布林带
        result['BOLL_upper_20'] = BBANDS_UPPER(close, 20, 2) / close - 1
        result['BOLL_lower_20'] = BBANDS_LOWER(close, 20, 2) / close - 1
        result['BOLL_width_20'] = (BBANDS_UPPER(close, 20, 2) - BBANDS_LOWER(close, 20, 2)) / close
        
        # ==================== 价格形态特征 ====================
        # 振幅
        for d in [5, 10, 20]:
            result[f'AMP_{d}'] = (high - low) / (close + 1e-10)
            result[f'AMP_mean_{d}'] = Mean((high - low) / (close + 1e-10), d)
        
        # 上下影线
        body = Abs(close - open_)
        upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
        
        result['UPPER_SHADOW'] = upper_shadow / (body + 1e-10)
        result['LOWER_SHADOW'] = lower_shadow / (body + 1e-10)
        result['BODY_RATIO'] = body / ((high - low) + 1e-10)
        
        # ==================== 波动率特征 ====================
        log_returns = LogReturns(close, 1)
        
        for d in [5, 10, 20, 60]:
            result[f'VOLATILITY_{d}'] = Std(log_returns, d)
            result[f'REALIZED_VOL_{d}'] = Std(log_returns, d) * np.sqrt(252)
        
        # ==================== 交叉特征 ====================
        # 价格与均线交叉
        ma5 = MA(close, 5)
        ma10 = MA(close, 10)
        ma20 = MA(close, 20)
        ma60 = MA(close, 60)
        
        result['MA5_MA10'] = ma5 / ma10 - 1
        result['MA5_MA20'] = ma5 / ma20 - 1
        result['MA10_MA20'] = ma10 / ma20 - 1
        result['MA20_MA60'] = ma20 / ma60 - 1
        result['CLOSE_MA5'] = close / ma5 - 1
        result['CLOSE_MA10'] = close / ma10 - 1
        result['CLOSE_MA20'] = close / ma20 - 1
        result['CLOSE_MA60'] = close / ma60 - 1
        
        # ==================== 高级量价特征 ====================
        # 量价背离
        for d in [5, 10, 20]:
            price_slope = Slope(close, d)
            volume_slope = Slope(volume, d)
            result[f'PV_DIVERGENCE_{d}'] = Sign(price_slope) != Sign(volume_slope)
            result[f'PV_CORR_{d}'] = Corr(close, volume, d)
        
        # 价格加速度
        for d in [5, 10, 20]:
            result[f'ACCELERATION_{d}'] = Delta(Returns(close, 1), d)
        
        # 动量指标
        for d in [5, 10, 20, 30]:
            result[f'MOM_{d}'] = close / Ref(close, d) - 1
            result[f'MOM_std_{d}'] = Std(Returns(close, 1), d)
        
        # ==================== 资金流向特征 ====================
        # 简化版 MFI (Money Flow Index)
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        for d in [5, 10, 20]:
            result[f'MFI_{d}'] = Sum(money_flow, d) / (Sum(volume, d) * close)
        
        # 成交额变化
        turnover = close * volume
        for d in [5, 10, 20]:
            result[f'TURNOVER_mean_{d}'] = Mean(turnover, d) / (turnover + 1)
            result[f'TURNOVER_std_{d}'] = Std(turnover, d) / (Mean(turnover, d) + 1)
        
        # 填充 NaN
        result = result.replace([np.inf, -np.inf], np.nan)
        
        # 保存因子名称
        self.factor_names = result.columns.tolist()
        
        return result
    
    def get_factor_names(self) -> list:
        """
        返回所有因子名称
        """
        return self.factor_names


def calculate_alpha158(df: pd.DataFrame) -> pd.DataFrame:
    """
    便捷函数：计算 Alpha158 因子
    
    Args:
        df: OHLCV DataFrame
    
    Returns:
        DataFrame with Alpha158 factors
    """
    alpha158 = Alpha158()
    return alpha158.calculate(df)


if __name__ == "__main__":
    # 示例用法
    from datetime import datetime, timedelta
    
    # 创建示例数据
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'high': 100 + np.cumsum(np.random.randn(len(dates)) * 2) + np.abs(np.random.randn(len(dates))),
        'low': 100 + np.cumsum(np.random.randn(len(dates)) * 2) - np.abs(np.random.randn(len(dates))),
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # 确保 high >= low
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # 计算因子
    print("计算 Alpha158 因子...")
    alpha158 = Alpha158()
    factors = alpha158.calculate(df)
    
    print(f"\n生成因子数量: {len(factors.columns)}")
    print(f"数据行数: {len(factors)}")
    print(f"\n前 10 个因子:")
    print(factors.iloc[:5, :10])
    
    print(f"\n因子统计信息:")
    print(factors.describe())

