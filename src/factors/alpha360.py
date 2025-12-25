"""
Alpha360 因子库
扩展版因子库，包含更多时间窗口和高级因子
在 Alpha158 基础上增加了更多的技术指标和特征工程
"""
import pandas as pd
import numpy as np
from .operators import *
from .alpha158 import Alpha158


class Alpha360:
    """
    Alpha360 因子计算器
    包含 Alpha158 的所有因子 + 额外的高级因子
    """
    
    def __init__(self, include_alpha158: bool = True):
        """
        Args:
            include_alpha158: 是否包含 Alpha158 的基础因子
        """
        self.include_alpha158 = include_alpha158
        self.factor_names = []
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有 Alpha360 因子
        
        Args:
            df: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        
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
        # 1. 先计算 Alpha158 基础因子
        if self.include_alpha158:
            alpha158 = Alpha158()
            result = alpha158.calculate(df)
        else:
            result = pd.DataFrame(index=df.index)
        
        # 提取基础数据
        open_ = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # ==================== 扩展时间窗口 ====================
        extended_windows = [3, 7, 14, 21, 40, 80, 120, 180]
        
        # ==================== 价格动量扩展 ====================
        for d in extended_windows:
            # ROC (Rate of Change)
            result[f'ROC_ext_{d}'] = Returns(close, d)
            
            # 动量强度
            result[f'MOM_ext_{d}'] = close / Ref(close, d) - 1
            
            # 加权移动平均
            result[f'WMA_ext_{d}'] = WMA(close, d) / close - 1
            
            # 指数移动平均
            result[f'EMA_ext_{d}'] = EMA(close, d) / close - 1
            
            # 价格振荡器
            result[f'OSCIL_ext_{d}'] = (close - Mean(close, d)) / Std(close, d)
        
        # ==================== 多周期交叉特征 ====================
        ma_periods = [3, 5, 7, 10, 14, 20, 30, 40, 60, 80, 120]
        mas = {d: MA(close, d) for d in ma_periods}
        
        # MA 交叉组合
        cross_pairs = [(3, 7), (5, 10), (7, 14), (10, 20), (20, 40), 
                       (30, 60), (40, 80), (60, 120)]
        
        for fast, slow in cross_pairs:
            result[f'MA_CROSS_{fast}_{slow}'] = mas[fast] / mas[slow] - 1
            result[f'CLOSE_MA_DIFF_{fast}'] = close / mas[fast] - 1
        
        # ==================== 波动率扩展 ====================
        log_returns = LogReturns(close, 1)
        
        for d in extended_windows:
            # 历史波动率
            result[f'HV_ext_{d}'] = Std(log_returns, d) * np.sqrt(252)
            
            # Parkinson 波动率（使用高低价）
            result[f'PARKINSON_VOL_{d}'] = Std(Log(high / low), d) * np.sqrt(252 / (4 * np.log(2)))
            
            # Garman-Klass 波动率
            hl = Log(high / low) ** 2
            co = Log(close / open_) ** 2
            result[f'GK_VOL_{d}'] = Std(0.5 * hl - (2 * np.log(2) - 1) * co, d) * np.sqrt(252)
            
            # 波动率的波动率
            vol = Std(log_returns, d)
            result[f'VOL_OF_VOL_{d}'] = Std(vol, d)
        
        # ==================== 高级技术指标 ====================
        # Williams %R
        for d in [14, 21, 42]:
            highest_high = Max(high, d)
            lowest_low = Min(low, d)
            result[f'WILLR_{d}'] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        
        # CCI (Commodity Channel Index)
        for d in [14, 20, 30]:
            typical_price = (high + low + close) / 3
            tp_mean = Mean(typical_price, d)
            tp_mad = Mad(typical_price, d)
            result[f'CCI_{d}'] = (typical_price - tp_mean) / (0.015 * tp_mad + 1e-10)
        
        # Stochastic Oscillator
        for d in [9, 14, 21]:
            lowest_low = Min(low, d)
            highest_high = Max(high, d)
            k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
            result[f'STOCH_K_{d}'] = k
            result[f'STOCH_D_{d}'] = Mean(k, 3)
        
        # ROC 多周期
        for d in [5, 10, 20, 40, 60]:
            result[f'ROCP_{d}'] = (close - Ref(close, d)) / Ref(close, d)
            result[f'ROCR_{d}'] = close / Ref(close, d)
        
        # ==================== 量能分析扩展 ====================
        for d in extended_windows:
            # OBV (On Balance Volume)
            price_change = Sign(close - Ref(close, 1))
            obv = (price_change * volume).cumsum()
            result[f'OBV_{d}'] = (obv - Mean(obv, d)) / (Std(obv, d) + 1)
            
            # 量比
            result[f'VR_ext_{d}'] = volume / (Mean(volume, d) + 1)
            
            # 成交量波动率
            result[f'VOLUME_VOL_{d}'] = Std(volume, d) / (Mean(volume, d) + 1)
            
            # 价量背离度
            price_trend = Slope(close, d)
            volume_trend = Slope(volume, d)
            result[f'PV_DIVERGENCE_ext_{d}'] = price_trend * volume_trend
        
        # ==================== 价格形态识别 ====================
        # Doji 形态（十字星）
        body = Abs(close - open_)
        total_range = high - low + 1e-10
        result['DOJI_PATTERN'] = (body / total_range < 0.1).astype(float)
        
        # Hammer 形态（锤子线）
        upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
        result['HAMMER_PATTERN'] = ((lower_shadow / total_range > 0.6) & 
                                    (upper_shadow / total_range < 0.1)).astype(float)
        
        # 涨跌幅度
        for d in [1, 3, 5, 10, 20]:
            result[f'PRICE_CHANGE_{d}'] = (close - Ref(close, d)) / Ref(close, d)
            result[f'HIGH_CHANGE_{d}'] = (high - Ref(high, d)) / Ref(high, d)
            result[f'LOW_CHANGE_{d}'] = (low - Ref(low, d)) / Ref(low, d)
        
        # ==================== 相对强弱扩展 ====================
        # 多周期 RSI
        for d in [6, 12, 24, 48]:
            result[f'RSI_ext_{d}'] = RSI(close, d)
        
        # RSI 背离
        rsi_14 = RSI(close, 14)
        for d in [5, 10, 20]:
            result[f'RSI_SLOPE_{d}'] = Slope(rsi_14, d)
        
        # ==================== 均值回归指标 ====================
        for d in [10, 20, 30, 60]:
            ma = MA(close, d)
            std = Std(close, d)
            
            # Z-Score
            result[f'ZSCORE_{d}'] = (close - ma) / (std + 1e-10)
            
            # 乖离率
            result[f'BIAS_{d}'] = (close - ma) / (ma + 1e-10)
            
            # 布林带位置
            upper = ma + 2 * std
            lower = ma - 2 * std
            result[f'BOLL_POS_{d}'] = (close - lower) / (upper - lower + 1e-10)
        
        # ==================== 趋势强度 ====================
        for d in [10, 20, 30, 60]:
            # ADX (Average Directional Index)
            # 简化版本
            up_move = high - Ref(high, 1)
            down_move = Ref(low, 1) - low
            
            plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index)
            minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=low.index)
            
            atr = ATR(high, low, close, d)
            plus_di = 100 * EMA(plus_dm, d) / atr
            minus_di = 100 * EMA(minus_dm, d) / atr
            
            dx = 100 * Abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            result[f'ADX_{d}'] = EMA(dx, d)
            result[f'PLUS_DI_{d}'] = plus_di
            result[f'MINUS_DI_{d}'] = minus_di
        
        # ==================== 高频特征（模拟） ====================
        # 价格跳跃
        result['PRICE_JUMP_1'] = Abs(Returns(close, 1)) > (3 * Std(Returns(close, 1), 20))
        
        # 价格加速度
        for d in [3, 5, 10, 20]:
            vel = Delta(close, 1)
            result[f'ACCELERATION_{d}'] = Delta(vel, d)
        
        # 价格动能
        for d in [5, 10, 20]:
            result[f'MOMENTUM_STRENGTH_{d}'] = Sum(Returns(close, 1) * volume, d) / Sum(volume, d)
        
        # ==================== 多因子组合特征 ====================
        # 价量综合指标
        price_mom = Returns(close, 20)
        volume_ratio = volume / Mean(volume, 20)
        result['PV_COMPOSITE_20'] = price_mom * Log(volume_ratio + 1)
        
        # 动量-波动率比率
        for d in [10, 20, 40]:
            mom = Returns(close, d)
            vol = Std(Returns(close, 1), d)
            result[f'MOM_VOL_RATIO_{d}'] = mom / (vol + 1e-10)
        
        # 趋势质量
        for d in [10, 20, 40]:
            slope = Slope(close, d)
            r2 = Rsquare(close, d)
            result[f'TREND_QUALITY_{d}'] = slope * r2
        
        # ==================== 季节性和周期性特征 ====================
        # 月度效应（如果有日期索引）
        if isinstance(df.index, pd.DatetimeIndex):
            result['MONTH'] = df.index.month / 12
            result['WEEKDAY'] = df.index.weekday / 7
            result['QUARTER'] = df.index.quarter / 4
        
        # ==================== 流动性指标 ====================
        for d in [5, 10, 20]:
            # Amihud 非流动性指标
            abs_return = Abs(Returns(close, 1))
            dollar_volume = close * volume
            result[f'AMIHUD_{d}'] = Mean(abs_return / (dollar_volume + 1), d)
            
            # 换手率（需要流通股本，这里用成交量代替）
            result[f'TURNOVER_RATIO_{d}'] = volume / Mean(volume, 60)
        
        # ==================== 高阶统计量 ====================
        for d in [10, 20, 40, 60]:
            returns = Returns(close, 1)
            
            # 偏度
            result[f'SKEW_ext_{d}'] = Skewness(returns, d)
            
            # 峰度
            result[f'KURT_ext_{d}'] = Kurtosis(returns, d)
            
            # 下偏标准差（下行风险）
            downside_returns = pd.Series(np.where(returns < 0, returns, 0), index=returns.index)
            result[f'DOWNSIDE_STD_{d}'] = Std(downside_returns, d)
            
            # 上行标准差
            upside_returns = pd.Series(np.where(returns > 0, returns, 0), index=returns.index)
            result[f'UPSIDE_STD_{d}'] = Std(upside_returns, d)
        
        # 填充 NaN 和无穷大
        result = result.replace([np.inf, -np.inf], np.nan)
        
        # 保存因子名称
        self.factor_names = result.columns.tolist()
        
        return result
    
    def get_factor_names(self) -> list:
        """
        返回所有因子名称
        """
        return self.factor_names


def calculate_alpha360(df: pd.DataFrame, include_alpha158: bool = True) -> pd.DataFrame:
    """
    便捷函数：计算 Alpha360 因子
    
    Args:
        df: OHLCV DataFrame
        include_alpha158: 是否包含 Alpha158 基础因子
    
    Returns:
        DataFrame with Alpha360 factors
    """
    alpha360 = Alpha360(include_alpha158=include_alpha158)
    return alpha360.calculate(df)


if __name__ == "__main__":
    # 示例用法
    from datetime import datetime
    
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
    print("计算 Alpha360 因子...")
    alpha360 = Alpha360(include_alpha158=True)
    factors = alpha360.calculate(df)
    
    print(f"\n生成因子数量: {len(factors.columns)}")
    print(f"数据行数: {len(factors)}")
    print(f"\n前 5 行，前 10 个因子:")
    print(factors.iloc[:5, :10])
    
    print(f"\n因子统计信息:")
    print(factors.describe())

