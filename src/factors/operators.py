"""
Qlib-style Operators
基础算子实现，用于因子计算
所有算子支持 Pandas 向量化运算
"""
import pandas as pd
import numpy as np
from typing import Union, Optional


# ==================== 基础算子 ====================

def Ref(series: pd.Series, d: int) -> pd.Series:
    """
    引用 d 天前的值
    Args:
        series: 输入序列
        d: 天数，正数表示往前看
    Returns:
        shifted series
    """
    return series.shift(d)


def Delta(series: pd.Series, d: int = 1) -> pd.Series:
    """
    当前值与 d 天前的差值
    Delta(close, d) = close - Ref(close, d)
    """
    return series - series.shift(d)


def Mean(series: pd.Series, d: int) -> pd.Series:
    """
    d 天移动平均
    """
    return series.rolling(window=d, min_periods=1).mean()


def MA(series: pd.Series, d: int) -> pd.Series:
    """
    移动平均别名
    """
    return Mean(series, d)


def Std(series: pd.Series, d: int) -> pd.Series:
    """
    d 天标准差
    """
    return series.rolling(window=d, min_periods=1).std()


def Var(series: pd.Series, d: int) -> pd.Series:
    """
    d 天方差
    """
    return series.rolling(window=d, min_periods=1).var()


def Sum(series: pd.Series, d: int) -> pd.Series:
    """
    d 天累加和
    """
    return series.rolling(window=d, min_periods=1).sum()


def Min(series: pd.Series, d: int) -> pd.Series:
    """
    d 天最小值
    """
    return series.rolling(window=d, min_periods=1).min()


def Max(series: pd.Series, d: int) -> pd.Series:
    """
    d 天最大值
    """
    return series.rolling(window=d, min_periods=1).max()


def Rank(series: pd.Series) -> pd.Series:
    """
    横截面排名（在同一时刻对不同股票排序）
    返回 [0, 1] 之间的值
    """
    return series.rank(pct=True)


def Corr(s1: pd.Series, s2: pd.Series, d: int) -> pd.Series:
    """
    d 天相关系数
    """
    return s1.rolling(window=d, min_periods=1).corr(s2)


def Cov(s1: pd.Series, s2: pd.Series, d: int) -> pd.Series:
    """
    d 天协方差
    """
    return s1.rolling(window=d, min_periods=1).cov(s2)


def Slope(series: pd.Series, d: int) -> pd.Series:
    """
    d 天线性回归斜率
    使用最小二乘法拟合：y = a + b*x，返回 b
    """
    def _slope(x):
        if len(x) < 2 or x.isna().all():
            return np.nan
        y = x.dropna().values
        if len(y) < 2:
            return np.nan
        X = np.arange(len(y))
        # 计算斜率：b = cov(X,Y) / var(X)
        slope = np.polyfit(X, y, 1)[0]
        return slope
    
    return series.rolling(window=d, min_periods=2).apply(_slope, raw=False)


def Rsquare(series: pd.Series, d: int) -> pd.Series:
    """
    d 天线性回归 R²
    """
    def _rsquare(x):
        if len(x) < 2 or x.isna().all():
            return np.nan
        y = x.dropna().values
        if len(y) < 2:
            return np.nan
        X = np.arange(len(y))
        try:
            p = np.polyfit(X, y, 1)
            y_pred = np.polyval(p, X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            if ss_tot == 0:
                return np.nan
            r2 = 1 - (ss_res / ss_tot)
            return r2
        except:
            return np.nan
    
    return series.rolling(window=d, min_periods=2).apply(_rsquare, raw=False)


def Resi(series: pd.Series, d: int) -> pd.Series:
    """
    d 天线性回归残差
    """
    def _resi(x):
        if len(x) < 2 or x.isna().all():
            return np.nan
        y = x.dropna().values
        if len(y) < 2:
            return np.nan
        X = np.arange(len(y))
        try:
            p = np.polyfit(X, y, 1)
            y_pred = np.polyval(p, X)[-1]  # 最后一个预测值
            return y[-1] - y_pred
        except:
            return np.nan
    
    return series.rolling(window=d, min_periods=2).apply(_resi, raw=False)


def WMA(series: pd.Series, d: int) -> pd.Series:
    """
    加权移动平均（线性权重）
    权重：1, 2, 3, ..., d
    """
    weights = np.arange(1, d + 1)
    
    def _wma(x):
        if len(x) < d or np.isnan(x).any():
            return np.nan
        return np.dot(x, weights) / weights.sum()
    
    return series.rolling(window=d, min_periods=d).apply(_wma, raw=True)


def EMA(series: pd.Series, d: int) -> pd.Series:
    """
    指数移动平均
    """
    return series.ewm(span=d, adjust=False, min_periods=1).mean()


def Skewness(series: pd.Series, d: int) -> pd.Series:
    """
    d 天偏度
    """
    return series.rolling(window=d, min_periods=1).skew()


def Kurtosis(series: pd.Series, d: int) -> pd.Series:
    """
    d 天峰度
    """
    return series.rolling(window=d, min_periods=1).kurt()


def Quantile(series: pd.Series, d: int, q: float) -> pd.Series:
    """
    d 天分位数
    Args:
        series: 输入序列
        d: 窗口大小
        q: 分位数 (0-1)
    """
    return series.rolling(window=d, min_periods=1).quantile(q)


def Mad(series: pd.Series, d: int) -> pd.Series:
    """
    d 天平均绝对偏差 (Mean Absolute Deviation)
    """
    def _mad(x):
        return np.abs(x - x.mean()).mean()
    
    return series.rolling(window=d, min_periods=1).apply(_mad, raw=True)


def Sign(series: pd.Series) -> pd.Series:
    """
    符号函数：正数返回1，负数返回-1，0返回0
    """
    return np.sign(series)


def Log(series: pd.Series) -> pd.Series:
    """
    自然对数
    """
    return np.log(series)


def Abs(series: pd.Series) -> pd.Series:
    """
    绝对值
    """
    return np.abs(series)


def Power(series: pd.Series, n: float) -> pd.Series:
    """
    幂运算
    """
    return np.power(series, n)


# ==================== 高级算子 ====================

def TSRank(series: pd.Series, d: int) -> pd.Series:
    """
    时间序列排名
    在过去 d 天内，当前值的排名位置（0-1）
    """
    def _tsrank(x):
        if len(x) < 1:
            return np.nan
        if np.all(np.isnan(x)):
            return np.nan
        return pd.Series(x).rank(pct=True).iloc[-1]
    
    return series.rolling(window=d, min_periods=1).apply(_tsrank, raw=True)


def TSMin(series: pd.Series, d: int) -> pd.Series:
    """
    过去 d 天最小值出现的位置（距今天数）
    """
    def _tsmin_idx(x):
        if len(x) < 1:
            return np.nan
        if np.all(np.isnan(x)):
            return np.nan
        return len(x) - 1 - np.argmin(x)
    
    return series.rolling(window=d, min_periods=1).apply(_tsmin_idx, raw=True)


def TSMax(series: pd.Series, d: int) -> pd.Series:
    """
    过去 d 天最大值出现的位置（距今天数）
    """
    def _tsmax_idx(x):
        if len(x) < 1:
            return np.nan
        if np.all(np.isnan(x)):
            return np.nan
        return len(x) - 1 - np.argmax(x)
    
    return series.rolling(window=d, min_periods=1).apply(_tsmax_idx, raw=True)


def BBANDS_UPPER(series: pd.Series, d: int, n_std: float = 2.0) -> pd.Series:
    """
    布林带上轨
    """
    ma = Mean(series, d)
    std = Std(series, d)
    return ma + n_std * std


def BBANDS_LOWER(series: pd.Series, d: int, n_std: float = 2.0) -> pd.Series:
    """
    布林带下轨
    """
    ma = Mean(series, d)
    std = Std(series, d)
    return ma - n_std * std


def RSI(series: pd.Series, d: int = 14) -> pd.Series:
    """
    相对强弱指标
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=d, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=d, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def MACD(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD 指标
    返回 DataFrame: ['MACD', 'signal', 'hist']
    """
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = EMA(macd, signal)
    macd_hist = macd - macd_signal
    
    return pd.DataFrame({
        'MACD': macd,
        'signal': macd_signal,
        'hist': macd_hist
    }, index=series.index)


def KDJ(high: pd.Series, low: pd.Series, close: pd.Series, 
        n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """
    KDJ 指标
    """
    lowest_low = low.rolling(window=n, min_periods=1).min()
    highest_high = high.rolling(window=n, min_periods=1).max()
    
    rsv = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return pd.DataFrame({
        'K': k,
        'D': d,
        'J': j
    }, index=close.index)


def ATR(high: pd.Series, low: pd.Series, close: pd.Series, d: int = 14) -> pd.Series:
    """
    平均真实波幅
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=d, min_periods=1).mean()
    
    return atr


def Returns(series: pd.Series, d: int = 1) -> pd.Series:
    """
    d 期收益率
    """
    return series.pct_change(d)


def LogReturns(series: pd.Series, d: int = 1) -> pd.Series:
    """
    d 期对数收益率
    """
    return np.log(series / series.shift(d))

