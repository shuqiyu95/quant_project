"""
Factors Module
Qlib 风格的因子计算库
包含基础算子、Alpha158 和 Alpha360 因子库
"""

# 导出基础算子
from .operators import (
    # 基础算子
    Ref, Delta, Mean, MA, Std, Var, Sum, Min, Max,
    Rank, Corr, Cov, Slope, Rsquare, Resi,
    WMA, EMA, Skewness, Kurtosis, Quantile, Mad,
    Sign, Log, Abs, Power,
    
    # 高级算子
    TSRank, TSMin, TSMax,
    BBANDS_UPPER, BBANDS_LOWER,
    RSI, MACD, KDJ, ATR,
    Returns, LogReturns
)

# 导出因子库
from .alpha158 import Alpha158, calculate_alpha158
from .alpha360 import Alpha360, calculate_alpha360

__all__ = [
    # 基础算子
    'Ref', 'Delta', 'Mean', 'MA', 'Std', 'Var', 'Sum', 'Min', 'Max',
    'Rank', 'Corr', 'Cov', 'Slope', 'Rsquare', 'Resi',
    'WMA', 'EMA', 'Skewness', 'Kurtosis', 'Quantile', 'Mad',
    'Sign', 'Log', 'Abs', 'Power',
    
    # 高级算子
    'TSRank', 'TSMin', 'TSMax',
    'BBANDS_UPPER', 'BBANDS_LOWER',
    'RSI', 'MACD', 'KDJ', 'ATR',
    'Returns', 'LogReturns',
    
    # 因子库
    'Alpha158', 'calculate_alpha158',
    'Alpha360', 'calculate_alpha360',
]

__version__ = '1.0.0'

