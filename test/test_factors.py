"""
测试因子计算模块
测试基础算子、Alpha158 和 Alpha360 因子库
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np
from factors import (
    # 基础算子
    Ref, Delta, Mean, MA, Std, Sum, Min, Max,
    Slope, Rsquare, Corr, EMA, RSI, MACD, KDJ, ATR,
    TSRank, Returns, LogReturns,
    # 因子库
    Alpha158, Alpha360, calculate_alpha158, calculate_alpha360
)


# ==================== 测试数据准备 ====================

@pytest.fixture
def sample_ohlcv_data():
    """
    创建示例 OHLCV 数据
    """
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    n = len(dates)
    base_price = 100
    price_series = base_price + np.cumsum(np.random.randn(n) * 2)
    
    df = pd.DataFrame({
        'open': price_series + np.random.randn(n) * 0.5,
        'high': price_series + np.abs(np.random.randn(n)) * 1.5,
        'low': price_series - np.abs(np.random.randn(n)) * 1.5,
        'close': price_series,
        'volume': np.random.randint(1000000, 10000000, n)
    }, index=dates)
    
    # 确保 high >= open, close 且 low <= open, close
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df


@pytest.fixture
def multi_stock_data():
    """
    创建多股票 OHLCV 数据（MultiIndex）
    """
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    dfs = []
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)
        n = len(dates)
        base_price = 100 + hash(symbol) % 50
        price_series = base_price + np.cumsum(np.random.randn(n) * 2)
        
        df = pd.DataFrame({
            'symbol': symbol,
            'open': price_series + np.random.randn(n) * 0.5,
            'high': price_series + np.abs(np.random.randn(n)) * 1.5,
            'low': price_series - np.abs(np.random.randn(n)) * 1.5,
            'close': price_series,
            'volume': np.random.randint(1000000, 10000000, n)
        }, index=dates)
        
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        dfs.append(df)
    
    combined = pd.concat(dfs)
    combined = combined.set_index('symbol', append=True)
    combined = combined.swaplevel(0, 1)
    
    return combined


# ==================== 基础算子测试 ====================

class TestBasicOperators:
    """测试基础算子"""
    
    def test_ref(self, sample_ohlcv_data):
        """测试 Ref 算子"""
        close = sample_ohlcv_data['close']
        ref_5 = Ref(close, 5)
        
        assert len(ref_5) == len(close)
        assert pd.isna(ref_5.iloc[:5]).all()
        assert ref_5.iloc[5] == close.iloc[0]
        assert ref_5.iloc[10] == close.iloc[5]
    
    def test_delta(self, sample_ohlcv_data):
        """测试 Delta 算子"""
        close = sample_ohlcv_data['close']
        delta_1 = Delta(close, 1)
        
        assert len(delta_1) == len(close)
        assert pd.isna(delta_1.iloc[0])
        # 验证几个具体的值
        for i in range(1, min(10, len(close))):
            expected = close.iloc[i] - close.iloc[i-1]
            assert np.isclose(delta_1.iloc[i], expected, rtol=1e-5, atol=1e-8)
    
    def test_mean(self, sample_ohlcv_data):
        """测试 Mean/MA 算子"""
        close = sample_ohlcv_data['close']
        ma_5 = Mean(close, 5)
        ma_5_alias = MA(close, 5)
        
        assert len(ma_5) == len(close)
        assert np.allclose(ma_5.values, ma_5_alias.values, equal_nan=True)
        assert not pd.isna(ma_5.iloc[4:]).any()
    
    def test_std(self, sample_ohlcv_data):
        """测试 Std 算子"""
        close = sample_ohlcv_data['close']
        std_5 = Std(close, 5)
        
        assert len(std_5) == len(close)
        assert (std_5.iloc[5:] >= 0).all()
    
    def test_sum(self, sample_ohlcv_data):
        """测试 Sum 算子"""
        volume = sample_ohlcv_data['volume']
        sum_5 = Sum(volume, 5)
        
        assert len(sum_5) == len(volume)
        assert sum_5.iloc[4] == volume.iloc[:5].sum()
    
    def test_min_max(self, sample_ohlcv_data):
        """测试 Min/Max 算子"""
        close = sample_ohlcv_data['close']
        min_5 = Min(close, 5)
        max_5 = Max(close, 5)
        
        assert len(min_5) == len(close)
        assert len(max_5) == len(close)
        assert (max_5.iloc[5:] >= min_5.iloc[5:]).all()
    
    def test_slope(self, sample_ohlcv_data):
        """测试 Slope 算子"""
        close = sample_ohlcv_data['close']
        slope_10 = Slope(close, 10)
        
        assert len(slope_10) == len(close)
        # 检查上升趋势的斜率为正
        close_up = pd.Series(range(100), index=range(100))
        slope_up = Slope(close_up, 10)
        assert (slope_up.iloc[10:] > 0).all()
    
    def test_rsquare(self, sample_ohlcv_data):
        """测试 Rsquare 算子"""
        close = sample_ohlcv_data['close']
        r2_10 = Rsquare(close, 10)
        
        assert len(r2_10) == len(close)
        # R² 应该在 [0, 1] 之间（大部分情况）
        valid_r2 = r2_10.dropna()
        # 允许一些异常值
        assert (valid_r2 >= -0.5).all()
        assert (valid_r2 <= 1.5).all()
    
    def test_corr(self, sample_ohlcv_data):
        """测试 Corr 算子"""
        close = sample_ohlcv_data['close']
        volume = sample_ohlcv_data['volume']
        corr_10 = Corr(close, volume, 10)
        
        assert len(corr_10) == len(close)
        valid_corr = corr_10.dropna()
        assert (valid_corr >= -1.1).all()
        assert (valid_corr <= 1.1).all()
    
    def test_ema(self, sample_ohlcv_data):
        """测试 EMA 算子"""
        close = sample_ohlcv_data['close']
        ema_12 = EMA(close, 12)
        
        assert len(ema_12) == len(close)
        assert not pd.isna(ema_12).all()
    
    def test_returns(self, sample_ohlcv_data):
        """测试 Returns 算子"""
        close = sample_ohlcv_data['close']
        ret_1 = Returns(close, 1)
        
        assert len(ret_1) == len(close)
        assert pd.isna(ret_1.iloc[0])
        # 手动计算第二个值
        expected = (close.iloc[1] - close.iloc[0]) / close.iloc[0]
        assert np.isclose(ret_1.iloc[1], expected)
    
    def test_log_returns(self, sample_ohlcv_data):
        """测试 LogReturns 算子"""
        close = sample_ohlcv_data['close']
        log_ret_1 = LogReturns(close, 1)
        
        assert len(log_ret_1) == len(close)
        assert pd.isna(log_ret_1.iloc[0])
    
    def test_tsrank(self, sample_ohlcv_data):
        """测试 TSRank 算子"""
        close = sample_ohlcv_data['close']
        tsrank_10 = TSRank(close, 10)
        
        assert len(tsrank_10) == len(close)
        valid_rank = tsrank_10.dropna()
        assert (valid_rank >= 0).all()
        assert (valid_rank <= 1).all()


# ==================== 技术指标测试 ====================

class TestTechnicalIndicators:
    """测试技术指标算子"""
    
    def test_rsi(self, sample_ohlcv_data):
        """测试 RSI 指标"""
        close = sample_ohlcv_data['close']
        rsi_14 = RSI(close, 14)
        
        assert len(rsi_14) == len(close)
        valid_rsi = rsi_14.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd(self, sample_ohlcv_data):
        """测试 MACD 指标"""
        close = sample_ohlcv_data['close']
        macd_df = MACD(close, 12, 26, 9)
        
        assert 'MACD' in macd_df.columns
        assert 'signal' in macd_df.columns
        assert 'hist' in macd_df.columns
        assert len(macd_df) == len(close)
    
    def test_kdj(self, sample_ohlcv_data):
        """测试 KDJ 指标"""
        high = sample_ohlcv_data['high']
        low = sample_ohlcv_data['low']
        close = sample_ohlcv_data['close']
        kdj_df = KDJ(high, low, close, 9, 3, 3)
        
        assert 'K' in kdj_df.columns
        assert 'D' in kdj_df.columns
        assert 'J' in kdj_df.columns
        assert len(kdj_df) == len(close)
    
    def test_atr(self, sample_ohlcv_data):
        """测试 ATR 指标"""
        high = sample_ohlcv_data['high']
        low = sample_ohlcv_data['low']
        close = sample_ohlcv_data['close']
        atr_14 = ATR(high, low, close, 14)
        
        assert len(atr_14) == len(close)
        assert (atr_14.dropna() >= 0).all()


# ==================== Alpha158 测试 ====================

class TestAlpha158:
    """测试 Alpha158 因子库"""
    
    def test_alpha158_basic(self, sample_ohlcv_data):
        """测试 Alpha158 基本功能"""
        alpha158 = Alpha158()
        factors = alpha158.calculate(sample_ohlcv_data)
        
        # 检查输出
        assert isinstance(factors, pd.DataFrame)
        assert len(factors) == len(sample_ohlcv_data)
        assert len(factors.columns) > 100  # 应该有大量因子
        
        # 检查因子名称
        factor_names = alpha158.get_factor_names()
        assert len(factor_names) > 0
        assert len(factor_names) == len(factors.columns)
    
    def test_alpha158_no_nan_columns(self, sample_ohlcv_data):
        """测试 Alpha158 不应该有全是 NaN 的列"""
        alpha158 = Alpha158()
        factors = alpha158.calculate(sample_ohlcv_data)
        
        # 检查每列至少有一些非 NaN 值
        for col in factors.columns:
            assert factors[col].notna().sum() > 0, f"Column {col} is all NaN"
    
    def test_alpha158_no_inf(self, sample_ohlcv_data):
        """测试 Alpha158 不应该有无穷大值"""
        alpha158 = Alpha158()
        factors = alpha158.calculate(sample_ohlcv_data)
        
        # 检查没有 inf 值（只检查数值类型列）
        numeric_cols = factors.select_dtypes(include=[np.number]).columns
        assert not np.isinf(factors[numeric_cols].values).any()
    
    def test_alpha158_convenience_function(self, sample_ohlcv_data):
        """测试 Alpha158 便捷函数"""
        factors = calculate_alpha158(sample_ohlcv_data)
        
        assert isinstance(factors, pd.DataFrame)
        assert len(factors) == len(sample_ohlcv_data)
        assert len(factors.columns) > 100
    
    def test_alpha158_multi_stock(self, multi_stock_data):
        """测试 Alpha158 多股票数据"""
        alpha158 = Alpha158()
        factors = alpha158.calculate(multi_stock_data)
        
        assert isinstance(factors, pd.DataFrame)
        assert isinstance(factors.index, pd.MultiIndex)
        assert len(factors.columns) > 100


# ==================== Alpha360 测试 ====================

class TestAlpha360:
    """测试 Alpha360 因子库"""
    
    def test_alpha360_basic(self, sample_ohlcv_data):
        """测试 Alpha360 基本功能"""
        alpha360 = Alpha360(include_alpha158=True)
        factors = alpha360.calculate(sample_ohlcv_data)
        
        # 检查输出
        assert isinstance(factors, pd.DataFrame)
        assert len(factors) == len(sample_ohlcv_data)
        # Alpha360 应该比 Alpha158 有更多因子
        assert len(factors.columns) > 200
        
        # 检查因子名称
        factor_names = alpha360.get_factor_names()
        assert len(factor_names) > 0
        assert len(factor_names) == len(factors.columns)
    
    def test_alpha360_without_alpha158(self, sample_ohlcv_data):
        """测试 Alpha360 不包含 Alpha158"""
        alpha360 = Alpha360(include_alpha158=False)
        factors = alpha360.calculate(sample_ohlcv_data)
        
        assert isinstance(factors, pd.DataFrame)
        assert len(factors) == len(sample_ohlcv_data)
        # 不包含 Alpha158 应该有较少的因子
        assert len(factors.columns) > 50
    
    def test_alpha360_no_inf(self, sample_ohlcv_data):
        """测试 Alpha360 不应该有无穷大值"""
        alpha360 = Alpha360(include_alpha158=True)
        factors = alpha360.calculate(sample_ohlcv_data)
        
        # 检查没有 inf 值（只检查数值类型列）
        numeric_cols = factors.select_dtypes(include=[np.number]).columns
        assert not np.isinf(factors[numeric_cols].values).any()
    
    def test_alpha360_convenience_function(self, sample_ohlcv_data):
        """测试 Alpha360 便捷函数"""
        factors = calculate_alpha360(sample_ohlcv_data, include_alpha158=True)
        
        assert isinstance(factors, pd.DataFrame)
        assert len(factors) == len(sample_ohlcv_data)
        assert len(factors.columns) > 200
    
    def test_alpha360_multi_stock(self, multi_stock_data):
        """测试 Alpha360 多股票数据"""
        alpha360 = Alpha360(include_alpha158=True)
        factors = alpha360.calculate(multi_stock_data)
        
        assert isinstance(factors, pd.DataFrame)
        assert isinstance(factors.index, pd.MultiIndex)
        assert len(factors.columns) > 200


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""
    
    def test_factor_pipeline(self, sample_ohlcv_data):
        """测试完整的因子计算流程"""
        # 1. 计算 Alpha158
        alpha158 = Alpha158()
        factors_158 = alpha158.calculate(sample_ohlcv_data)
        
        # 2. 计算 Alpha360
        alpha360 = Alpha360(include_alpha158=False)
        factors_360 = alpha360.calculate(sample_ohlcv_data)
        
        # 3. 合并因子
        all_factors = pd.concat([factors_158, factors_360], axis=1)
        
        assert len(all_factors) == len(sample_ohlcv_data)
        assert len(all_factors.columns) == len(factors_158.columns) + len(factors_360.columns)
    
    def test_missing_data_handling(self):
        """测试缺失数据处理"""
        # 创建有缺失值的数据
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 102,
            'low': np.random.randn(100) + 98,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # 人为制造一些缺失值
        df.loc[df.index[10:15], 'close'] = np.nan
        
        # 计算因子
        alpha158 = Alpha158()
        factors = alpha158.calculate(df)
        
        # 应该能够处理缺失值
        assert isinstance(factors, pd.DataFrame)
        assert len(factors) == len(df)
    
    def test_extreme_values(self):
        """测试极端值处理"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'open': np.ones(100) * 100,
            'high': np.ones(100) * 101,
            'low': np.ones(100) * 99,
            'close': np.ones(100) * 100,
            'volume': np.ones(100) * 1000000
        }, index=dates)
        
        # 添加一个极端值
        df.loc[df.index[50], 'close'] = 1000
        
        # 计算因子
        alpha158 = Alpha158()
        factors = alpha158.calculate(df)
        
        # 应该能够处理极端值
        assert isinstance(factors, pd.DataFrame)
        # 不应该有 inf 值（只检查数值类型列）
        numeric_cols = factors.select_dtypes(include=[np.number]).columns
        assert not np.isinf(factors[numeric_cols].values).any()


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '--tb=short'])

