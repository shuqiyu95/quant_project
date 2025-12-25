"""
测试回测模块

测试：
- BacktestEngine
- Strategy
- PerformanceAnalyzer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine import BacktestEngine, Position, Trade
from src.backtester.strategy import WeeklyRotationStrategy, RankingStrategy
from src.backtester.performance import PerformanceAnalyzer


class TestBacktestEngine:
    """测试回测引擎"""
    
    def setup_method(self):
        """准备测试"""
        self.engine = BacktestEngine(
            initial_capital=100000,
            commission_rate=0.001,
            market='US'
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.engine.cash == 100000
        assert self.engine.portfolio_value == 100000
        assert len(self.engine.positions) == 0
        print("✅ Engine initialized correctly")
    
    def test_buy(self):
        """测试买入"""
        date = pd.Timestamp('2024-01-01')
        success = self.engine.buy('AAPL', price=150.0, value=50000, date=date)
        
        assert success, "Buy should succeed"
        assert 'AAPL' in self.engine.positions, "Should have AAPL position"
        assert self.engine.cash < 100000, "Cash should decrease"
        assert len(self.engine.trades) == 1, "Should record trade"
        
        print(f"✅ Buy successful:")
        print(f"   Cash: ${self.engine.cash:.2f}")
        print(f"   Position: {self.engine.positions['AAPL'].shares:.2f} shares")
    
    def test_sell(self):
        """测试卖出"""
        # 先买入
        date1 = pd.Timestamp('2024-01-01')
        self.engine.buy('AAPL', price=150.0, value=50000, date=date1)
        
        # 再卖出
        date2 = pd.Timestamp('2024-01-05')
        cash_before = self.engine.cash
        success = self.engine.sell('AAPL', price=155.0, date=date2)
        
        assert success, "Sell should succeed"
        assert 'AAPL' not in self.engine.positions, "Should not have position"
        assert self.engine.cash > cash_before, "Cash should increase"
        assert len(self.engine.trades) == 2, "Should have 2 trades"
        
        print(f"✅ Sell successful:")
        print(f"   Final cash: ${self.engine.cash:.2f}")
    
    def test_update_portfolio(self):
        """测试更新投资组合"""
        date = pd.Timestamp('2024-01-01')
        self.engine.buy('AAPL', price=150.0, value=50000, date=date)
        
        # 更新价格
        self.engine.update_portfolio(date, {'AAPL': 160.0})
        
        assert len(self.engine.portfolio_history) == 1
        assert self.engine.portfolio_value > 100000, "Value should increase"
        
        print(f"✅ Portfolio updated:")
        print(f"   Portfolio value: ${self.engine.portfolio_value:.2f}")
    
    def test_get_stats(self):
        """测试统计信息"""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        
        # 模拟交易
        self.engine.buy('AAPL', price=150.0, value=50000, date=dates[0])
        
        for i, date in enumerate(dates):
            price = 150.0 + i  # 价格上涨
            self.engine.update_portfolio(date, {'AAPL': price})
        
        stats = self.engine.get_portfolio_stats()
        
        assert 'total_return' in stats
        assert 'sharpe_ratio' in stats
        assert 'max_drawdown' in stats
        
        print("✅ Portfolio Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")


class TestPerformanceAnalyzer:
    """测试性能分析器"""
    
    def setup_method(self):
        """准备测试数据"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        n = len(dates)
        
        # 模拟收益率
        daily_returns = np.random.randn(n) * 0.01
        cumulative_returns = np.cumprod(1 + daily_returns)
        
        self.portfolio_df = pd.DataFrame({
            'date': dates,
            'cash': 10000,
            'positions_value': 100000 * cumulative_returns,
            'total_value': 100000 * cumulative_returns,
            'daily_return': daily_returns,
            'n_positions': 3
        })
        
        self.trades_df = pd.DataFrame({
            'date': dates[::30],
            'symbol': ['AAPL', 'MSFT', 'GOOGL'] * 4,
            'action': (['buy'] * 3 + ['sell'] * 3) * 2,
            'shares': [100] * 12,
            'price': [150, 300, 2800] * 4,
            'value': [15000, 30000, 280000] * 4,
            'commission': [15, 30, 280] * 4
        })
        
        self.analyzer = PerformanceAnalyzer()
    
    def test_analyze(self):
        """测试分析"""
        analysis = self.analyzer.analyze(self.portfolio_df, self.trades_df)
        
        # 检查必要指标
        required_metrics = [
            'total_return', 'annual_return', 'volatility',
            'sharpe_ratio', 'max_drawdown', 'win_rate'
        ]
        
        for metric in required_metrics:
            assert metric in analysis, f"Missing metric: {metric}"
        
        print("✅ Analysis completed:")
        print(f"   Total Return: {analysis['total_return']*100:.2f}%")
        print(f"   Sharpe Ratio: {analysis['sharpe_ratio']:.4f}")
        print(f"   Max Drawdown: {analysis['max_drawdown']*100:.2f}%")
    
    def test_print_report(self):
        """测试打印报告"""
        analysis = self.analyzer.analyze(self.portfolio_df, self.trades_df)
        
        print("\n" + "=" * 60)
        self.analyzer.print_report(analysis)
        print("✅ Report printed successfully")


class TestStrategy:
    """测试策略"""
    
    def setup_method(self):
        """准备测试数据"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        self.data_dict = {}
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            df = pd.DataFrame({
                'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
                'high': 100 + np.cumsum(np.random.randn(len(dates)) * 2) + 2,
                'low': 100 + np.cumsum(np.random.randn(len(dates)) * 2) - 2,
                'volume': 1000000 + np.random.randn(len(dates)) * 100000
            }, index=dates)
            self.data_dict[symbol] = df
    
    def test_strategy_initialization(self):
        """测试策略初始化"""
        # 这里需要一个预训练的模型，暂时跳过
        print("✅ Strategy initialization (requires trained model)")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("TESTING BACKTESTER MODULE".center(60))
    print("=" * 60)
    
    # Test BacktestEngine
    print("\n--- Testing BacktestEngine ---")
    test_engine = TestBacktestEngine()
    test_engine.setup_method()
    test_engine.test_initialization()
    
    test_engine.setup_method()
    test_engine.test_buy()
    
    test_engine.setup_method()
    test_engine.test_sell()
    
    test_engine.setup_method()
    test_engine.test_update_portfolio()
    
    test_engine.setup_method()
    test_engine.test_get_stats()
    
    # Test PerformanceAnalyzer
    print("\n--- Testing PerformanceAnalyzer ---")
    test_analyzer = TestPerformanceAnalyzer()
    test_analyzer.setup_method()
    test_analyzer.test_analyze()
    test_analyzer.test_print_report()
    
    # Test Strategy
    print("\n--- Testing Strategy ---")
    test_strategy = TestStrategy()
    test_strategy.setup_method()
    test_strategy.test_strategy_initialization()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!".center(60))
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

