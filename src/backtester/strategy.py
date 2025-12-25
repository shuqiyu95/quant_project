"""
交易策略

实现多种选股策略：
1. 周轮动策略（每周一调仓）
2. 排序策略（基于预测排名）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
    
    @abstractmethod
    def generate_signals(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> Dict[str, float]:
        """
        生成交易信号
        
        Args:
            date: 当前日期
            data: {symbol: df} 历史数据
            **kwargs: 其他参数
            
        Returns:
            signals: {symbol: weight} 目标持仓权重
        """
        pass


class WeeklyRotationStrategy(BaseStrategy):
    """
    每周轮动策略
    
    每周一（或第一个交易日）：
    1. 根据预测模型选出最优股票
    2. 卖出当前持仓
    3. 买入新选中的股票
    """
    
    def __init__(
        self,
        predictor,
        feature_engineer,
        top_k: int = 1,
        rebalance_weekday: int = 0  # 0=Monday
    ):
        """
        Args:
            predictor: 预测模型
            feature_engineer: 特征工程器
            top_k: 选择前k只股票
            rebalance_weekday: 调仓日（0=周一, 4=周五）
        """
        super().__init__(name="WeeklyRotation")
        self.predictor = predictor
        self.feature_engineer = feature_engineer
        self.top_k = top_k
        self.rebalance_weekday = rebalance_weekday
        self.last_rebalance_date = None
    
    def should_rebalance(self, date: pd.Timestamp) -> bool:
        """判断是否需要调仓"""
        # 检查是否是目标工作日
        if date.weekday() != self.rebalance_weekday:
            return False
        
        # 检查是否是新的一周
        if self.last_rebalance_date is None:
            return True
        
        # 确保不在同一周内重复调仓
        last_week = self.last_rebalance_date.isocalendar()[1]
        current_week = date.isocalendar()[1]
        
        return current_week != last_week
    
    def generate_signals(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> Dict[str, float]:
        """
        生成交易信号
        
        Returns:
            signals: {symbol: weight}，权重总和为1
        """
        # 检查是否需要调仓
        if not self.should_rebalance(date):
            return {}
        
        # 为每个股票计算特征和预测
        predictions = {}
        
        for symbol, df in data.items():
            # 确保有足够的历史数据
            if len(df) < 60:
                continue
            
            # 只使用到当前日期的数据
            df_current = df[df.index <= date]
            if len(df_current) == 0:
                continue
            
            # 计算特征
            features = self.feature_engineer.create_features(df_current)
            
            # 获取最新一天的特征
            if len(features) > 0 and not features.iloc[-1].isna().any():
                X = features.iloc[-1:].values
                pred = self.predictor.predict(X)[0]
                predictions[symbol] = pred
        
        # 选择 top-k
        if len(predictions) < self.top_k:
            return {}
        
        sorted_symbols = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_symbols[:self.top_k]]
        
        # 等权重分配
        weight = 1.0 / self.top_k
        signals = {symbol: weight for symbol in top_symbols}
        
        # 更新调仓日期
        self.last_rebalance_date = date
        
        return signals


class RankingStrategy(BaseStrategy):
    """
    排序策略
    
    更通用的基于排序的策略：
    - 可以每天调仓或按固定频率调仓
    - 支持多种权重方案（等权、按排名权重等）
    """
    
    def __init__(
        self,
        predictor,
        feature_engineer,
        top_k: int = 1,
        rebalance_freq: str = 'W',  # 'D'=每天, 'W'=每周, 'M'=每月
        weight_scheme: str = 'equal'  # 'equal', 'rank', 'score'
    ):
        """
        Args:
            predictor: 预测模型
            feature_engineer: 特征工程器
            top_k: 选择前k只股票
            rebalance_freq: 调仓频率
            weight_scheme: 权重方案
                - 'equal': 等权重
                - 'rank': 按排名权重（排名越高权重越大）
                - 'score': 按预测分数权重
        """
        super().__init__(name="Ranking")
        self.predictor = predictor
        self.feature_engineer = feature_engineer
        self.top_k = top_k
        self.rebalance_freq = rebalance_freq
        self.weight_scheme = weight_scheme
        self.last_rebalance_date = None
    
    def should_rebalance(self, date: pd.Timestamp) -> bool:
        """判断是否需要调仓"""
        if self.last_rebalance_date is None:
            return True
        
        if self.rebalance_freq == 'D':
            return True
        elif self.rebalance_freq == 'W':
            # 每周一次
            last_week = self.last_rebalance_date.isocalendar()[1]
            current_week = date.isocalendar()[1]
            return current_week != last_week
        elif self.rebalance_freq == 'M':
            # 每月一次
            return date.month != self.last_rebalance_date.month
        else:
            return False
    
    def calculate_weights(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """
        根据预测值计算权重
        
        Args:
            predictions: {symbol: predicted_return}
            
        Returns:
            weights: {symbol: weight}
        """
        if len(predictions) == 0:
            return {}
        
        # 选择 top-k
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:self.top_k]
        
        if self.weight_scheme == 'equal':
            # 等权重
            weight = 1.0 / len(top_items)
            return {symbol: weight for symbol, _ in top_items}
        
        elif self.weight_scheme == 'rank':
            # 按排名权重：排名越高（数字越小）权重越大
            # 权重 = (k - rank + 1) / sum(1 to k)
            k = len(top_items)
            total = k * (k + 1) / 2
            weights = {}
            for rank, (symbol, _) in enumerate(top_items, 1):
                weights[symbol] = (k - rank + 1) / total
            return weights
        
        elif self.weight_scheme == 'score':
            # 按预测分数权重（确保都是正数）
            scores = np.array([score for _, score in top_items])
            
            # 如果有负数，平移到正数
            if scores.min() < 0:
                scores = scores - scores.min() + 1e-6
            
            # 归一化
            scores = scores / scores.sum()
            
            return {symbol: weight for (symbol, _), weight in zip(top_items, scores)}
        
        else:
            raise ValueError(f"Unknown weight scheme: {self.weight_scheme}")
    
    def generate_signals(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> Dict[str, float]:
        """生成交易信号"""
        # 检查是否需要调仓
        if not self.should_rebalance(date):
            return {}
        
        # 为每个股票进行预测
        predictions = {}
        
        for symbol, df in data.items():
            if len(df) < 60:
                continue
            
            df_current = df[df.index <= date]
            if len(df_current) == 0:
                continue
            
            # 计算特征
            features = self.feature_engineer.create_features(df_current)
            
            # 获取最新特征并预测
            if len(features) > 0:
                latest_features = features.iloc[-1:]
                if not latest_features.isna().any(axis=1).iloc[0]:
                    X = latest_features.values
                    pred = self.predictor.predict(X)[0]
                    predictions[symbol] = pred
        
        # 计算权重
        signals = self.calculate_weights(predictions)
        
        # 更新调仓日期
        if len(signals) > 0:
            self.last_rebalance_date = date
        
        return signals


def run_backtest_with_strategy(
    engine,
    strategy: BaseStrategy,
    data: Dict[str, pd.DataFrame],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    verbose: bool = True
) -> Dict:
    """
    使用策略运行回测
    
    Args:
        engine: BacktestEngine 实例
        strategy: 策略实例
        data: {symbol: df} 数据字典
        start_date: 开始日期
        end_date: 结束日期
        verbose: 是否打印进度
        
    Returns:
        results: 回测结果
    """
    # 重置引擎
    engine.reset()
    
    # 获取所有交易日期（取所有股票日期的并集）
    all_dates = set()
    for df in data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)
    
    # 过滤日期范围
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]
    
    # 记录当前持仓目标
    current_targets = {}
    
    # 遍历每个交易日
    for i, date in enumerate(all_dates):
        # 生成交易信号
        signals = strategy.generate_signals(date, data)
        
        # 如果有新信号，进行调仓
        if len(signals) > 0:
            if verbose:
                print(f"\n[{date.date()}] Rebalancing...")
                print(f"  New positions: {signals}")
            
            # 1. 计算当前持仓市值（使用当前价格）
            positions_value = 0.0
            for symbol in list(engine.positions.keys()):
                if symbol in data and date in data[symbol].index:
                    current_price = data[symbol].loc[date, 'close']
                    positions_value += engine.positions[symbol].shares * current_price
            
            # 总资本 = 现金 + 持仓市值
            total_capital = engine.cash + positions_value
            
            if verbose:
                print(f"  Total capital: ${total_capital:.2f} (Cash: ${engine.cash:.2f}, Positions: ${positions_value:.2f})")
            
            # 2. 卖出所有当前持仓
            current_positions = list(engine.positions.keys())
            for symbol in current_positions:
                if symbol in data and date in data[symbol].index:
                    price = data[symbol].loc[date, 'close']
                    if verbose:
                        print(f"  Selling {symbol}")
                    engine.sell(symbol, price, date)
            
            # 3. 买入新目标持仓
            for symbol, weight in signals.items():
                # 计算目标金额（考虑交易成本）
                # 总成本 = value * (1 + commission_rate) ≤ target_capital
                # value = target_capital / (1 + commission_rate)
                target_capital = total_capital * weight
                target_value = target_capital / (1 + engine.commission_rate)
                
                # 获取当前价格
                if symbol in data and date in data[symbol].index:
                    price = data[symbol].loc[date, 'close']
                    
                    # 买入目标金额
                    success = engine.buy(symbol, price, target_value, date)
                    if verbose:
                        status = "✅" if success else "❌"
                        print(f"  {status} Buying {symbol}: ${target_value:.2f}")
            
            current_targets = signals
        
        # 更新投资组合价值
        current_prices = {}
        for symbol in engine.positions.keys():
            if symbol in data and date in data[symbol].index:
                current_prices[symbol] = data[symbol].loc[date, 'close']
        
        engine.update_portfolio(date, current_prices)
        
        # 打印进度
        if verbose and (i + 1) % 50 == 0:
            stats = engine.get_portfolio_stats()
            print(f"  Day {i+1}/{len(all_dates)}: Portfolio Value = ${stats['final_value']:.2f}, Return = {stats['total_return']*100:.2f}%")
    
    # 返回结果
    results = {
        'engine': engine,
        'stats': engine.get_portfolio_stats(),
        'trades': engine.get_trades_df(),
        'portfolio': engine.get_portfolio_df()
    }
    
    return results


if __name__ == "__main__":
    # 测试示例
    print("=== Testing Strategy ===\n")
    
    # 创建模拟数据
    from datetime import datetime, timedelta
    
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # 模拟3只股票
    data = {}
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        df = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
            'high': 100 + np.cumsum(np.random.randn(len(dates)) * 2) + 2,
            'low': 100 + np.cumsum(np.random.randn(len(dates)) * 2) - 2,
            'volume': 1000000 + np.random.randn(len(dates)) * 100000
        }, index=dates)
        data[symbol] = df
    
    print("Created mock data for 3 stocks")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")
    
    # 测试信号生成（需要先训练模型，这里跳过）
    print("\n✅ Strategy module created!")
    print("Note: Full test requires trained predictor model")

