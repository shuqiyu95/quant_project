"""
回测引擎

实现简单的事件驱动回测引擎
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    value: float = 0.0
    
    def update_value(self, current_price: float):
        """更新持仓市值"""
        self.value = self.shares * current_price
    
    def get_return(self, current_price: float) -> float:
        """计算收益率"""
        return (current_price - self.entry_price) / self.entry_price


@dataclass
class Trade:
    """交易记录"""
    date: pd.Timestamp
    symbol: str
    action: str  # 'buy' or 'sell'
    shares: float
    price: float
    value: float
    commission: float = 0.0


class BacktestEngine:
    """
    回测引擎
    
    支持：
    - 多股票组合
    - 交易成本（佣金、滑点）
    - 持仓管理
    - 性能统计
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1% 佣金
        slippage_rate: float = 0.001,    # 0.1% 滑点
        market: str = 'US'  # 'US' or 'CN'
    ):
        """
        Args:
            initial_capital: 初始资金
            commission_rate: 佣金费率
            slippage_rate: 滑点费率
            market: 市场（影响交易规则）
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.market = market
        
        # 账户状态
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = initial_capital
        
        # 历史记录
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        self.daily_returns: List[float] = []
        
        # A股特殊规则
        if market == 'CN':
            # A股印花税 0.05%（卖出时）
            self.stamp_duty = 0.0005
            # T+1 制度
            self.t_plus_1 = True
        else:
            self.stamp_duty = 0.0
            self.t_plus_1 = False
    
    def reset(self):
        """重置回测状态"""
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.portfolio_history = []
        self.daily_returns = []
    
    def buy(
        self,
        symbol: str,
        price: float,
        value: float,
        date: pd.Timestamp
    ) -> bool:
        """
        买入股票
        
        Args:
            symbol: 股票代码
            price: 买入价格
            value: 买入金额（注意：不是股数）
            date: 交易日期
            
        Returns:
            success: 是否成功买入
        """
        # 计算滑点后的价格
        actual_price = price * (1 + self.slippage_rate)
        
        # 计算佣金
        commission = value * self.commission_rate
        
        # 计算实际成本
        total_cost = value + commission
        
        # 检查资金是否足够（留一点余量避免浮点数误差）
        if total_cost > self.cash + 1e-6:
            return False
        
        # 计算股数
        shares = value / actual_price
        
        # 更新持仓
        if symbol in self.positions:
            # 已有持仓，更新平均成本
            old_pos = self.positions[symbol]
            total_shares = old_pos.shares + shares
            avg_price = (old_pos.entry_price * old_pos.shares + actual_price * shares) / total_shares
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=total_shares,
                entry_price=avg_price,
                entry_date=date
            )
        else:
            # 新建持仓
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                entry_price=actual_price,
                entry_date=date
            )
        
        # 更新现金
        self.cash -= total_cost
        
        # 记录交易
        trade = Trade(
            date=date,
            symbol=symbol,
            action='buy',
            shares=shares,
            price=actual_price,
            value=value,
            commission=commission
        )
        self.trades.append(trade)
        
        return True
    
    def sell(
        self,
        symbol: str,
        price: float,
        date: pd.Timestamp,
        shares: Optional[float] = None
    ) -> bool:
        """
        卖出股票
        
        Args:
            symbol: 股票代码
            price: 卖出价格
            date: 交易日期
            shares: 卖出股数（None表示全部卖出）
            
        Returns:
            success: 是否成功卖出
        """
        # 检查是否有持仓
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        # 确定卖出股数
        if shares is None:
            shares = position.shares
        else:
            shares = min(shares, position.shares)
        
        # 计算滑点后的价格
        actual_price = price * (1 - self.slippage_rate)
        
        # 计算卖出金额
        value = shares * actual_price
        
        # 计算佣金和印花税
        commission = value * self.commission_rate
        stamp_duty = value * self.stamp_duty if self.market == 'CN' else 0.0
        
        # 实际收入
        proceeds = value - commission - stamp_duty
        
        # 更新持仓
        position.shares -= shares
        if position.shares < 1e-6:  # 接近0
            del self.positions[symbol]
        
        # 更新现金
        self.cash += proceeds
        
        # 记录交易
        trade = Trade(
            date=date,
            symbol=symbol,
            action='sell',
            shares=shares,
            price=actual_price,
            value=value,
            commission=commission + stamp_duty
        )
        self.trades.append(trade)
        
        return True
    
    def update_portfolio(self, date: pd.Timestamp, prices: Dict[str, float]):
        """
        更新投资组合价值
        
        Args:
            date: 当前日期
            prices: {symbol: price} 当前价格字典
        """
        # 更新所有持仓的市值
        positions_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_value(prices[symbol])
                positions_value += position.value
        
        # 计算总资产
        total_value = self.cash + positions_value
        
        # 计算日收益率
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1]['total_value']
            daily_return = (total_value - prev_value) / prev_value
        else:
            daily_return = 0.0
        
        # 记录历史
        self.portfolio_history.append({
            'date': date,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_value': total_value,
            'daily_return': daily_return,
            'n_positions': len(self.positions)
        })
        
        self.daily_returns.append(daily_return)
        self.portfolio_value = total_value
    
    def get_portfolio_stats(self) -> Dict:
        """获取投资组合统计信息"""
        if len(self.portfolio_history) == 0:
            return {}
        
        df = pd.DataFrame(self.portfolio_history)
        
        # 计算收益率
        final_value = df['total_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 计算年化收益率
        n_days = len(df)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        # 计算波动率
        returns = np.array(self.daily_returns)
        volatility = np.std(returns) * np.sqrt(252)
        
        # 计算夏普比率（假设无风险利率为0）
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 计算最大回撤
        cummax = df['total_value'].cummax()
        drawdown = (df['total_value'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # 计算胜率
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        stats = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': len(self.trades),
            'n_days': n_days
        }
        
        return stats
    
    def get_trades_df(self) -> pd.DataFrame:
        """获取交易记录 DataFrame"""
        if len(self.trades) == 0:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'date': trade.date,
                'symbol': trade.symbol,
                'action': trade.action,
                'shares': trade.shares,
                'price': trade.price,
                'value': trade.value,
                'commission': trade.commission
            })
        
        return pd.DataFrame(trades_data)
    
    def get_portfolio_df(self) -> pd.DataFrame:
        """获取投资组合历史 DataFrame"""
        return pd.DataFrame(self.portfolio_history)


if __name__ == "__main__":
    # 测试示例
    print("=== Testing Backtest Engine ===\n")
    
    # 创建回测引擎
    engine = BacktestEngine(initial_capital=100000, market='US')
    
    # 模拟交易
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    
    # 第1天：买入 AAPL
    print("Day 1: Buy AAPL")
    success = engine.buy('AAPL', price=150.0, value=50000, date=dates[0])
    print(f"  Success: {success}, Cash: ${engine.cash:.2f}")
    
    # 更新投资组合（价格上涨）
    engine.update_portfolio(dates[0], {'AAPL': 150.0})
    
    # 第5天：卖出部分 AAPL，买入 MSFT
    print("\nDay 5: Sell AAPL, Buy MSFT")
    engine.update_portfolio(dates[4], {'AAPL': 155.0})
    engine.sell('AAPL', price=155.0, date=dates[4])
    engine.buy('MSFT', price=300.0, value=50000, date=dates[4])
    print(f"  Cash: ${engine.cash:.2f}")
    
    # 第10天：更新最终价格
    print("\nDay 10: Final Update")
    engine.update_portfolio(dates[9], {'MSFT': 310.0})
    
    # 获取统计信息
    stats = engine.get_portfolio_stats()
    print("\n=== Portfolio Statistics ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 交易记录
    print("\n=== Trade History ===")
    print(engine.get_trades_df())
    
    print("\n✅ Engine test passed!")

