"""
性能分析模块

提供详细的回测性能指标和可视化
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """
    性能分析器
    
    计算各种回测性能指标
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: 无风险利率（年化）
        """
        self.risk_free_rate = risk_free_rate
    
    def analyze(self, portfolio_df: pd.DataFrame, trades_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        全面分析回测结果
        
        Args:
            portfolio_df: 投资组合历史数据
            trades_df: 交易记录
            
        Returns:
            analysis: 分析结果字典
        """
        analysis = {}
        
        # 基础收益指标
        analysis.update(self.calculate_returns_metrics(portfolio_df))
        
        # 风险指标
        analysis.update(self.calculate_risk_metrics(portfolio_df))
        
        # 风险调整收益指标
        analysis.update(self.calculate_risk_adjusted_metrics(portfolio_df))
        
        # 交易统计
        if trades_df is not None and len(trades_df) > 0:
            analysis.update(self.calculate_trading_metrics(trades_df))
        
        return analysis
    
    def calculate_returns_metrics(self, portfolio_df: pd.DataFrame) -> Dict:
        """计算收益相关指标"""
        if len(portfolio_df) == 0:
            return {}
        
        total_value = portfolio_df['total_value']
        initial_value = total_value.iloc[0]
        final_value = total_value.iloc[-1]
        
        # 总收益率
        total_return = (final_value - initial_value) / initial_value
        
        # 年化收益率
        n_days = len(portfolio_df)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        # 累计收益率序列
        cumulative_returns = (total_value / initial_value - 1)
        
        # 日收益率
        daily_returns = portfolio_df['daily_return'].values
        
        # 平均日收益率
        avg_daily_return = np.mean(daily_returns)
        
        # 最好和最差的单日收益
        best_day_return = np.max(daily_returns)
        worst_day_return = np.min(daily_returns)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'avg_daily_return': avg_daily_return,
            'best_day_return': best_day_return,
            'worst_day_return': worst_day_return,
            'final_value': final_value
        }
    
    def calculate_risk_metrics(self, portfolio_df: pd.DataFrame) -> Dict:
        """计算风险相关指标"""
        if len(portfolio_df) == 0:
            return {}
        
        daily_returns = portfolio_df['daily_return'].values
        total_value = portfolio_df['total_value']
        
        # 波动率（年化）
        volatility = np.std(daily_returns) * np.sqrt(252)
        
        # 下行波动率（只考虑负收益）
        negative_returns = daily_returns[daily_returns < 0]
        downside_volatility = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        
        # 最大回撤
        cummax = total_value.cummax()
        drawdown = (total_value - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # 最大回撤持续期
        max_dd_duration = self._calculate_max_drawdown_duration(drawdown)
        
        # VaR (Value at Risk) - 95% 置信度
        var_95 = np.percentile(daily_returns, 5)
        
        # CVaR (Conditional VaR) - 95% 置信度下的平均损失
        cvar_95 = np.mean(daily_returns[daily_returns <= var_95]) if len(daily_returns[daily_returns <= var_95]) > 0 else 0
        
        return {
            'volatility': volatility,
            'downside_volatility': downside_volatility,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def calculate_risk_adjusted_metrics(self, portfolio_df: pd.DataFrame) -> Dict:
        """计算风险调整后的收益指标"""
        if len(portfolio_df) == 0:
            return {}
        
        daily_returns = portfolio_df['daily_return'].values
        n_days = len(portfolio_df)
        
        # 计算年化收益
        total_value = portfolio_df['total_value']
        total_return = (total_value.iloc[-1] / total_value.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        # 波动率
        volatility = np.std(daily_returns) * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # 索提诺比率（使用下行波动率）
        negative_returns = daily_returns[daily_returns < 0]
        downside_vol = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Calmar 比率（年化收益 / 最大回撤）
        cummax = total_value.cummax()
        drawdown = (total_value - cummax) / cummax
        max_drawdown = abs(drawdown.min())
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # 信息比率（假设基准收益为0）
        information_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        
        # 胜率
        win_rate = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0
        
        # 盈亏比
        avg_win = np.mean(daily_returns[daily_returns > 0]) if len(daily_returns[daily_returns > 0]) > 0 else 0
        avg_loss = abs(np.mean(daily_returns[daily_returns < 0])) if len(daily_returns[daily_returns < 0]) > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio
        }
    
    def calculate_trading_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """计算交易相关指标"""
        if len(trades_df) == 0:
            return {}
        
        # 交易次数
        n_trades = len(trades_df)
        n_buy = len(trades_df[trades_df['action'] == 'buy'])
        n_sell = len(trades_df[trades_df['action'] == 'sell'])
        
        # 总交易成本
        total_commission = trades_df['commission'].sum()
        
        # 平均交易规模
        avg_trade_value = trades_df['value'].mean()
        
        # 交易频率（每月交易次数）
        if 'date' in trades_df.columns:
            date_range_days = (trades_df['date'].max() - trades_df['date'].min()).days
            trades_per_month = n_trades / (date_range_days / 30) if date_range_days > 0 else 0
        else:
            trades_per_month = 0
        
        return {
            'n_trades': n_trades,
            'n_buy': n_buy,
            'n_sell': n_sell,
            'total_commission': total_commission,
            'avg_trade_value': avg_trade_value,
            'trades_per_month': trades_per_month
        }
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """计算最大回撤持续期（天数）"""
        is_drawdown = drawdown < 0
        
        # 找到所有回撤期
        drawdown_periods = []
        start = None
        
        for i, in_dd in enumerate(is_drawdown):
            if in_dd and start is None:
                start = i
            elif not in_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        # 如果最后还在回撤中
        if start is not None:
            drawdown_periods.append(len(drawdown) - start)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def print_report(self, analysis: Dict):
        """打印格式化的分析报告"""
        print("\n" + "=" * 60)
        print("BACKTEST PERFORMANCE REPORT".center(60))
        print("=" * 60)
        
        # 收益指标
        print("\n📈 RETURNS METRICS")
        print("-" * 60)
        self._print_metric("Total Return", analysis.get('total_return', 0), is_percentage=True)
        self._print_metric("Annual Return", analysis.get('annual_return', 0), is_percentage=True)
        self._print_metric("Avg Daily Return", analysis.get('avg_daily_return', 0), is_percentage=True)
        self._print_metric("Best Day", analysis.get('best_day_return', 0), is_percentage=True)
        self._print_metric("Worst Day", analysis.get('worst_day_return', 0), is_percentage=True)
        self._print_metric("Final Value", analysis.get('final_value', 0), is_currency=True)
        
        # 风险指标
        print("\n⚠️  RISK METRICS")
        print("-" * 60)
        self._print_metric("Volatility (Annual)", analysis.get('volatility', 0), is_percentage=True)
        self._print_metric("Downside Volatility", analysis.get('downside_volatility', 0), is_percentage=True)
        self._print_metric("Max Drawdown", analysis.get('max_drawdown', 0), is_percentage=True)
        self._print_metric("Max DD Duration", analysis.get('max_drawdown_duration', 0), suffix=" days")
        self._print_metric("VaR (95%)", analysis.get('var_95', 0), is_percentage=True)
        self._print_metric("CVaR (95%)", analysis.get('cvar_95', 0), is_percentage=True)
        
        # 风险调整收益
        print("\n🎯 RISK-ADJUSTED RETURNS")
        print("-" * 60)
        self._print_metric("Sharpe Ratio", analysis.get('sharpe_ratio', 0))
        self._print_metric("Sortino Ratio", analysis.get('sortino_ratio', 0))
        self._print_metric("Calmar Ratio", analysis.get('calmar_ratio', 0))
        self._print_metric("Information Ratio", analysis.get('information_ratio', 0))
        self._print_metric("Win Rate", analysis.get('win_rate', 0), is_percentage=True)
        self._print_metric("Profit/Loss Ratio", analysis.get('profit_loss_ratio', 0))
        
        # 交易统计
        if 'n_trades' in analysis:
            print("\n💼 TRADING METRICS")
            print("-" * 60)
            self._print_metric("Total Trades", analysis.get('n_trades', 0))
            self._print_metric("Buy Orders", analysis.get('n_buy', 0))
            self._print_metric("Sell Orders", analysis.get('n_sell', 0))
            self._print_metric("Total Commission", analysis.get('total_commission', 0), is_currency=True)
            self._print_metric("Avg Trade Value", analysis.get('avg_trade_value', 0), is_currency=True)
            self._print_metric("Trades per Month", analysis.get('trades_per_month', 0))
        
        print("\n" + "=" * 60)
    
    def _print_metric(self, name: str, value: float, is_percentage: bool = False, 
                     is_currency: bool = False, suffix: str = ""):
        """打印单个指标"""
        if is_percentage:
            print(f"{name:.<40} {value*100:>10.2f}%")
        elif is_currency:
            print(f"{name:.<40} ${value:>10,.2f}")
        else:
            print(f"{name:.<40} {value:>10.4f}{suffix}")


if __name__ == "__main__":
    # 测试示例
    print("=== Testing Performance Analyzer ===\n")
    
    # 创建模拟投资组合数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    n = len(dates)
    
    # 模拟收益率
    daily_returns = np.random.randn(n) * 0.01  # 1% 日波动
    cumulative_returns = np.cumprod(1 + daily_returns)
    
    portfolio_df = pd.DataFrame({
        'date': dates,
        'cash': 10000,
        'positions_value': 100000 * cumulative_returns,
        'total_value': 100000 * cumulative_returns,
        'daily_return': daily_returns,
        'n_positions': 3
    })
    
    # 创建模拟交易数据
    trades_df = pd.DataFrame({
        'date': dates[::30],  # 每月交易
        'symbol': ['AAPL', 'MSFT', 'GOOGL'] * 4,
        'action': (['buy'] * 3 + ['sell'] * 3) * 2,
        'shares': [100] * 12,
        'price': [150, 300, 2800] * 4,
        'value': [15000, 30000, 280000] * 4,
        'commission': [15, 30, 280] * 4
    })
    
    # 分析
    analyzer = PerformanceAnalyzer()
    analysis = analyzer.analyze(portfolio_df, trades_df)
    
    # 打印报告
    analyzer.print_report(analysis)
    
    print("\n✅ Performance analyzer test passed!")

