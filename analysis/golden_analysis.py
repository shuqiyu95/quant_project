"""
金风科技(002202)尾盘买入策略分析
策略：T日尾盘买入，T+1日30分钟内高点卖出
分析过去半年的收益情况
"""
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_engine.cn_fetcher import CNFetcher
from src.data_engine.data_manager import DataManager


def get_close_auction_price(symbol: str, date: datetime, manager: DataManager, df_daily: pd.DataFrame) -> float:
    """
    获取尾盘价格（14:55-15:00的均价或收盘价）
    
    Args:
        symbol: 股票代码
        date: 日期
        manager: 数据管理器（支持缓存）
        df_daily: 日线数据（作为fallback）
        
    Returns:
        尾盘价格
    """
    try:
        # 获取当日分钟数据（使用缓存）
        next_day = date + timedelta(days=1)
        df_min = manager.fetch_intraday_data(
            symbol=symbol,
            start_date=date,
            end_date=next_day,
            period="1",
            use_cache=True
        )
        
        if not df_min.empty:
            # 筛选尾盘时间段 14:55-15:00
            df_close = df_min.between_time('14:55', '15:00')
            
            if not df_close.empty:
                # 使用尾盘最后一个价格（收盘价）
                return df_close.iloc[-1]['close']
        
        # 如果没有分钟数据，使用日线收盘价作为fallback
        daily_row = df_daily[df_daily.index == date]
        if not daily_row.empty:
            return daily_row.iloc[0]['close']
        
        return None
        
    except Exception as e:
        print(f"    ⚠ 获取尾盘价格异常 {date.date()}: {str(e)}")
        # 使用日线收盘价作为fallback
        daily_row = df_daily[df_daily.index == date]
        if not daily_row.empty:
            return daily_row.iloc[0]['close']
        return None


def get_next_day_high_30min(symbol: str, next_date: datetime, fetcher: CNFetcher, df_daily: pd.DataFrame) -> tuple:
    """
    获取次日开盘后30分钟内的最高价
    
    Args:
        symbol: 股票代码
        next_date: 次日日期
        fetcher: 数据获取器
        df_daily: 日线数据（作为fallback）
        
    Returns:
        (最高价, 最高价时间)
    """
    try:
        # 获取次日数据
        end_day = next_date + timedelta(days=1)
        
        df_min = fetcher.fetch_intraday_data(
            symbol=symbol,
            start_date=next_date,
            end_date=end_day,
            period="1"
        )
        
        if not df_min.empty:
            # 筛选开盘后30分钟 9:30-10:00
            df_morning = df_min.between_time('09:30', '10:00')
            
            if not df_morning.empty:
                # 找到最高价
                max_idx = df_morning['high'].idxmax()
                max_price = df_morning.loc[max_idx, 'high']
                return max_price, max_idx
        
        # 如果没有分钟数据，使用日线开盘价作为fallback
        daily_row = df_daily[df_daily.index == next_date]
        if not daily_row.empty:
            # 使用开盘价和最高价的平均值作为估计
            open_price = daily_row.iloc[0]['open']
            high_price = daily_row.iloc[0]['high']
            estimated_price = (open_price + high_price) / 2
            return estimated_price, next_date.replace(hour=9, minute=45)
        
        return None, None
        
    except Exception as e:
        print(f"    ⚠ 获取次日30分钟高点异常 {next_date.date()}: {str(e)}")
        # 使用日线数据作为fallback
        daily_row = df_daily[df_daily.index == next_date]
        if not daily_row.empty:
            open_price = daily_row.iloc[0]['open']
            high_price = daily_row.iloc[0]['high']
            estimated_price = (open_price + high_price) / 2
            return estimated_price, next_date.replace(hour=9, minute=45)
        return None, None


def calculate_strategy_returns(symbol: str, start_date: datetime, end_date: datetime):
    """
    计算策略收益
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        包含每日收益的DataFrame
    """
    fetcher = CNFetcher()
    
    # 获取日线数据，确定交易日
    print(f"正在获取 {symbol} 的日线数据...")
    df_daily = fetcher.fetch_daily_data(symbol, start_date, end_date)
    
    if df_daily.empty:
        print("没有找到日线数据")
        return pd.DataFrame()
    
    print(f"找到 {len(df_daily)} 个交易日")
    
    # 存储结果
    results = []
    
    # 遍历每个交易日（除了最后一天，因为需要T+1数据）
    total_days = len(df_daily) - 1
    success_count = 0
    failed_count = 0
    
    for i in range(total_days):
        trade_date = df_daily.index[i]
        next_trade_date = df_daily.index[i + 1]
        
        print(f"\n[{i+1}/{total_days}] 处理交易日: {trade_date.date()} -> {next_trade_date.date()}")
        
        # 1. 获取T日尾盘买入价
        buy_price = get_close_auction_price(symbol, trade_date, fetcher, df_daily)
        
        if buy_price is None:
            print(f"  ⚠ 无法获取买入价格，跳过此交易日")
            failed_count += 1
            continue
        
        print(f"  ✓ 买入价格: {buy_price:.2f}")
        
        # 2. 获取T+1日30分钟内最高价
        sell_price, sell_time = get_next_day_high_30min(symbol, next_trade_date, fetcher, df_daily)
        
        if sell_price is None:
            print(f"  ⚠ 无法获取卖出价格，跳过此交易日")
            failed_count += 1
            continue
        
        print(f"  ✓ 卖出价格: {sell_price:.2f} (时间: {sell_time.strftime('%H:%M') if sell_time else 'N/A'})")
        
        # 3. 计算收益
        ret = (sell_price - buy_price) / buy_price * 100
        
        print(f"  ✓ 收益率: {ret:.2f}%")
        
        success_count += 1
        
        results.append({
            'trade_date': trade_date,
            'next_date': next_trade_date,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'sell_time': sell_time,
            'return_pct': ret,
            'cumulative_return': 0  # 稍后计算
        })
        
        # 添加小延迟避免API限流
        if i < total_days - 1:
            time.sleep(0.5)
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    print(f"\n" + "="*60)
    print(f"数据获取完成:")
    print(f"  成功: {success_count} 个交易日")
    print(f"  失败: {failed_count} 个交易日")
    print(f"  成功率: {success_count/(success_count+failed_count)*100:.1f}%" if (success_count+failed_count) > 0 else "  成功率: N/A")
    print("="*60)
    
    if df_results.empty:
        return df_results
    
    # 计算累计收益（复利）
    df_results['cumulative_return'] = (1 + df_results['return_pct'] / 100).cumprod() - 1
    df_results['cumulative_return'] *= 100  # 转换为百分比
    
    return df_results


def plot_returns(df_results: pd.DataFrame, symbol: str):
    """
    使用plotly绘制收益曲线
    
    Args:
        df_results: 收益结果DataFrame
        symbol: 股票代码
    """
    if df_results.empty:
        print("没有数据可以绘制")
        return
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'{symbol} 尾盘买入策略 - 累计收益曲线',
            '每日收益率分布'
        ),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )
    
    # 1. 累计收益曲线
    fig.add_trace(
        go.Scatter(
            x=df_results['trade_date'],
            y=df_results['cumulative_return'],
            mode='lines+markers',
            name='累计收益',
            line=dict(color='#2E86DE', width=2),
            marker=dict(size=4),
            hovertemplate='<b>日期</b>: %{x|%Y-%m-%d}<br>' +
                         '<b>累计收益</b>: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # 2. 每日收益率柱状图
    colors = ['#EE5A6F' if x < 0 else '#26DE81' for x in df_results['return_pct']]
    
    fig.add_trace(
        go.Bar(
            x=df_results['trade_date'],
            y=df_results['return_pct'],
            name='每日收益',
            marker_color=colors,
            hovertemplate='<b>日期</b>: %{x|%Y-%m-%d}<br>' +
                         '<b>收益率</b>: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_xaxes(title_text="交易日期", row=2, col=1)
    fig.update_yaxes(title_text="累计收益率 (%)", row=1, col=1)
    fig.update_yaxes(title_text="日收益率 (%)", row=2, col=1)
    
    fig.update_layout(
        title={
            'text': f'<b>{symbol} 金风科技 - T日尾盘买入 T+1日30分钟高点卖出策略</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        showlegend=True,
        height=800,
        hovermode='x unified',
        template='plotly_white',
        font=dict(family="Arial, sans-serif")
    )
    
    # 保存为HTML
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'../output/{symbol}_strategy_analysis.html'
    )
    fig.write_html(output_path)
    print(f"\n✓ 图表已保存至: {output_path}")
    
    # 显示图表
    fig.show()


def print_statistics(df_results: pd.DataFrame):
    """打印统计信息"""
    if df_results.empty:
        return
    
    print("\n" + "="*60)
    print("策略统计信息")
    print("="*60)
    
    total_trades = len(df_results)
    win_trades = len(df_results[df_results['return_pct'] > 0])
    lose_trades = len(df_results[df_results['return_pct'] < 0])
    win_rate = win_trades / total_trades * 100
    
    avg_return = df_results['return_pct'].mean()
    avg_win = df_results[df_results['return_pct'] > 0]['return_pct'].mean() if win_trades > 0 else 0
    avg_loss = df_results[df_results['return_pct'] < 0]['return_pct'].mean() if lose_trades > 0 else 0
    
    max_return = df_results['return_pct'].max()
    min_return = df_results['return_pct'].min()
    
    final_cumulative = df_results['cumulative_return'].iloc[-1]
    
    print(f"\n交易次数: {total_trades}")
    print(f"盈利次数: {win_trades} ({win_rate:.1f}%)")
    print(f"亏损次数: {lose_trades} ({100-win_rate:.1f}%)")
    print(f"\n平均收益: {avg_return:.2f}%")
    print(f"平均盈利: {avg_win:.2f}%")
    print(f"平均亏损: {avg_loss:.2f}%")
    print(f"盈亏比: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A")
    print(f"\n最大单日收益: {max_return:.2f}%")
    print(f"最大单日亏损: {min_return:.2f}%")
    print(f"\n累计收益: {final_cumulative:.2f}%")
    print("="*60)


def main():
    """主函数"""
    print("\n" + "🚀 "*30)
    print("金风科技(002202) - 尾盘买入策略回测分析")
    print("策略: T日尾盘(14:55-15:00)买入，T+1日开盘后30分钟内高点卖出")
    print("🚀 "*30)
    
    symbol = "002202"  # 金风科技
    
    # 过去半年
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    print(f"\n分析周期: {start_date.date()} 至 {end_date.date()}")
    
    # 计算策略收益
    df_results = calculate_strategy_returns(symbol, start_date, end_date)
    
    if df_results.empty:
        print("\n❌ 没有获取到有效数据")
        return
    
    # 保存结果
    output_csv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'../output/{symbol}_strategy_results.csv'
    )
    df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存至: {output_csv}")
    
    # 打印统计信息
    print_statistics(df_results)
    
    # 绘制图表
    plot_returns(df_results, symbol)
    
    print("\n✓ 分析完成!")


if __name__ == "__main__":
    main()

