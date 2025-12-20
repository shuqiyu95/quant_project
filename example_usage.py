"""
使用示例 - 展示如何使用 data_engine 模块
"""
import sys
from pathlib import Path

# 添加 src 到 Python 路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data_engine import DataManager
from datetime import datetime


def example_1_basic_usage():
    """示例 1: 基本使用 - 获取单只股票数据"""
    print("\n" + "="*60)
    print("示例 1: 基本使用 - 获取单只股票数据")
    print("="*60)
    
    dm = DataManager(data_dir="data")
    
    # 获取美股数据（自动识别）
    print("\n获取 AAPL 数据...")
    df_aapl = dm.fetch_data("AAPL")
    
    print(f"✓ 数据形状: {df_aapl.shape}")
    print(f"✓ 日期范围: {df_aapl.index.min().date()} 到 {df_aapl.index.max().date()}")
    print(f"\n最新收盘价: ${df_aapl['close'].iloc[-1]:.2f}")
    print(f"最近5日平均成交量: {df_aapl['volume'].tail(5).mean():,.0f}")
    

def example_2_custom_date_range():
    """示例 2: 自定义日期范围"""
    print("\n" + "="*60)
    print("示例 2: 自定义日期范围")
    print("="*60)
    
    dm = DataManager(data_dir="data")
    
    # 获取特定日期范围的数据
    print("\n获取 2024年 1-6月 的 NVDA 数据...")
    df = dm.fetch_data(
        "NVDA",
        start_date="2024-01-01",
        end_date="2024-06-30"
    )
    
    print(f"✓ 数据行数: {len(df)}")
    print(f"✓ 期间涨跌幅: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.2f}%")


def example_3_mag7_analysis():
    """示例 3: Mag7 组合分析"""
    print("\n" + "="*60)
    print("示例 3: Mag7 组合分析")
    print("="*60)
    
    dm = DataManager(data_dir="data")
    
    mag7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    print("\n批量获取 Mag7 数据...")
    data = dm.fetch_multiple(mag7, use_cache=True)
    
    print("\nMag7 最近一年表现:")
    print("-" * 60)
    
    for symbol, df in data.items():
        if df is not None and len(df) > 0:
            # 计算收益率
            returns = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
            # 计算波动率
            volatility = df['close'].pct_change().std() * 100
            
            print(f"{symbol:6s} | 收益: {returns:+7.2f}% | 波动率: {volatility:.2f}% | 最新价: ${df['close'].iloc[-1]:,.2f}")


def example_4_cn_market():
    """示例 4: A股市场数据"""
    print("\n" + "="*60)
    print("示例 4: A股市场数据")
    print("="*60)
    
    dm = DataManager(data_dir="data")
    
    # A股热门股票
    cn_stocks = {
        '600519': '贵州茅台',
        '000858': '五粮液',
        '600036': '招商银行'
    }
    
    print("\n获取A股数据...")
    for code, name in cn_stocks.items():
        try:
            df = dm.fetch_data(code)
            if df is not None and len(df) > 0:
                returns = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                print(f"✓ {name}({code}): {len(df)}条数据, 年度收益: {returns:+.2f}%")
        except Exception as e:
            print(f"✗ {name}({code}): 获取失败 - {e}")


def example_5_mixed_markets():
    """示例 5: 跨市场对比"""
    print("\n" + "="*60)
    print("示例 5: 跨市场对比 - 美股 vs A股")
    print("="*60)
    
    dm = DataManager(data_dir="data")
    
    symbols = {
        'AAPL': '苹果(美股)',
        'MSFT': '微软(美股)', 
        '600519': '茅台(A股)',
        '000858': '五粮液(A股)'
    }
    
    print("\n跨市场数据获取:")
    for symbol, name in symbols.items():
        market = dm.identify_market(symbol)
        print(f"  {name:15s} -> 市场: {market:3s}, 代码: {symbol}")
    
    print("\n自动识别市场并获取数据...")
    for symbol, name in symbols.items():
        try:
            df = dm.fetch_data(symbol, use_cache=True)
            if df is not None:
                print(f"✓ {name}: {len(df)}条数据")
        except Exception as e:
            print(f"✗ {name}: {e}")


if __name__ == "__main__":
    print("\n" + "🚀 Data Engine 使用示例".center(60, "="))
    
    # 运行所有示例
    example_1_basic_usage()
    example_2_custom_date_range()
    example_3_mag7_analysis()
    example_4_cn_market()
    example_5_mixed_markets()
    
    print("\n" + "="*60)
    print("✅ 所有示例运行完成！")
    print("="*60)
    print("\n💡 提示:")
    print("  - 数据已缓存在 data/ 目录")
    print("  - 再次运行将使用缓存，速度更快")
    print("  - 可以查看 README.md 了解更多用法")
    print()

