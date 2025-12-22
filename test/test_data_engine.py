"""
测试脚本 - 验证 data_engine 模块功能
"""
import sys
from pathlib import Path

# 添加 src 到 Python 路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.data_engine import DataManager


def test_market_identification():
    """测试市场识别功能"""
    print("=" * 60)
    print("测试 1: 市场识别")
    print("=" * 60)
    
    dm = DataManager(data_dir="data")
    
    test_cases = [
        ("AAPL", "US"),
        ("MSFT", "US"),
        ("GOOGL", "US"),
        ("600519", "CN"),
        ("000001", "CN"),
        ("123", "UNKNOWN"),
    ]
    
    for symbol, expected in test_cases:
        market = dm.identify_market(symbol)
        status = "✓" if market == expected else "✗"
        print(f"{status} {symbol:10s} -> {market:10s} (expected: {expected})")
    
    print()


def test_us_data_fetch():
    """测试美股数据获取"""
    print("=" * 60)
    print("测试 2: 美股数据获取 (AAPL)")
    print("=" * 60)
    
    dm = DataManager(data_dir="data")
    
    try:
        df = dm.fetch_data("AAPL", use_cache=False)
        
        print(f"✓ 成功获取数据")
        print(f"  - 数据行数: {len(df)}")
        print(f"  - 日期范围: {df.index.min().date()} 到 {df.index.max().date()}")
        print(f"  - 列: {list(df.columns)}")
        print(f"  - 数据类型: {df.dtypes.to_dict()}")
        print(f"\n最近5天数据:")
        print(df[['open', 'high', 'low', 'close', 'volume']].tail())
        
    except Exception as e:
        print(f"✗ 获取失败: {e}")
    
    print()


def test_cn_data_fetch():
    """测试A股数据获取"""
    print("=" * 60)
    print("测试 3: A股数据获取 (600519 - 贵州茅台)")
    print("=" * 60)
    
    dm = DataManager(data_dir="data")
    
    try:
        df = dm.fetch_data("600519", use_cache=False)
        
        print(f"✓ 成功获取数据")
        print(f"  - 数据行数: {len(df)}")
        print(f"  - 日期范围: {df.index.min().date()} 到 {df.index.max().date()}")
        print(f"  - 列: {list(df.columns)}")
        print(f"  - 数据类型: {df.dtypes.to_dict()}")
        print(f"\n最近5天数据:")
        print(df[['open', 'high', 'low', 'close', 'volume']].tail())
        
    except Exception as e:
        print(f"✗ 获取失败: {e}")
    
    print()


def test_batch_fetch():
    """测试批量获取"""
    print("=" * 60)
    print("测试 4: 批量获取 Mag7 数据")
    print("=" * 60)
    
    dm = DataManager(data_dir="data")
    
    mag7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    try:
        results = dm.fetch_multiple(mag7, use_cache=True)
        
        print(f"批量获取结果:")
        for symbol, df in results.items():
            if df is not None:
                print(f"  ✓ {symbol:6s}: {len(df):4d} 条数据")
            else:
                print(f"  ✗ {symbol:6s}: 获取失败")
        
    except Exception as e:
        print(f"✗ 批量获取失败: {e}")
    
    print()


def test_cache():
    """测试缓存功能"""
    print("=" * 60)
    print("测试 5: 缓存功能")
    print("=" * 60)
    
    dm = DataManager(data_dir="data")
    
    print("第一次获取（无缓存）...")
    import time
    start = time.time()
    df1 = dm.fetch_data("NVDA", use_cache=False)
    time1 = time.time() - start
    
    print(f"第二次获取（使用缓存）...")
    start = time.time()
    df2 = dm.fetch_data("NVDA", use_cache=True)
    time2 = time.time() - start
    
    print(f"\n性能对比:")
    print(f"  无缓存: {time1:.2f} 秒")
    print(f"  有缓存: {time2:.2f} 秒")
    print(f"  加速: {time1/time2:.1f}x")
    
    print()


if __name__ == "__main__":
    print("\n🚀 Data Engine 测试开始\n")
    
    # 运行所有测试
    test_market_identification()
    test_us_data_fetch()
    test_cn_data_fetch()
    test_batch_fetch()
    test_cache()
    
    print("=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)

