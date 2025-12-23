"""
A股数据获取模块演示脚本
展示各项功能的使用方法
"""
from datetime import datetime, timedelta
from src.data_engine.cn_fetcher import CNFetcher
from src.data_engine.data_manager import DataManager


def demo_basic_daily_data():
    """演示1: 获取基础日线数据"""
    print("\n" + "="*60)
    print("演示 1: 获取A股日线数据")
    print("="*60)
    
    fetcher = CNFetcher()
    symbol = "600519"  # 贵州茅台
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    print(f"\n正在获取 {symbol} 的数据...")
    df = fetcher.fetch_daily_data(symbol, start_date, end_date)
    
    print(f"\n✓ 数据获取成功!")
    print(f"  股票代码: {symbol}")
    print(f"  数据条数: {len(df)}")
    print(f"  日期范围: {df.index.min().date()} 到 {df.index.max().date()}")
    print(f"\n最近5天数据:")
    print(df[['open', 'high', 'low', 'close', 'volume']].tail())
    
    return df


def demo_data_manager():
    """演示2: 使用DataManager进行数据管理"""
    print("\n" + "="*60)
    print("演示 2: 数据管理器 - 缓存与增量更新")
    print("="*60)
    
    manager = DataManager(data_dir="data")
    symbol = "600036"  # 招商银行
    
    # 首次获取
    print(f"\n首次获取 {symbol} 数据（30天）...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    df1 = manager.fetch_data(symbol, start_date, end_date, use_cache=False)
    print(f"✓ 获取了 {len(df1)} 条记录")
    
    # 使用缓存
    print(f"\n再次获取（使用缓存）...")
    df2 = manager.fetch_data(symbol, start_date, end_date, use_cache=True)
    print(f"✓ 从缓存读取 {len(df2)} 条记录")
    
    # 增量更新
    print(f"\n执行增量更新...")
    df3 = manager.fetch_data_incremental(symbol)
    print(f"✓ 更新后共 {len(df3)} 条记录")
    
    return df3


def demo_multiple_symbols():
    """演示3: 批量获取多只股票"""
    print("\n" + "="*60)
    print("演示 3: 批量获取多只股票数据")
    print("="*60)
    
    manager = DataManager(data_dir="data")
    symbols = ["600519", "600036", "000858"]  # 茅台、招行、五粮液
    
    print(f"\n正在获取 {len(symbols)} 只股票的数据...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    results = manager.fetch_multiple(symbols, start_date, end_date)
    
    print(f"\n✓ 批量获取完成:")
    for symbol, df in results.items():
        if df is not None and not df.empty:
            latest_close = df['close'].iloc[-1]
            print(f"  {symbol}: {len(df)} 条记录, 最新价 {latest_close:.2f}")
        else:
            print(f"  {symbol}: 获取失败")
    
    return results


def demo_industry_data():
    """演示4: 获取行业数据"""
    print("\n" + "="*60)
    print("演示 4: 获取股票行业信息")
    print("="*60)
    
    fetcher = CNFetcher()
    symbol = "600519"
    
    print(f"\n正在获取 {symbol} 的行业信息...")
    
    try:
        industry_info = fetcher.fetch_industry_data(symbol)
        
        if industry_info:
            print(f"\n✓ 行业信息:")
            for key, value in industry_info.items():
                if value:
                    print(f"  {key}: {value}")
        else:
            print("\n⚠ 暂无行业数据")
            
    except Exception as e:
        print(f"\n⚠ 行业数据获取遇到问题: {str(e)}")
        print("  (可能需要特定的API权限)")


def demo_turnover_analysis():
    """演示5: 换手率分析"""
    print("\n" + "="*60)
    print("演示 5: 换手率分位数分析")
    print("="*60)
    
    fetcher = CNFetcher()
    symbol = "600519"
    
    print(f"\n正在分析 {symbol} 的换手率...")
    
    try:
        quantile = fetcher.fetch_turnover_quantile(
            symbol=symbol,
            current_date=datetime.now(),
            lookback_days=100
        )
        
        if quantile is not None:
            print(f"\n✓ 换手率分析结果:")
            print(f"  当前换手率分位数: {quantile:.2%}")
            print(f"  （在最近100个交易日中的相对位置）")
            
            if quantile < 0.2:
                print("  📊 解读: 地量区域，成交清淡")
            elif quantile > 0.8:
                print("  📊 解读: 放量区域，交易活跃")
            else:
                print("  📊 解读: 正常成交量水平")
        else:
            print("\n⚠ 无法计算换手率分位数")
            
    except Exception as e:
        print(f"\n⚠ 换手率分析失败: {str(e)}")


def demo_realtime_quotes():
    """演示6: 实时行情"""
    print("\n" + "="*60)
    print("演示 6: 获取实时行情")
    print("="*60)
    
    fetcher = CNFetcher()
    symbols = ["600519", "600036", "000858"]
    
    print(f"\n正在获取 {len(symbols)} 只股票的实时行情...")
    
    try:
        df = fetcher.get_realtime_quotes(symbols)
        
        if not df.empty:
            print(f"\n✓ 实时行情:")
            print(df[['symbol', 'name', 'price', 'pct_change', 'volume']].to_string(index=False))
        else:
            print("\n⚠ 暂无实时行情数据")
            
    except Exception as e:
        print(f"\n⚠ 实时行情获取失败: {str(e)}")


def main():
    """运行所有演示"""
    print("\n" + "🚀 "*30)
    print("A股数据获取模块 - 功能演示")
    print("🚀 "*30)
    
    try:
        # 演示1: 基础日线数据
        demo_basic_daily_data()
        
        # 演示2: 数据管理器
        demo_data_manager()
        
        # 演示3: 批量获取
        demo_multiple_symbols()
        
        # 演示4: 行业数据
        demo_industry_data()
        
        # 演示5: 换手率分析
        demo_turnover_analysis()
        
        # 演示6: 实时行情
        demo_realtime_quotes()
        
        print("\n" + "="*60)
        print("✓ 所有演示完成!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 演示过程出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

