"""
热股榜数据分析示例
展示如何使用热股榜数据和涨幅数据进行分析
"""

import sys
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.hot_stock import HotStockTracker


def analyze_today_hot_stocks():
    """分析今日热股榜"""
    print("="*60)
    print("今日热股榜分析")
    print("="*60)
    
    tracker = HotStockTracker()
    
    # 获取今日数据
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    df = tracker.load_daily_data(today)
    
    if df.empty:
        print(f"没有找到 {today} 的数据")
        print("请先运行: python analysis/hot_stock.py --daily")
        return
    
    print(f"\n📊 今日热股榜 TOP 10")
    print("-" * 60)
    display_df = df[['rank', 'symbol', 'name', 'price', 'change_pct']].copy()
    display_df.columns = ['排名', '代码', '名称', '最新价', '今日涨跌%']
    print(display_df.to_string(index=False))
    
    # 涨幅分析
    print(f"\n\n📈 多周期涨幅分析")
    print("-" * 60)
    
    # 短期强势股
    if 'return_3d' in df.columns:
        strong_3d = df[df['return_3d'] > 15].copy()
        if not strong_3d.empty:
            print("\n🔥 短期强势股 (3日涨幅 > 15%):")
            for _, row in strong_3d.iterrows():
                print(f"  {row['rank']:2d}. {row['symbol']} {row['name']:8s} | "
                      f"1日: {row.get('return_1d', 0):6.2f}% | "
                      f"3日: {row.get('return_3d', 0):6.2f}% | "
                      f"5日: {row.get('return_5d', 0):6.2f}%")
        else:
            print("\n无股票满足短期强势条件 (3日涨幅 > 15%)")
    
    # 中期强势股
    if 'return_10d' in df.columns:
        strong_10d = df[df['return_10d'] > 30].copy()
        if not strong_10d.empty:
            print("\n🚀 中期强势股 (10日涨幅 > 30%):")
            for _, row in strong_10d.iterrows():
                print(f"  {row['rank']:2d}. {row['symbol']} {row['name']:8s} | "
                      f"5日: {row.get('return_5d', 0):6.2f}% | "
                      f"10日: {row.get('return_10d', 0):6.2f}%")
        else:
            print("\n无股票满足中期强势条件 (10日涨幅 > 30%)")
    
    # 动能分析
    if 'return_1d' in df.columns and 'return_3d' in df.columns:
        df['momentum'] = df['return_3d'] - df['return_1d']
        momentum_stocks = df[df['momentum'] > 5].copy()
        if not momentum_stocks.empty:
            print("\n⚡ 有持续上涨动能的股票 (3日累计涨幅 - 1日涨幅 > 5%):")
            for _, row in momentum_stocks.iterrows():
                print(f"  {row['rank']:2d}. {row['symbol']} {row['name']:8s} | "
                      f"1日: {row.get('return_1d', 0):6.2f}% | "
                      f"3日: {row.get('return_3d', 0):6.2f}% | "
                      f"动能: {row['momentum']:6.2f}%")
    
    # 风险提示
    print(f"\n\n⚠️  风险提示")
    print("-" * 60)
    
    warnings_found = False
    
    # 过热警告
    if 'return_10d' in df.columns:
        overheated = df[df['return_10d'] > 50].copy()
        if not overheated.empty:
            warnings_found = True
            print("\n⚠️  短期涨幅过大 (10日 > 50%)，注意回调风险:")
            for _, row in overheated.iterrows():
                print(f"  {row['symbol']} {row['name']:8s} | 10日涨幅: {row['return_10d']:6.2f}%")
    
    # 动能减弱
    if 'return_1d' in df.columns and 'return_3d' in df.columns:
        df['avg_3d'] = df['return_3d'] / 3
        losing_momentum = df[df['return_1d'] < df['avg_3d'] / 2].copy()
        if not losing_momentum.empty:
            warnings_found = True
            print("\n⚠️  上涨动能减弱:")
            for _, row in losing_momentum.iterrows():
                print(f"  {row['symbol']} {row['name']:8s} | "
                      f"1日: {row.get('return_1d', 0):6.2f}% | "
                      f"3日均: {row['avg_3d']:6.2f}%")
    
    # 热股下跌
    if 'return_1d' in df.columns:
        negative_hot = df[(df['rank'] <= 10) & (df['return_1d'] < 0)].copy()
        if not negative_hot.empty:
            warnings_found = True
            print("\n⚠️  热股出现下跌，可能有利空消息:")
            for _, row in negative_hot.iterrows():
                print(f"  {row['rank']:2d}. {row['symbol']} {row['name']:8s} | "
                      f"今日跌幅: {row['return_1d']:6.2f}%")
    
    if not warnings_found:
        print("\n✅ 暂无明显风险信号")
    
    print("\n")


def analyze_weekly_trends():
    """分析本周热度趋势"""
    print("="*60)
    print("本周热度趋势分析")
    print("="*60)
    
    tracker = HotStockTracker()
    
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
    
    # 生成热度因子
    heat_df = tracker.generate_heat_factor(start_date, end_date, method='weighted')
    
    if heat_df.empty:
        print(f"\n没有找到 {start_date} 到 {end_date} 的数据")
        return
    
    print(f"\n📊 本周最热股票 TOP 10 ({start_date} ~ {end_date})")
    print("-" * 60)
    
    top10 = heat_df.head(10).copy()
    display_df = top10[['heat_rank', 'symbol', 'name', 'heat_score', 
                        'appearance_count', 'avg_rank', 'min_rank']].copy()
    display_df.columns = ['热度排名', '代码', '名称', '热度得分', 
                          '上榜次数', '平均排名', '最高排名']
    print(display_df.to_string(index=False))
    
    # 持续热门
    continuous_hot = heat_df[heat_df['appearance_count'] >= 5].copy()
    if not continuous_hot.empty:
        print(f"\n\n🔥 持续热门股票 (上榜 ≥ 5天):")
        print("-" * 60)
        for _, row in continuous_hot.iterrows():
            print(f"  {row['symbol']} {row['name']:8s} | "
                  f"上榜{int(row['appearance_count'])}天 | "
                  f"平均排名: {row['avg_rank']:4.1f} | "
                  f"热度得分: {row['heat_score']:5.1f}")
    
    print("\n")


def main():
    """主函数"""
    try:
        # 分析今日热股
        analyze_today_hot_stocks()
        
        # 分析本周趋势
        analyze_weekly_trends()
        
    except Exception as e:
        print(f"❌ 分析过程出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

