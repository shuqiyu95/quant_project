"""
因子计算示例
演示如何使用 Alpha158 和 Alpha360 因子库
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from factors import (
    # 基础算子
    Ref, MA, Std, Slope, RSI, MACD,
    # 因子库
    Alpha158, Alpha360, calculate_alpha158, calculate_alpha360
)
from data_engine.data_manager import DataManager


def example_basic_operators():
    """示例 1: 使用基础算子"""
    print("\n" + "="*60)
    print("示例 1: 使用基础算子")
    print("="*60)
    
    # 加载数据
    dm = DataManager()
    df = dm.get_stock_data('AAPL', market='us')
    
    if df is not None and len(df) > 0:
        # 计算一些基础指标
        close = df['close']
        
        # 5日均线
        ma5 = MA(close, 5)
        
        # 20日波动率
        std20 = Std(close, 20)
        
        # 10日斜率
        slope10 = Slope(close, 10)
        
        # RSI 指标
        rsi14 = RSI(close, 14)
        
        # 组合结果
        result = pd.DataFrame({
            'close': close,
            'MA5': ma5,
            'STD20': std20,
            'SLOPE10': slope10,
            'RSI14': rsi14
        })
        
        print(f"\n{result.tail(10)}")
        print(f"\n数据统计:")
        print(result.describe())
    else:
        print("无法加载数据，请先运行数据获取")


def example_alpha158():
    """示例 2: 使用 Alpha158 因子库"""
    print("\n" + "="*60)
    print("示例 2: 使用 Alpha158 因子库")
    print("="*60)
    
    # 加载数据
    dm = DataManager()
    df = dm.get_stock_data('AAPL', market='us')
    
    if df is not None and len(df) > 0:
        # 方法 1: 使用便捷函数
        factors = calculate_alpha158(df)
        
        print(f"\n生成的因子数量: {len(factors.columns)}")
        print(f"数据行数: {len(factors)}")
        print(f"\n前 10 个因子的前 5 行:")
        print(factors.iloc[:5, :10])
        
        # 方法 2: 使用类
        alpha158 = Alpha158()
        factors2 = alpha158.calculate(df)
        
        # 获取所有因子名称
        factor_names = alpha158.get_factor_names()
        print(f"\n因子名称示例: {factor_names[:20]}")
        
        # 保存因子数据
        output_path = Path(__file__).parent / 'data' / 'factors'
        output_path.mkdir(parents=True, exist_ok=True)
        factors.to_parquet(output_path / 'AAPL_alpha158.parquet')
        print(f"\n因子数据已保存到: {output_path / 'AAPL_alpha158.parquet'}")
        
    else:
        print("无法加载数据，请先运行数据获取")


def example_alpha360():
    """示例 3: 使用 Alpha360 因子库"""
    print("\n" + "="*60)
    print("示例 3: 使用 Alpha360 因子库")
    print("="*60)
    
    # 加载数据
    dm = DataManager()
    df = dm.get_stock_data('MSFT', market='us')
    
    if df is not None and len(df) > 0:
        # 计算 Alpha360（包含 Alpha158）
        factors = calculate_alpha360(df, include_alpha158=True)
        
        print(f"\n生成的因子数量: {len(factors.columns)}")
        print(f"数据行数: {len(factors)}")
        
        # 计算因子统计信息
        print(f"\n因子统计信息（前 10 个因子）:")
        print(factors.iloc[:, :10].describe())
        
        # 检查缺失值
        missing_ratio = factors.isna().sum() / len(factors)
        print(f"\n缺失值最多的 10 个因子:")
        print(missing_ratio.sort_values(ascending=False).head(10))
        
        # 只计算 Alpha360 扩展因子（不包含 Alpha158）
        factors_ext = calculate_alpha360(df, include_alpha158=False)
        print(f"\n仅 Alpha360 扩展因子数量: {len(factors_ext.columns)}")
        
    else:
        print("无法加载数据，请先运行数据获取")


def example_mag7_factors():
    """示例 4: 批量计算 Mag7 的因子"""
    print("\n" + "="*60)
    print("示例 4: 批量计算 Mag7 的因子")
    print("="*60)
    
    mag7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    dm = DataManager()
    
    all_factors = {}
    
    for symbol in mag7:
        print(f"\n处理 {symbol}...")
        df = dm.get_stock_data(symbol, market='us')
        
        if df is not None and len(df) > 0:
            # 计算 Alpha158 因子
            factors = calculate_alpha158(df)
            all_factors[symbol] = factors
            print(f"  - 生成 {len(factors.columns)} 个因子，{len(factors)} 条记录")
            
            # 保存
            output_path = Path(__file__).parent / 'data' / 'factors'
            output_path.mkdir(parents=True, exist_ok=True)
            factors.to_parquet(output_path / f'{symbol}_alpha158.parquet')
            print(f"  - 已保存到: {output_path / f'{symbol}_alpha158.parquet'}")
    
    print(f"\n总共处理了 {len(all_factors)} 只股票")
    
    # 合并所有股票的最新因子值（用于横截面分析）
    if all_factors:
        latest_factors = {}
        for symbol, factors in all_factors.items():
            latest_factors[symbol] = factors.iloc[-1]
        
        latest_df = pd.DataFrame(latest_factors).T
        print(f"\n最新因子数据（前 5 个因子）:")
        print(latest_df.iloc[:, :5])


def example_factor_analysis():
    """示例 5: 因子分析"""
    print("\n" + "="*60)
    print("示例 5: 因子分析")
    print("="*60)
    
    dm = DataManager()
    df = dm.get_stock_data('AAPL', market='us')
    
    if df is not None and len(df) > 0:
        # 计算因子
        factors = calculate_alpha158(df)
        
        # 计算未来收益率（用于因子有效性分析）
        df['return_5d'] = df['close'].pct_change(5).shift(-5)
        
        # 合并因子和收益率
        data = pd.concat([factors, df[['return_5d']]], axis=1)
        data = data.dropna()
        
        # 计算因子与未来收益率的相关性
        correlations = data.corr()['return_5d'].sort_values(ascending=False)
        
        print(f"\n与 5 日未来收益率相关性最高的 20 个因子:")
        print(correlations.head(20))
        
        print(f"\n与 5 日未来收益率相关性最低的 20 个因子:")
        print(correlations.tail(20))
        
        # 因子分组测试（简单的 IC 分析）
        # 选择一个因子进行分析
        factor_name = correlations.index[1]  # 选择相关性第二高的（第一个是自己）
        print(f"\n对因子 '{factor_name}' 进行分组测试:")
        
        # 按因子值分成 5 组
        data['quintile'] = pd.qcut(data[factor_name], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        # 计算每组的平均收益率
        group_returns = data.groupby('quintile')['return_5d'].mean()
        print(f"\n各分组平均收益率:")
        print(group_returns)
        
    else:
        print("无法加载数据，请先运行数据获取")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Qlib 风格因子库使用示例")
    print("="*60)
    
    try:
        # 运行各个示例
        example_basic_operators()
        example_alpha158()
        example_alpha360()
        example_mag7_factors()
        example_factor_analysis()
        
        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

