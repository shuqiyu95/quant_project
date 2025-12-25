"""
A股次日高点预测策略 - 金风科技演示

演示完整流程：
1. 数据获取：金风科技 (002202) 历史数据
2. 特征工程：Alpha158 + A股特色因子
3. 模型训练：多分类预测次日高点区间
4. 回测评估：评估策略收益
5. 模型保存：保存训练好的模型

作者：Quant Team
日期：2025-12-23
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cn_intraday_high_strategy import CNIntradayHighPredictor


def main():
    """主函数"""
    
    print("=" * 80)
    print("🚀 A股次日高点预测策略 - 金风科技演示")
    print("=" * 80)
    
    # ========== 配置参数 ==========
    SYMBOL = '002202'  # 金风科技
    STOCK_NAME = '金风科技'
    START_DATE = '2022-01-01'
    END_DATE = '2024-12-20'
    
    MODEL_TYPE = 'random_forest'  # 'random_forest' or 'gbdt'
    INITIAL_CAPITAL = 100000.0
    
    OUTPUT_DIR = 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n📋 Configuration:")
    print(f"   Stock: {STOCK_NAME} ({SYMBOL})")
    print(f"   Date range: {START_DATE} to {END_DATE}")
    print(f"   Model: {MODEL_TYPE}")
    print(f"   Initial capital: ¥{INITIAL_CAPITAL:,.2f}")
    
    # ========== 步骤 1: 初始化预测器 ==========
    print(f"\n{'='*80}")
    print("STEP 1: Initialize Predictor")
    print("="*80)
    
    predictor = CNIntradayHighPredictor(
        model_type=MODEL_TYPE,
        data_dir='data'
    )
    
    # ========== 步骤 2: 准备数据集 ==========
    print(f"\n{'='*80}")
    print("STEP 2: Prepare Dataset")
    print("="*80)
    
    try:
        X, y, dates = predictor.prepare_dataset(
            symbol=SYMBOL,
            start_date=START_DATE,
            end_date=END_DATE,
            use_cache=True,
            min_periods=60
        )
    except Exception as e:
        print(f"\n❌ Error preparing dataset: {e}")
        print("\n💡 Tip: 请确保:")
        print("   1. 股票代码正确 (金风科技: 002202)")
        print("   2. 网络连接正常")
        print("   3. AkShare 可以正常访问数据源")
        return
    
    # 检查数据量
    if len(X) < 100:
        print(f"\n⚠️  Warning: Dataset too small ({len(X)} samples)")
        print("   建议至少有 100+ 样本用于训练")
        return
    
    # ========== 步骤 3: 训练模型 ==========
    print(f"\n{'='*80}")
    print("STEP 3: Train Model")
    print("="*80)
    
    predictor.train(
        X=X,
        y=y,
        validation_split=0.2
    )
    
    # ========== 步骤 4: 回测评估 ==========
    print(f"\n{'='*80}")
    print("STEP 4: Backtest Strategy")
    print("="*80)
    
    # 使用后20%的数据作为测试集
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    dates_test = dates[split_idx:]
    
    # 获取日线数据（用于回测）
    daily_df = predictor.dm.fetch_data(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        use_cache=True
    )
    
    backtest_results = predictor.backtest(
        symbol=SYMBOL,
        X_test=X_test,
        dates_test=dates_test,
        daily_df=daily_df,
        initial_capital=INITIAL_CAPITAL
    )
    
    # ========== 步骤 5: 保存结果 ==========
    print(f"\n{'='*80}")
    print("STEP 5: Save Results")
    print("="*80)
    
    # 保存模型
    model_path = os.path.join(OUTPUT_DIR, f'cn_intraday_high_{SYMBOL}.pkl')
    predictor.save(model_path)
    
    # 保存交易记录
    if len(backtest_results['trades']) > 0:
        trades_path = os.path.join(OUTPUT_DIR, f'trades_{SYMBOL}.csv')
        backtest_results['trades'].to_csv(trades_path, index=False, encoding='utf-8-sig')
        print(f"✅ Trades saved to {trades_path}")
    
    # 保存投资组合历史
    portfolio_df = backtest_results['engine'].get_portfolio_df()
    if len(portfolio_df) > 0:
        portfolio_path = os.path.join(OUTPUT_DIR, f'portfolio_{SYMBOL}.csv')
        portfolio_df.to_csv(portfolio_path, index=False, encoding='utf-8-sig')
        print(f"✅ Portfolio history saved to {portfolio_path}")
    
    # ========== 总结 ==========
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print("="*80)
    
    stats = backtest_results['portfolio_stats']
    
    print(f"\n🎯 Strategy: 次日高点预测 (预测 > 3% 时开盘买入，30分钟后卖出)")
    print(f"\n📈 Performance Metrics:")
    print(f"   Initial Capital: ¥{stats['initial_capital']:,.2f}")
    print(f"   Final Value: ¥{stats['final_value']:,.2f}")
    print(f"   Total Return: {stats['total_return']:.2%}")
    print(f"   Annual Return: {stats['annual_return']:.2%}")
    print(f"   Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
    print(f"   Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"   Win Rate: {stats['win_rate']:.2%}")
    print(f"   Total Trades: {stats['n_trades']}")
    
    # 评估策略表现
    print(f"\n💡 Strategy Evaluation:")
    if stats['total_return'] > 0:
        print(f"   ✅ Profitable strategy (+{stats['total_return']:.2%})")
    else:
        print(f"   ⚠️  Loss-making strategy ({stats['total_return']:.2%})")
    
    if stats['sharpe_ratio'] > 1.0:
        print(f"   ✅ Good risk-adjusted return (Sharpe: {stats['sharpe_ratio']:.2f})")
    elif stats['sharpe_ratio'] > 0.5:
        print(f"   📊 Moderate risk-adjusted return (Sharpe: {stats['sharpe_ratio']:.2f})")
    else:
        print(f"   ⚠️  Low risk-adjusted return (Sharpe: {stats['sharpe_ratio']:.2f})")
    
    if abs(stats['max_drawdown']) < 0.2:
        print(f"   ✅ Controlled drawdown ({stats['max_drawdown']:.2%})")
    else:
        print(f"   ⚠️  Large drawdown ({stats['max_drawdown']:.2%})")
    
    print(f"\n{'='*80}")
    print("✅ Demo completed successfully!")
    print("="*80)
    
    print(f"\n📁 Output files:")
    print(f"   - Model: {model_path}")
    if len(backtest_results['trades']) > 0:
        print(f"   - Trades: {trades_path}")
    if len(portfolio_df) > 0:
        print(f"   - Portfolio: {portfolio_path}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. 查看交易明细: trades_{SYMBOL}.csv")
    print(f"   2. 分析投资组合变化: portfolio_{SYMBOL}.csv")
    print(f"   3. 优化模型参数以提升表现")
    print(f"   4. 尝试其他股票代码")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

