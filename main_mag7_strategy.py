"""
Mag7 五日轮动策略 - 主运行脚本

实现完整的流程：
1. 数据获取（Mag7 过去一年数据）
2. 特征工程（基础量价因子）
3. 模型训练（随机森林/线性模型）
4. 回测（每周一调仓）
5. 性能分析

Usage:
    python main_mag7_strategy.py [--model_type random_forest] [--loss_type rank_mse]
"""

import os
import sys
import argparse
import pickle
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_engine import DataManager
from src.models import FeatureEngineer, StockPredictor
from src.backtester import BacktestEngine, WeeklyRotationStrategy, PerformanceAnalyzer
from src.backtester.strategy import run_backtest_with_strategy


# Mag7 股票列表
MAG7_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']


def fetch_mag7_data(
    data_dir: str = "data",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years: int = 1
) -> dict:
    """
    获取 Mag7 数据
    
    Args:
        data_dir: 数据目录
        start_date: 开始日期 (YYYY-MM-DD)，如果为None则自动计算
        end_date: 结束日期 (YYYY-MM-DD)，如果为None则使用当前日期
        years: 获取多少年的数据 (当start_date为None时使用)
        
    Returns:
        data_dict: {symbol: df}
    """
    print("=" * 60)
    print("📊 STEP 1: Fetching Mag7 Data")
    print("=" * 60)
    
    dm = DataManager(data_dir=data_dir)
    
    # 计算日期范围
    if end_date is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    if start_date is None:
        start_dt = end_dt - timedelta(days=365 * years)
    else:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    
    # 转换为字符串格式
    start_date_str = start_dt.strftime('%Y-%m-%d')
    end_date_str = end_dt.strftime('%Y-%m-%d')
    
    print(f"\nFetching data from {start_dt.date()} to {end_dt.date()}")
    print(f"Symbols: {', '.join(MAG7_SYMBOLS)}\n")
    
    data_dict = {}
    
    for symbol in MAG7_SYMBOLS:
        try:
            df = dm.fetch_data(
                symbol=symbol,
                start_date=start_date_str,
                end_date=end_date_str,
                use_cache=True
            )
            
            if df is not None and len(df) > 0:
                data_dict[symbol] = df
                print(f"✅ {symbol}: {len(df)} days")
            else:
                print(f"❌ {symbol}: No data")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    print(f"\n✅ Successfully fetched {len(data_dict)}/{len(MAG7_SYMBOLS)} stocks")
    
    return data_dict


def prepare_training_data(
    data_dict: dict,
    forward_days: int = 5,
    test_ratio: float = 0.3
) -> tuple:
    """
    准备训练数据
    
    Args:
        data_dict: {symbol: df}
        forward_days: 预测未来几天
        test_ratio: 测试集比例
        
    Returns:
        train_data, test_data, feature_engineer
    """
    print("\n" + "=" * 60)
    print("🔧 STEP 2: Feature Engineering")
    print("=" * 60)
    
    # 创建特征工程器
    fe = FeatureEngineer()
    
    # 准备数据集
    print("\nGenerating features and labels...")
    X, y, dates, symbols = fe.prepare_dataset(
        data_dict,
        forward_days=forward_days,
        min_periods=60
    )
    
    print(f"✅ Dataset prepared:")
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Date range: {min(dates).date()} to {max(dates).date()}")
    print(f"   Unique stocks: {len(set(symbols))}")
    
    # 划分训练测试集（按时间）
    n_samples = len(X)
    split_idx = int(n_samples * (1 - test_ratio))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    dates_train = dates[:split_idx]
    dates_test = dates[split_idx:]
    
    print(f"\nTrain/Test Split:")
    print(f"   Training: {len(X_train)} samples ({dates_train[0].date()} to {dates_train[-1].date()})")
    print(f"   Testing:  {len(X_test)} samples ({dates_test[0].date()} to {dates_test[-1].date()})")
    
    train_data = (X_train, y_train, dates_train)
    test_data = (X_test, y_test, dates_test)
    
    return train_data, test_data, fe


def save_dataset(
    train_data: tuple,
    test_data: tuple,
    feature_engineer: FeatureEngineer,
    save_path: str
):
    """
    保存处理好的数据集
    
    Args:
        train_data: (X_train, y_train, dates_train)
        test_data: (X_test, y_test, dates_test)
        feature_engineer: 特征工程器
        save_path: 保存路径（.pkl文件）
    """
    print("\n" + "=" * 60)
    print("💾 Saving Dataset")
    print("=" * 60)
    
    X_train, y_train, dates_train = train_data
    X_test, y_test, dates_test = test_data
    
    # 打包数据
    dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'dates_train': dates_train,
        'X_test': X_test,
        'y_test': y_test,
        'dates_test': dates_test,
        'feature_engineer': feature_engineer,
        'metadata': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train.shape[1],
            'train_date_range': (min(dates_train).date(), max(dates_train).date()),
            'test_date_range': (min(dates_test).date(), max(dates_test).date()),
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\n✅ Dataset saved to: {save_path}")
    print(f"   Training samples: {dataset['metadata']['train_samples']}")
    print(f"   Testing samples: {dataset['metadata']['test_samples']}")
    print(f"   Features: {dataset['metadata']['features']}")
    print(f"   Train dates: {dataset['metadata']['train_date_range'][0]} to {dataset['metadata']['train_date_range'][1]}")
    print(f"   Test dates: {dataset['metadata']['test_date_range'][0]} to {dataset['metadata']['test_date_range'][1]}")


def load_dataset(load_path: str) -> tuple:
    """
    加载处理好的数据集
    
    Args:
        load_path: 数据集路径（.pkl文件）
        
    Returns:
        train_data, test_data, feature_engineer
    """
    print("\n" + "=" * 60)
    print("📂 Loading Dataset")
    print("=" * 60)
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Dataset file not found: {load_path}")
    
    # 加载
    with open(load_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"\n✅ Dataset loaded from: {load_path}")
    print(f"   Saved at: {dataset['metadata']['saved_at']}")
    print(f"   Training samples: {dataset['metadata']['train_samples']}")
    print(f"   Testing samples: {dataset['metadata']['test_samples']}")
    print(f"   Features: {dataset['metadata']['features']}")
    print(f"   Train dates: {dataset['metadata']['train_date_range'][0]} to {dataset['metadata']['train_date_range'][1]}")
    print(f"   Test dates: {dataset['metadata']['test_date_range'][0]} to {dataset['metadata']['test_date_range'][1]}")
    
    train_data = (dataset['X_train'], dataset['y_train'], dataset['dates_train'])
    test_data = (dataset['X_test'], dataset['y_test'], dataset['dates_test'])
    feature_engineer = dataset['feature_engineer']
    
    return train_data, test_data, feature_engineer


def train_model(
    train_data: tuple,
    test_data: tuple,
    model_type: str = 'random_forest',
    loss_type: str = 'rank_mse',
    save_path: Optional[str] = None
) -> StockPredictor:
    """
    训练预测模型
    
    Args:
        train_data: (X_train, y_train, dates_train)
        test_data: (X_test, y_test, dates_test)
        model_type: 模型类型
        loss_type: 损失函数类型
        save_path: 模型保存路径
        
    Returns:
        predictor: 训练好的模型
    """
    print("\n" + "=" * 60)
    print("🤖 STEP 3: Training Model")
    print("=" * 60)
    
    X_train, y_train, _ = train_data
    X_test, y_test, _ = test_data
    
    print(f"\nModel Type: {model_type}")
    print(f"Loss Type: {loss_type}")
    
    # 创建并训练模型
    predictor = StockPredictor(
        model_type=model_type,
        loss_type=loss_type,
        scale_features=True
    )
    
    print("\nTraining...")
    predictor.fit(X_train, y_train)
    print("✅ Training completed!")
    
    # 评估模型
    print("\n" + "-" * 60)
    print("Model Evaluation")
    print("-" * 60)
    
    # 训练集评估
    train_metrics = predictor.evaluate(X_train, y_train, k=3)
    print("\n📊 Training Set:")
    for key, value in train_metrics.items():
        print(f"   {key}: {value:.4f}")
    
    # 测试集评估
    test_metrics = predictor.evaluate(X_test, y_test, k=3)
    print("\n📊 Test Set:")
    for key, value in test_metrics.items():
        print(f"   {key}: {value:.4f}")
    
    # 特征重要性
    if hasattr(predictor.model, 'feature_importances_'):
        print("\n📈 Top 10 Important Features:")
        importance_df = predictor.get_feature_importance(top_k=10)
        for idx, row in importance_df.iterrows():
            print(f"   {row['feature']:.<30} {row['importance']:.4f}")
    
    # 保存模型
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        predictor.save(save_path)
        print(f"\n💾 Model saved to {save_path}")
    
    return predictor


def run_backtest(
    predictor: StockPredictor,
    feature_engineer: FeatureEngineer,
    data_dict: dict,
    initial_capital: float = 100000,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> dict:
    """
    运行回测
    
    Args:
        predictor: 预测模型
        feature_engineer: 特征工程器
        data_dict: 数据字典
        initial_capital: 初始资金
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        results: 回测结果
    """
    print("\n" + "=" * 60)
    print("🔄 STEP 4: Running Backtest")
    print("=" * 60)
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=0.001,  # 0.1% 佣金
        slippage_rate=0.001,    # 0.1% 滑点
        market='US'
    )
    
    # 创建策略
    strategy = WeeklyRotationStrategy(
        predictor=predictor,
        feature_engineer=feature_engineer,
        top_k=1,  # 每次只选1只股票
        rebalance_weekday=0  # 周一
    )
    
    print(f"\nStrategy: {strategy.name}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Top K: {strategy.top_k}")
    print(f"Rebalance: Every Monday")
    
    # 运行回测
    print("\nRunning backtest...")
    results = run_backtest_with_strategy(
        engine=engine,
        strategy=strategy,
        data=data_dict,
        start_date=start_date,
        end_date=end_date,
        verbose=True
    )
    
    print("\n✅ Backtest completed!")
    
    return results


def analyze_performance(results: dict):
    """
    分析性能
    
    Args:
        results: 回测结果
    """
    print("\n" + "=" * 60)
    print("📈 STEP 5: Performance Analysis")
    print("=" * 60)
    
    # 创建性能分析器
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    
    # 分析
    analysis = analyzer.analyze(
        portfolio_df=results['portfolio'],
        trades_df=results['trades']
    )
    
    # 打印报告
    analyzer.print_report(analysis)
    
    # 保存结果
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存交易记录
    if len(results['trades']) > 0:
        trades_path = os.path.join(output_dir, "trades.csv")
        results['trades'].to_csv(trades_path, index=False)
        print(f"\n💾 Trades saved to {trades_path}")
    
    # 保存投资组合历史
    portfolio_path = os.path.join(output_dir, "portfolio.csv")
    results['portfolio'].to_csv(portfolio_path, index=False)
    print(f"💾 Portfolio history saved to {portfolio_path}")
    
    return analysis


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Mag7 Weekly Rotation Strategy')
    parser.add_argument('--model_type', type=str, default='random_forest',
                       choices=['random_forest', 'ridge', 'lasso', 'linear', 'gbdt'],
                       help='Model type')
    parser.add_argument('--loss_type', type=str, default='rank_mse',
                       choices=['mse', 'rank_mse', 'pairwise', 'listnet'],
                       help='Loss function type')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--start_date', type=str, default=None,
                       help='Start date for data fetching (YYYY-MM-DD). If not provided, will use --years')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date for data fetching (YYYY-MM-DD). If not provided, will use current date')
    parser.add_argument('--years', type=int, default=1,
                       help='Years of historical data (used when --start_date is not provided)')
    parser.add_argument('--forward_days', type=int, default=5,
                       help='Forward prediction days')
    parser.add_argument('--initial_capital', type=float, default=100000,
                       help='Initial capital')
    parser.add_argument('--test_ratio', type=float, default=0.3,
                       help='Test set ratio')
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--save_dataset', action='store_true',
                       help='Save processed train/test dataset')
    parser.add_argument('--load_dataset', type=str, default=None,
                       help='Load processed dataset from path (skip data fetching and feature engineering)')
    parser.add_argument('--dataset_path', type=str, default='output/dataset.pkl',
                       help='Path to save/load dataset')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MAG7 WEEKLY ROTATION STRATEGY".center(60))
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model Type: {args.model_type}")
    print(f"  Loss Type: {args.loss_type}")
    print(f"  Forward Days: {args.forward_days}")
    print(f"  Initial Capital: ${args.initial_capital:,.2f}")
    print(f"  Test Ratio: {args.test_ratio}")
    if args.start_date:
        print(f"  Start Date: {args.start_date}")
    if args.end_date:
        print(f"  End Date: {args.end_date}")
    if args.load_dataset:
        print(f"  Loading Dataset: {args.load_dataset}")
    if args.save_dataset:
        print(f"  Saving Dataset: {args.dataset_path}")
    
    try:
        # 判断是加载数据集还是重新生成
        if args.load_dataset:
            # 从文件加载数据集
            train_data, test_data, feature_engineer = load_dataset(args.load_dataset)
            # 如果需要回测，还需要获取原始数据
            data_dict = fetch_mag7_data(
                data_dir=args.data_dir,
                start_date=args.start_date,
                end_date=args.end_date,
                years=args.years
            )
        else:
            # 1. 获取数据
            data_dict = fetch_mag7_data(
                data_dir=args.data_dir,
                start_date=args.start_date,
                end_date=args.end_date,
                years=args.years
            )
            
            if len(data_dict) < 3:
                print("\n❌ Error: Not enough data. Need at least 3 stocks.")
                return
            
            # 2. 准备训练数据
            train_data, test_data, feature_engineer = prepare_training_data(
                data_dict,
                forward_days=args.forward_days,
                test_ratio=args.test_ratio
            )
            
            # 保存数据集（如果需要）
            if args.save_dataset:
                save_dataset(
                    train_data,
                    test_data,
                    feature_engineer,
                    args.dataset_path
                )
        
        # 3. 训练模型
        model_path = f"output/model_{args.model_type}_{args.loss_type}.pkl" if args.save_model else None
        predictor = train_model(
            train_data,
            test_data,
            model_type=args.model_type,
            loss_type=args.loss_type,
            save_path=model_path
        )
        
        # 4. 运行回测（使用测试集日期范围）
        _, _, dates_test = test_data
        start_date = pd.Timestamp(min(dates_test))
        end_date = pd.Timestamp(max(dates_test))
        
        results = run_backtest(
            predictor=predictor,
            feature_engineer=feature_engineer,
            data_dict=data_dict,
            initial_capital=args.initial_capital,
            start_date=start_date,
            end_date=end_date
        )
        
        # 5. 性能分析
        analysis = analyze_performance(results)
        
        print("\n" + "=" * 60)
        print("✅ ALL DONE!".center(60))
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

