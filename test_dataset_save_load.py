"""
测试数据集保存和加载功能

Usage:
    python test_dataset_save_load.py
"""

import os
import sys
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_engine import DataManager
from src.models import FeatureEngineer

# Test configuration
MAG7_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']  # 使用3个股票进行快速测试
TEST_DATASET_PATH = "output/test_dataset.pkl"


def test_dataset_save_load():
    """测试数据集的保存和加载"""
    
    print("\n" + "=" * 60)
    print("测试数据集保存和加载功能")
    print("=" * 60)
    
    # Step 1: 获取测试数据
    print("\n📊 Step 1: 获取测试数据...")
    dm = DataManager(data_dir="data")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6个月数据用于测试
    
    data_dict = {}
    for symbol in MAG7_SYMBOLS:
        try:
            df = dm.fetch_data(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                use_cache=True
            )
            if df is not None and len(df) > 0:
                data_dict[symbol] = df
                print(f"✅ {symbol}: {len(df)} days")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    if len(data_dict) < 2:
        print("❌ 测试失败：数据不足")
        return False
    
    # Step 2: 特征工程
    print("\n🔧 Step 2: 特征工程...")
    fe = FeatureEngineer()
    
    X, y, dates, symbols = fe.prepare_dataset(
        data_dict,
        forward_days=5,
        min_periods=30
    )
    
    print(f"✅ 特征: {X.shape}, 标签: {y.shape}")
    
    # 划分训练测试集
    n_samples = len(X)
    split_idx = int(n_samples * 0.7)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    dates_train = dates[:split_idx]
    dates_test = dates[split_idx:]
    
    train_data = (X_train, y_train, dates_train)
    test_data = (X_test, y_test, dates_test)
    
    # Step 3: 保存数据集
    print("\n💾 Step 3: 保存数据集...")
    import pickle
    
    dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'dates_train': dates_train,
        'X_test': X_test,
        'y_test': y_test,
        'dates_test': dates_test,
        'feature_engineer': fe,
        'metadata': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train.shape[1],
            'train_date_range': (min(dates_train).date(), max(dates_train).date()),
            'test_date_range': (min(dates_test).date(), max(dates_test).date()),
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    os.makedirs(os.path.dirname(TEST_DATASET_PATH), exist_ok=True)
    
    with open(TEST_DATASET_PATH, 'wb') as f:
        pickle.dump(dataset, f)
    
    file_size = os.path.getsize(TEST_DATASET_PATH) / 1024 / 1024  # MB
    print(f"✅ 数据集已保存到: {TEST_DATASET_PATH}")
    print(f"   文件大小: {file_size:.2f} MB")
    print(f"   训练样本: {dataset['metadata']['train_samples']}")
    print(f"   测试样本: {dataset['metadata']['test_samples']}")
    
    # Step 4: 加载数据集
    print("\n📂 Step 4: 加载数据集...")
    
    with open(TEST_DATASET_PATH, 'rb') as f:
        loaded_dataset = pickle.load(f)
    
    print(f"✅ 数据集已加载")
    print(f"   保存时间: {loaded_dataset['metadata']['saved_at']}")
    print(f"   训练样本: {loaded_dataset['metadata']['train_samples']}")
    print(f"   测试样本: {loaded_dataset['metadata']['test_samples']}")
    
    # Step 5: 验证数据一致性
    print("\n🔍 Step 5: 验证数据一致性...")
    
    checks = [
        ("训练特征", X_train.shape == loaded_dataset['X_train'].shape),
        ("训练标签", y_train.shape == loaded_dataset['y_train'].shape),
        ("测试特征", X_test.shape == loaded_dataset['X_test'].shape),
        ("测试标签", y_test.shape == loaded_dataset['y_test'].shape),
        ("训练日期", len(dates_train) == len(loaded_dataset['dates_train'])),
        ("测试日期", len(dates_test) == len(loaded_dataset['dates_test'])),
        ("特征工程器", loaded_dataset['feature_engineer'] is not None),
    ]
    
    all_passed = True
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    # Step 6: 清理测试文件
    print("\n🧹 Step 6: 清理测试文件...")
    if os.path.exists(TEST_DATASET_PATH):
        os.remove(TEST_DATASET_PATH)
        print(f"✅ 已删除测试文件: {TEST_DATASET_PATH}")
    
    # 最终结果
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有测试通过！")
        print("=" * 60)
        return True
    else:
        print("❌ 部分测试失败")
        print("=" * 60)
        return False


if __name__ == "__main__":
    try:
        success = test_dataset_save_load()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

