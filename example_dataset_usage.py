"""
数据集保存和加载示例

演示如何使用新的数据集管理功能来加速实验流程。
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("""
================================================================================
                  数据集管理功能使用示例
================================================================================

本脚本演示三个常见使用场景：

1️⃣  首次运行：获取数据、处理并保存数据集
2️⃣  快速实验：加载数据集测试不同模型
3️⃣  多时间段：创建不同时间范围的数据集

================================================================================
""")

print("\n" + "=" * 80)
print("场景 1: 首次运行 - 保存数据集")
print("=" * 80)

print("""
命令：
python main_mag7_strategy.py \\
    --start_date 2023-01-01 \\
    --end_date 2024-12-31 \\
    --save_dataset \\
    --dataset_path output/mag7_2023_2024.pkl \\
    --model_type random_forest

说明：
- 从 yfinance 获取 2023-2024 年的 Mag7 数据
- 计算所有量价因子（约 40+ 个特征）
- 将处理好的数据集保存到 output/mag7_2023_2024.pkl
- 训练随机森林模型并运行回测
- 首次运行时间：约 2-3 分钟
""")

print("\n" + "=" * 80)
print("场景 2: 快速实验 - 加载数据集测试不同模型")
print("=" * 80)

print("""
命令：
# 测试随机森林
python main_mag7_strategy.py \\
    --load_dataset output/mag7_2023_2024.pkl \\
    --model_type random_forest \\
    --loss_type rank_mse

# 测试 GBDT
python main_mag7_strategy.py \\
    --load_dataset output/mag7_2023_2024.pkl \\
    --model_type gbdt \\
    --loss_type rank_mse

# 测试岭回归
python main_mag7_strategy.py \\
    --load_dataset output/mag7_2023_2024.pkl \\
    --model_type ridge \\
    --loss_type mse

说明：
- 直接加载之前保存的数据集
- 跳过数据获取和特征工程步骤
- 只需训练模型和运行回测
- 运行时间：约 30 秒
- 时间节省：约 70%
""")

print("\n" + "=" * 80)
print("场景 3: 多时间段 - 创建不同数据集进行对比")
print("=" * 80)

print("""
命令：
# 创建 2022 数据集
python main_mag7_strategy.py \\
    --start_date 2022-01-01 \\
    --end_date 2022-12-31 \\
    --save_dataset \\
    --dataset_path output/mag7_2022.pkl

# 创建 2023 数据集
python main_mag7_strategy.py \\
    --start_date 2023-01-01 \\
    --end_date 2023-12-31 \\
    --save_dataset \\
    --dataset_path output/mag7_2023.pkl

# 创建 2024 数据集
python main_mag7_strategy.py \\
    --start_date 2024-01-01 \\
    --end_date 2024-12-31 \\
    --save_dataset \\
    --dataset_path output/mag7_2024.pkl

# 使用同一模型测试不同年份的数据
python main_mag7_strategy.py --load_dataset output/mag7_2022.pkl --model_type random_forest
python main_mag7_strategy.py --load_dataset output/mag7_2023.pkl --model_type random_forest
python main_mag7_strategy.py --load_dataset output/mag7_2024.pkl --model_type random_forest

说明：
- 创建多个时间段的数据集
- 使用相同模型测试不同市场环境
- 评估策略在不同时期的稳定性
- 便于进行前向测试（walk-forward testing）
""")

print("\n" + "=" * 80)
print("数据集文件内容")
print("=" * 80)

print("""
每个 .pkl 文件包含：

📊 训练数据：
   - X_train: 训练特征（DataFrame）
   - y_train: 训练标签（Series）
   - dates_train: 训练日期（List）

📊 测试数据：
   - X_test: 测试特征（DataFrame）
   - y_test: 测试标签（Series）
   - dates_test: 测试日期（List）

🔧 工具：
   - feature_engineer: 特征工程器对象

📝 元信息：
   - train_samples: 训练样本数
   - test_samples: 测试样本数
   - features: 特征数量
   - train_date_range: 训练日期范围
   - test_date_range: 测试日期范围
   - saved_at: 保存时间

文件大小：约 5-20 MB（取决于时间范围）
""")

print("\n" + "=" * 80)
print("最佳实践")
print("=" * 80)

print("""
✅ 实验管理
   - 为不同实验创建不同的数据集文件
   - 使用描述性文件名，如 mag7_2023_rf.pkl

✅ 命名规范
   - 包含时间范围：dataset_2023_2024.pkl
   - 包含特殊配置：dataset_forward7d.pkl
   
✅ 版本控制
   - 保存重要实验的数据集副本
   - 定期更新数据集以包含最新数据

✅ 快速迭代
   1. 首次运行使用 --save_dataset
   2. 后续实验使用 --load_dataset
   3. 测试完所有模型后再更新数据

⚠️  注意事项
   - 确保有足够的磁盘空间（每个文件约 5-20 MB）
   - 数据集包含的日期范围应满足回测需求
   - 不同特征工程配置需要不同的数据集
""")

print("\n" + "=" * 80)
print("实际运行建议")
print("=" * 80)

print(f"""
现在就开始使用数据集功能：

步骤 1: 创建并保存数据集
    cd {os.path.dirname(os.path.abspath(__file__))}
    python main_mag7_strategy.py --save_dataset

步骤 2: 快速测试不同模型（使用保存的数据集）
    python main_mag7_strategy.py --load_dataset output/dataset.pkl --model_type gbdt
    python main_mag7_strategy.py --load_dataset output/dataset.pkl --model_type ridge

步骤 3: 查看结果
    ls -lh output/
    # 你会看到：
    # - dataset.pkl (保存的数据集)
    # - trades.csv (交易记录)
    # - portfolio.csv (投资组合历史)
    # - model_*.pkl (训练好的模型)

更多详细信息：
    - 完整使用指南：DATASET_USAGE.md
    - Mag7 策略说明：QUICKSTART_MAG7.md
    - 项目文档：README.md
""")

print("\n" + "=" * 80)
print("🎉 开始你的量化实验之旅！")
print("=" * 80 + "\n")

