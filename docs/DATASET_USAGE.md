# Dataset 保存和加载使用指南

## 📋 概述

`main_mag7_strategy.py` 现在支持保存和加载处理好的训练/测试数据集，避免重复的数据获取和特征工程过程。

## 🆕 新增功能

### 1. 日期参数控制

现在可以通过命令行参数指定数据获取的日期范围：

```bash
# 使用默认配置（过去1年）
python main_mag7_strategy.py

# 指定开始和结束日期
python main_mag7_strategy.py --start_date 2023-01-01 --end_date 2024-12-31

# 只指定开始日期（结束日期使用当前日期）
python main_mag7_strategy.py --start_date 2023-01-01

# 使用years参数（当不指定start_date时）
python main_mag7_strategy.py --years 2
```

### 2. 数据集保存

保存处理好的数据集，避免重复的数据获取和特征工程：

```bash
# 获取数据、处理并保存数据集
python main_mag7_strategy.py --save_dataset

# 自定义保存路径
python main_mag7_strategy.py --save_dataset --dataset_path output/my_dataset.pkl

# 指定日期范围并保存
python main_mag7_strategy.py \
    --start_date 2022-01-01 \
    --end_date 2024-12-31 \
    --save_dataset \
    --dataset_path output/dataset_2022_2024.pkl
```

### 3. 数据集加载

直接加载之前保存的数据集，跳过数据获取和特征工程步骤：

```bash
# 加载数据集并运行完整流程
python main_mag7_strategy.py --load_dataset output/dataset.pkl

# 加载数据集并使用不同的模型
python main_mag7_strategy.py \
    --load_dataset output/dataset.pkl \
    --model_type gbdt \
    --loss_type rank_mse
```

## 💾 数据集文件内容

保存的 `.pkl` 文件包含：

- **X_train**: 训练特征 (DataFrame)
- **y_train**: 训练标签 (Series)
- **dates_train**: 训练日期 (List)
- **X_test**: 测试特征 (DataFrame)
- **y_test**: 测试标签 (Series)
- **dates_test**: 测试日期 (List)
- **feature_engineer**: 特征工程器对象 (FeatureEngineer)
- **metadata**: 数据集元信息
  - train_samples: 训练样本数
  - test_samples: 测试样本数
  - features: 特征数量
  - train_date_range: 训练日期范围
  - test_date_range: 测试日期范围
  - saved_at: 保存时间

## 🎯 实用场景

### 场景 1: 首次运行，保存数据集

```bash
# 获取2年的数据，处理后保存
python main_mag7_strategy.py \
    --years 2 \
    --save_dataset \
    --dataset_path output/mag7_2y.pkl \
    --model_type random_forest \
    --loss_type rank_mse
```

### 场景 2: 测试不同模型（使用同一数据集）

```bash
# 测试随机森林
python main_mag7_strategy.py \
    --load_dataset output/mag7_2y.pkl \
    --model_type random_forest \
    --loss_type rank_mse

# 测试 GBDT
python main_mag7_strategy.py \
    --load_dataset output/mag7_2y.pkl \
    --model_type gbdt \
    --loss_type rank_mse

# 测试岭回归
python main_mag7_strategy.py \
    --load_dataset output/mag7_2y.pkl \
    --model_type ridge \
    --loss_type mse
```

### 场景 3: 准备多个时间段的数据集

```bash
# 2022-2023 数据集
python main_mag7_strategy.py \
    --start_date 2022-01-01 \
    --end_date 2023-12-31 \
    --save_dataset \
    --dataset_path output/mag7_2022_2023.pkl

# 2023-2024 数据集
python main_mag7_strategy.py \
    --start_date 2023-01-01 \
    --end_date 2024-12-31 \
    --save_dataset \
    --dataset_path output/mag7_2023_2024.pkl

# 后续使用不同数据集测试模型泛化能力
python main_mag7_strategy.py --load_dataset output/mag7_2022_2023.pkl
python main_mag7_strategy.py --load_dataset output/mag7_2023_2024.pkl
```

## 📊 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--start_date` | str | None | 数据开始日期 (YYYY-MM-DD) |
| `--end_date` | str | None | 数据结束日期 (YYYY-MM-DD) |
| `--years` | int | 1 | 历史数据年数（当start_date为空时使用） |
| `--save_dataset` | flag | False | 保存处理好的数据集 |
| `--load_dataset` | str | None | 加载数据集路径 |
| `--dataset_path` | str | output/dataset.pkl | 数据集保存路径 |
| `--model_type` | str | random_forest | 模型类型 |
| `--loss_type` | str | rank_mse | 损失函数类型 |
| `--forward_days` | int | 5 | 预测未来天数 |
| `--test_ratio` | float | 0.3 | 测试集比例 |
| `--initial_capital` | float | 100000 | 初始资金 |
| `--save_model` | flag | False | 保存训练好的模型 |

## ⚡ 性能优化

使用数据集保存/加载功能可以显著提升效率：

1. **首次运行**（数据获取 + 特征工程 + 训练 + 回测）：~2-3分钟
2. **后续运行**（加载数据集 + 训练 + 回测）：~30秒

**时间节省**：约 60-80% 的时间

## 🔍 调试和验证

加载数据集时会显示详细信息：

```
============================================================
📂 Loading Dataset
============================================================

✅ Dataset loaded from: output/dataset.pkl
   Saved at: 2024-12-22 10:30:45
   Training samples: 1250
   Testing samples: 537
   Features: 42
   Train dates: 2023-01-03 to 2024-05-15
   Test dates: 2024-05-16 to 2024-12-20
```

## ⚠️ 注意事项

1. **数据集版本管理**: 建议在文件名中包含日期或版本信息
2. **特征一致性**: 同一数据集应使用相同的特征工程配置
3. **时间范围**: 确保加载的数据集日期范围满足研究需求
4. **磁盘空间**: 每个数据集文件约 5-20 MB

## 🚀 最佳实践

1. **实验管理**: 为不同实验创建不同的数据集文件
2. **命名规范**: 使用描述性文件名，如 `mag7_2023_rf.pkl`
3. **定期更新**: 定期重新获取数据并更新数据集
4. **备份重要数据集**: 保存关键实验的数据集副本

