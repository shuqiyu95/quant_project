# Mag7 五日轮动策略 - 快速开始指南

## 概述

本策略实现了基于机器学习的 Mag7（美股七巨头）每周轮动择股系统：

- **标的池**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **预测目标**: 未来5天的收益率
- **交易频率**: 每周一调仓
- **选股数量**: 1只（预测收益率最高）
- **模型**: 随机森林 / 线性回归
- **损失函数**: RankMSE（排序均方误差）

## 核心特性

### 1. RankLoss 函数

实现了多种排序损失函数：

- **RankMSE**: 基于排名的均方误差，关注相对排序而非绝对数值
- **PairwiseRankLoss**: 成对比较损失，类似 LambdaRank 思想
- **ListNetLoss**: 基于概率分布的排序损失
- **BinaryClassificationLoss**: 简化为二分类问题（预测"赢家"）

### 2. 特征工程

使用 Qlib 风格的量价因子：

- 动量特征：5日、10日、20日收益率
- 波动率特征：5日、20日波动率
- 成交量特征：成交量比率
- 技术指标：RSI、MACD、均线比率等

### 3. 模型

支持多种模型：

- 随机森林（Random Forest）
- 线性回归（Linear Regression）
- Ridge 回归
- LASSO 回归
- GBDT

### 4. 回测引擎

完整的回测功能：

- 交易成本（佣金、滑点）
- 持仓管理
- 性能统计（夏普比率、最大回撤等）
- 交易记录

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 方法 1: 使用默认参数运行

```bash
python main_mag7_strategy.py
```

默认配置：
- 模型: Random Forest
- 损失函数: RankMSE
- 预测周期: 5天
- 初始资金: $100,000

### 方法 2: 数据集保存和加载 🆕

**首次运行：获取数据并保存**
```bash
# 保存处理好的数据集，方便后续快速实验
python main_mag7_strategy.py --save_dataset
```

**后续运行：加载数据集**
```bash
# 直接加载数据集，跳过数据获取和特征工程（节省 60-80% 时间）
python main_mag7_strategy.py --load_dataset output/dataset.pkl
```

**自定义日期范围**
```bash
# 指定开始和结束日期
python main_mag7_strategy.py \
    --start_date 2022-01-01 \
    --end_date 2024-12-31 \
    --save_dataset \
    --dataset_path output/dataset_2022_2024.pkl

# 后续加载使用
python main_mag7_strategy.py --load_dataset output/dataset_2022_2024.pkl
```

**快速测试不同模型**
```bash
# 使用同一数据集测试不同模型
python main_mag7_strategy.py --load_dataset output/dataset.pkl --model_type random_forest
python main_mag7_strategy.py --load_dataset output/dataset.pkl --model_type gbdt
python main_mag7_strategy.py --load_dataset output/dataset.pkl --model_type ridge
```

📖 详细说明请查看 [DATASET_USAGE.md](DATASET_USAGE.md)

### 方法 3: 自定义参数

```bash
python main_mag7_strategy.py \
    --model_type random_forest \
    --loss_type rank_mse \
    --forward_days 5 \
    --initial_capital 100000 \
    --save_model
```

参数说明：
- `--model_type`: 模型类型 (`random_forest`, `ridge`, `lasso`, `linear`, `gbdt`)
- `--loss_type`: 损失函数 (`mse`, `rank_mse`, `pairwise`, `listnet`)
- `--forward_days`: 预测未来几天
- `--initial_capital`: 初始资金
- `--start_date`: 数据开始日期 (YYYY-MM-DD) 🆕
- `--end_date`: 数据结束日期 (YYYY-MM-DD) 🆕
- `--years`: 获取几年的历史数据（默认1年，当 start_date 未指定时使用）
- `--test_ratio`: 测试集比例（默认0.3）
- `--save_model`: 是否保存训练好的模型
- `--save_dataset`: 保存处理好的数据集 🆕
- `--load_dataset`: 加载已保存的数据集路径 🆕
- `--dataset_path`: 数据集保存/加载路径（默认 output/dataset.pkl）🆕

### 示例：测试不同模型

```bash
# 随机森林 + RankMSE
python main_mag7_strategy.py --model_type random_forest --loss_type rank_mse --save_model

# Ridge 回归 + Pairwise Loss
python main_mag7_strategy.py --model_type ridge --loss_type pairwise --save_model

# GBDT + ListNet Loss
python main_mag7_strategy.py --model_type gbdt --loss_type listnet --save_model
```

## 输出结果

运行完成后，会在 `output/` 目录下生成：

1. **trades.csv**: 所有交易记录
2. **portfolio.csv**: 投资组合价值历史
3. **model_xxx.pkl**: 训练好的模型（如果使用 `--save_model`）
4. **dataset.pkl**: 处理好的数据集（如果使用 `--save_dataset`）🆕

同时会在终端打印详细的性能报告：

```
============================================================
              BACKTEST PERFORMANCE REPORT
============================================================

📈 RETURNS METRICS
------------------------------------------------------------
Total Return................................      12.45%
Annual Return...............................      18.23%
Avg Daily Return............................       0.07%
Best Day....................................       3.45%
Worst Day...................................      -2.89%
Final Value.................................  $112,450.00

⚠️  RISK METRICS
------------------------------------------------------------
Volatility (Annual).........................      15.67%
Max Drawdown................................      -8.34%
...

🎯 RISK-ADJUSTED RETURNS
------------------------------------------------------------
Sharpe Ratio................................       1.04
Sortino Ratio...............................       1.52
Calmar Ratio................................       2.19
...
```

## 运行测试

### 测试模型模块

```bash
python test/test_models.py
```

测试内容：
- RankLoss 函数
- 特征工程
- 预测模型

### 测试回测模块

```bash
python test/test_backtester.py
```

测试内容：
- 回测引擎
- 交易执行
- 性能分析

### 使用 pytest 运行全部测试

```bash
pytest test/ -v
```

## 策略流程

```
1. 数据获取
   ├─ 获取 Mag7 过去一年的日线数据
   └─ 使用 yfinance 自动缓存

2. 特征工程
   ├─ 计算量价因子（动量、波动率、成交量等）
   ├─ 生成标签（未来5天收益率）
   └─ 划分训练/测试集（时间序列方式）

3. 模型训练
   ├─ 使用训练集训练模型
   ├─ 在测试集上评估
   ├─ 计算排序指标（Spearman、NDCG等）
   └─ 显示特征重要性

4. 回测
   ├─ 每周一生成预测
   ├─ 选择预测收益率最高的股票
   ├─ 卖出当前持仓，买入新选中的股票
   └─ 考虑交易成本

5. 性能分析
   ├─ 计算收益指标（总收益、年化收益等）
   ├─ 计算风险指标（波动率、最大回撤等）
   ├─ 计算风险调整收益（夏普比率、索提诺比率等）
   └─ 生成报告和可视化
```

## 代码结构

```
src/
├── models/
│   ├── rank_loss.py           # RankLoss 函数实现
│   ├── feature_engineering.py # 特征工程
│   └── predictor.py           # 预测模型
├── backtester/
│   ├── engine.py              # 回测引擎
│   ├── strategy.py            # 交易策略
│   └── performance.py         # 性能分析
└── data_engine/               # 数据获取（已实现）

test/
├── test_models.py             # 模型测试
└── test_backtester.py         # 回测测试

main_mag7_strategy.py          # 主运行脚本
```

## 关键实现细节

### RankMSE Loss

```python
# 计算排名（值越大排名越高）
rank_true = rankdata(-y_true, method='average')
rank_pred = rankdata(-y_pred, method='average')

# 计算 MSE
loss = mean((rank_pred - rank_true)^2)
```

### 特征计算

```python
# 动量特征
features['return_5d'] = Ref(close, 5) / close - 1

# 波动率特征
returns = close / Ref(close, 1) - 1
features['volatility_5d'] = Std(returns, 5)

# 技术指标
features['rsi_14'] = RSI(close, 14)
features['macd'] = MACD(close)
```

### 回测策略

```python
# 每周一
if date.weekday() == 0:
    # 1. 为所有股票生成预测
    predictions = {}
    for symbol in MAG7:
        pred = model.predict(features[symbol])
        predictions[symbol] = pred
    
    # 2. 选择预测最高的股票
    best_symbol = max(predictions, key=predictions.get)
    
    # 3. 调仓
    sell_all_positions()
    buy(best_symbol, weight=1.0)
```

## 性能优化建议

1. **增加数据周期**: 使用 `--years 2` 获取更多历史数据
2. **调整模型参数**: 通过修改 `src/models/predictor.py` 中的默认参数
3. **特征选择**: 分析特征重要性，移除不重要的特征
4. **集成学习**: 组合多个模型的预测结果
5. **动态调仓**: 根据市场情况调整 top_k 数量

## 常见问题

### Q: 如何使用其他股票？

修改 `main_mag7_strategy.py` 中的 `MAG7_SYMBOLS` 列表：

```python
MAG7_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', ...]  # 添加你的股票
```

### Q: 如何改变调仓频率？

修改策略初始化参数：

```python
strategy = WeeklyRotationStrategy(
    rebalance_weekday=0  # 0=周一, 1=周二, ...
)

# 或使用 RankingStrategy 支持更灵活的频率
strategy = RankingStrategy(
    rebalance_freq='D'  # 'D'=每天, 'W'=每周, 'M'=每月
)
```

### Q: 如何同时持有多只股票？

修改 top_k 参数：

```python
strategy = WeeklyRotationStrategy(
    top_k=3  # 同时持有3只股票
)
```

### Q: 数据缓存在哪里？

数据缓存在 `data/` 目录下：
- `data/us/AAPL.parquet`
- `data/us/MSFT.parquet`
- ...

## 进一步改进

1. **添加止损止盈**: 在策略中实现风险控制
2. **优化特征**: 添加更多因子（如技术形态、资金流等）
3. **集成 Qlib**: 使用完整的 Qlib 因子库
4. **可视化**: 添加收益曲线、持仓变化等图表
5. **实时交易**: 连接到实盘交易接口

## 参考资料

- [Qlib 文档](https://qlib.readthedocs.io/)
- [LambdaRank 论文](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
- [ListNet 论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf)

## License

MIT

