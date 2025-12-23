# 更新日志

本文档记录项目的重要变更和版本历史。

---

## [1.2.0] - 2024-12-22

### 🎯 主题：数据集管理与日期参数控制

#### ✨ 新增功能

##### 数据集管理系统
- **数据集保存功能**
  - 保存处理好的训练/测试数据集到 `.pkl` 文件
  - 包含完整元信息（样本数、特征数、日期范围等）
  - 节省 60-80% 的运行时间
  - 方便测试不同模型和参数组合

- **数据集加载功能**
  - 快速加载已保存的数据集
  - 跳过数据获取和特征工程步骤
  - 支持灵活的数据集路径配置

##### 日期参数控制
- 新增 `--start_date` 参数：指定数据开始日期 (YYYY-MM-DD)
- 新增 `--end_date` 参数：指定数据结束日期 (YYYY-MM-DD)
- 支持自定义时间范围进行回测
- 向后兼容原有的 `--years` 参数

#### 📝 新增命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--start_date` | str | None | 数据开始日期 (YYYY-MM-DD) |
| `--end_date` | str | None | 数据结束日期 (YYYY-MM-DD) |
| `--save_dataset` | flag | False | 保存处理好的数据集 |
| `--load_dataset` | str | None | 加载已保存的数据集路径 |
| `--dataset_path` | str | output/dataset.pkl | 数据集保存/加载路径 |

#### 💻 使用示例

```bash
# 首次运行：获取数据并保存数据集
python main_mag7_strategy.py \
    --start_date 2022-01-01 \
    --end_date 2024-12-31 \
    --save_dataset

# 后续运行：加载数据集快速测试不同模型
python main_mag7_strategy.py --load_dataset output/dataset.pkl --model_type gbdt
python main_mag7_strategy.py --load_dataset output/dataset.pkl --model_type ridge
```

#### ⚡ 性能优化

| 场景 | 原方案 | 新方案（数据集加载） | 时间节省 |
|------|--------|---------------------|----------|
| 首次运行 | 2-3 分钟 | 2-3 分钟 + 保存 | - |
| 测试新模型 | 2-3 分钟 | ~30 秒 | 70-80% |
| 测试3个模型 | 6-9 分钟 | ~1.5 分钟 | 75-83% |

#### 📚 新增文档
- ✅ `DATASET_USAGE.md` - 数据集管理详细指南
- ✅ `test_dataset_save_load.py` - 数据集功能测试脚本
- ✅ `example_dataset_usage.py` - 数据集使用示例
- ✅ 更新 `QUICKSTART_MAG7.md` - 添加数据集使用方法
- ✅ 更新 `README.md` - 添加新功能说明

#### 🔧 技术细节

**数据集文件结构**：
```python
dataset = {
    'X_train': DataFrame,           # 训练特征
    'y_train': Series,              # 训练标签
    'dates_train': List,            # 训练日期
    'X_test': DataFrame,            # 测试特征
    'y_test': Series,               # 测试标签
    'dates_test': List,             # 测试日期
    'feature_engineer': FeatureEngineer,  # 特征工程器
    'metadata': {                   # 元信息
        'train_samples': int,
        'test_samples': int,
        'features': int,
        'train_date_range': tuple,
        'test_date_range': tuple,
        'saved_at': str
    }
}
```

---

## [1.1.0] - 2024-12-22

### 🎯 主题：因子库模块

#### ✨ 新增功能

##### 基础算子模块 (`src/factors/operators.py`)
- 实现 **30+ 个 Qlib 风格**的时间序列算子
- **时间平移**：Ref, Delta
- **移动平均**：MA, EMA, WMA, DEMA
- **统计量**：Std, Var, Sum, Min, Max, Mean
- **回归算子**：Slope, Rsquare, Resi
- **相关性算子**：Corr, Cov
- **技术指标**：RSI, MACD, KDJ, ATR, 布林带
- **高级算子**：TSRank, TSMin, TSMax
- 所有算子基于 Pandas 向量化运算，性能优异

##### Alpha158 因子库 (`src/factors/alpha158.py`)
- 实现 **158+ 个经典技术指标**因子
- **KBAR 特征**：OPEN/HIGH/LOW/CLOSE 相关特征（5 个时间窗口）
- **PRICE 特征**：ROC, MA, STD, BETA, RSQR 等
- **VOLUME 特征**：成交量统计和变化
- **技术指标**：RSI, MACD, KDJ, ATR, BOLL
- **价格形态**：振幅、上下影线、实体占比
- **波动率特征**：历史波动率、实现波动率
- **交叉特征**：均线交叉、量价关系
- **量价特征**：量价背离、资金流向

##### Alpha360 因子库 (`src/factors/alpha360.py`)
- 实现 **360+ 个扩展因子**（包含 Alpha158）
- **扩展时间窗口**：3, 7, 14, 21, 40, 80, 120, 180 天
- **高级技术指标**：Williams %R, CCI, Stochastic, ADX
- **波动率扩展**：Parkinson, Garman-Klass, Vol of Vol
- **价格形态识别**：Doji, Hammer, Price Jump
- **量能分析**：OBV, Amihud 非流动性
- **均值回归**：Z-Score, Bias, Bollinger Position
- **多因子组合**：量价综合、动量-波动率比率、趋势质量
- **高阶统计量**：下偏标准差、上行标准差

#### 🎨 功能特性
- ✅ 支持单股票和多股票（MultiIndex）批量计算
- ✅ 自动处理缺失值和无穷大值
- ✅ Parquet 格式因子存储支持
- ✅ 完整的类型提示
- ✅ 向量化计算，性能优异

#### 🧪 测试覆盖
- 新增 `test/test_factors.py`
  - **30 个测试用例**
  - 覆盖所有基础算子
  - 覆盖 Alpha158 和 Alpha360
  - 测试单股票和多股票场景
  - 测试极端值和缺失值处理
- ✅ 所有测试通过（30/30）

#### 💻 使用示例

```python
# 基础算子
from factors import MA, RSI, MACD
ma5 = MA(close, 5)
rsi14 = RSI(close, 14)

# Alpha158 因子库
from factors import calculate_alpha158
factors = calculate_alpha158(df)  # 生成 214 个因子

# Alpha360 因子库
from factors import calculate_alpha360
factors = calculate_alpha360(df, include_alpha158=True)  # 生成 360+ 个因子
```

#### 📚 新增文档
- ✅ `docs/factors_guide.md` - 因子库详细使用指南（472 行）
- ✅ `example_factors.py` - 5 个完整使用示例
- ✅ `docs/QUICKSTART_FACTORS.md` - 因子库快速开始
- ✅ 更新 `README.md` - 添加因子库说明

#### ⚡ 性能基准
- **Alpha158**：~3 秒（730 天数据）
- **Alpha360**：~67 秒（730 天数据）
- 所有算子均使用 Pandas 向量化运算

#### 📦 依赖更新
- 新增 `pytest >= 7.4.0`
- 新增 `pytest-cov >= 4.1.0`

---

## [1.0.0] - 2024-12-20

### 🎯 主题：项目初始化

#### ✨ 初始功能

##### 数据引擎模块 (`src/data_engine/`)
- ✅ 自动市场识别（美股/A股）
- ✅ 美股数据获取（yfinance）
  - 日线 OHLCV 数据
  - 自动时区处理（US/Eastern）
  - 股票分拆和股息调整
- ✅ A股数据获取（AkShare）
  - 日线 OHLCV + 扩展指标
  - 分钟线数据（1/5/15/30/60 分钟）
  - 行业数据（申万分类、概念板块）
  - 微观数据（集合竞价、换手率分位数）
  - 实时行情
- ✅ 统一数据格式（OHLCV）
- ✅ Parquet 格式本地缓存
- ✅ 智能增量更新机制
- ✅ 时区自动处理

##### 模型模块 (`src/models/`)
- ✅ RankLoss 函数实现
  - RankMSE
  - PairwiseRankLoss
  - ListNetLoss
  - BinaryClassificationLoss
- ✅ 特征工程模块
- ✅ 预测模型封装
  - Random Forest
  - Ridge / LASSO
  - Linear Regression
  - GBDT

##### 回测引擎 (`src/backtester/`)
- ✅ 回测核心引擎
- ✅ 交易策略实现
  - WeeklyRotationStrategy（每周轮动）
  - RankingStrategy（排序策略）
- ✅ 性能分析模块
  - 收益指标（总收益、年化收益等）
  - 风险指标（波动率、最大回撤等）
  - 风险调整收益（夏普比率、索提诺比率等）

##### Mag7 策略
- ✅ 完整的 Mag7 每周轮动策略实现
- ✅ 基于 RankMSE 的收益率预测
- ✅ 交易成本和滑点处理
- ✅ 详细的性能报告

#### 📚 文档体系
- ✅ `README.md` - 项目主文档
- ✅ `docs/claude.md` - 开发指南
- ✅ `docs/QUICKSTART.md` - 快速开始
- ✅ `docs/QUICKSTART_MAG7.md` - Mag7 策略快速开始
- ✅ `docs/CN_DATA_MODULE.md` - A股数据模块文档
- ✅ `docs/CN_DATA_QUICKSTART.md` - A股数据快速开始
- ✅ `docs/CN_DATA_IMPLEMENTATION.md` - A股数据实现总结

#### 🧪 测试套件
- ✅ `test/test_data_engine.py` - 数据引擎测试
- ✅ `test/test_cn_fetcher.py` - A股数据获取测试（12 个测试）
- ✅ `test/test_models.py` - 模型测试
- ✅ `test/test_backtester.py` - 回测引擎测试

#### 💻 示例脚本
- ✅ `main_mag7_strategy.py` - Mag7 策略主脚本
- ✅ `example_usage.py` - 基础使用示例
- ✅ `demo_cn_data.py` - A股数据演示（6 个演示）

#### 📦 依赖包
```
pandas >= 2.0.0
numpy >= 1.24.0
yfinance >= 0.2.32
akshare >= 1.12.0
pyarrow >= 14.0.0
scikit-learn >= 1.3.0
scipy >= 1.11.0
```

---

## 版本说明

### 版本号规则
采用语义化版本号（Semantic Versioning）：`主版本号.次版本号.修订号`

- **主版本号**：不兼容的 API 变更
- **次版本号**：向后兼容的功能性新增
- **修订号**：向后兼容的问题修正

### 图标说明
- ✨ 新增功能
- 🔧 功能改进
- 🐛 Bug 修复
- 📚 文档更新
- ⚡ 性能优化
- 🎨 代码重构
- 🧪 测试相关
- 📦 依赖更新
- 🚨 重要变更

---

**最后更新**: 2025-12-23
