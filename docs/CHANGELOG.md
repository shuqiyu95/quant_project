# 更新日志

本文档记录项目的重要变更和版本历史。

---

## [1.4.0] - 2025-12-25

### 🎯 主题：LLM 模块 - Gemini Deep Research 集成

#### ✨ 新增功能

##### LLM 模块 (`src/llm/`)
- **Gemini Deep Research API 客户端** (`gemini_client.py`)
  - 支持 Gemini Deep Research API 调用
  - 自动重试和错误处理机制
  - 批量研究支持
  - 完整的日志记录
  - 支持自定义模型参数（temperature, max_tokens 等）

- **报告管理器** (`report_manager.py`)
  - 按日期自动组织报告（`data/reports/YYYY-MM-DD/`）
  - 保存报告内容、元数据和思考过程
  - 报告检索和加载功能
  - 关键词搜索（支持元数据和内容搜索）
  - 报告删除和管理功能

##### 示例和文档
- ✅ `example_llm.py` - 5 个完整使用示例
  - 基础研究示例
  - 批量研究示例
  - 报告管理示例
  - 加载报告示例
  - 自定义日期保存示例

- ✅ `docs/LLM_MODULE.md` - LLM 模块详细文档
  - 快速开始指南
  - API 参考
  - 使用场景示例
  - 与量化项目集成
  - 最佳实践

#### 🧪 测试覆盖
- 新增 `test/test_llm.py`
  - **40+ 个测试用例**
  - 覆盖 GeminiDeepResearchClient 所有功能
  - 覆盖 ReportManager 所有功能
  - Mock API 调用测试
  - 错误处理测试

#### 💻 使用示例

```python
from src.llm import GeminiDeepResearchClient, ReportManager

# 初始化
client = GeminiDeepResearchClient()
manager = ReportManager(base_dir='data/reports')

# 执行研究
result = client.deep_research(
    query="分析特斯拉 (TSLA) 2024年Q4的财务表现"
)

# 保存报告（自动按日期组织）
report_path = manager.save_report(
    report_data=result,
    filename='tsla_q4_2024_analysis'
)
```

#### 🎨 功能特性
- ✅ 环境变量或代码配置 API Key
- ✅ 自动重试机制（最多 3 次）
- ✅ 批量请求支持（自动延迟避免限流）
- ✅ 报告按日期自动组织
- ✅ 完整的元数据管理
- ✅ 支持搜索和检索
- ✅ 思考过程单独保存

#### 📊 应用场景
- 股票深度研究（基本面分析、技术分析）
- 行业研究（竞争格局、趋势分析）
- 定期研究报告（每日市场分析）
- 批量竞品分析
- 与量化分析结合（数据驱动的研究报告）

#### 📦 依赖更新
- 新增 `requests >= 2.31.0`

#### 📁 目录结构

```
data/reports/
├── 2024-12-25/
│   ├── tsla_q4_2024_analysis.txt          # 报告内容
│   ├── tsla_q4_2024_analysis.json         # 元数据
│   └── tsla_q4_2024_analysis_thinking.txt # 思考过程
└── 2024-12-24/
    └── ...
```

---

## [1.3.0] - 2025-12-25

### 🎯 主题：A股次日高点预测策略

#### ✨ 新增功能

##### A股次日高点预测完整策略
- **核心策略文件** (`cn_intraday_high_strategy.py`)
  - 预测次日开盘后30分钟内的最高涨幅
  - 多分类模型：将涨幅分为5个桶 (`<-3%`, `-3%~0%`, `0%~3%`, `3%~6%`, `>6%`)
  - 完整的数据准备、特征工程、模型训练、回测流程

- **A股特色因子库**
  - **量能因子**：换手率分位数、量比、成交量变化率
  - **竞价因子**：竞价量占比、竞价涨幅（需分钟线数据）
  - **情绪因子**：连续涨跌天数、创新高/新低标记
  - **波动率因子**：多周期历史波动率、上下行波动率
  - 共80+个A股特色因子，结合Alpha158形成完整特征集

- **多分类预测模型**
  - 支持随机森林（Random Forest）
  - 支持梯度提升树（GBDT）
  - 自动处理样本不平衡（class_weight='balanced'）
  - 特征标准化和重要性分析

- **回测引擎**
  - A股交易规则：T+1、佣金0.03%、印花税0.05%
  - 策略逻辑：预测涨幅>=3%时开盘买入，30分钟后卖出
  - 完整性能指标：收益率、夏普比率、最大回撤、胜率等

##### 金风科技演示脚本 (`demo_cn_intraday_high.py`)
- 完整的端到端演示流程
- 使用金风科技(002202)作为示例
- 自动生成交易记录和投资组合历史
- 详细的策略评估报告

##### 详细文档
- **策略文档** (`docs/CN_INTRADAY_HIGH_STRATEGY.md`)
  - 策略逻辑详解
  - 特征工程说明
  - 使用指南
  - 性能优化建议
  - 风险提示

#### 🎨 优化改进

- 兼容无分钟线数据的情况（用次日开盘价近似）
- 自动识别市场类型（A股/美股）
- 灵活的模型参数配置
- 完善的错误处理和用户提示

#### 💻 使用示例

```bash
# 运行金风科技演示
python demo_cn_intraday_high.py

# 自定义股票
from cn_intraday_high_strategy import CNIntradayHighPredictor
predictor = CNIntradayHighPredictor(model_type='random_forest')
X, y, dates = predictor.prepare_dataset('600519', '2022-01-01', '2024-12-20')
predictor.train(X, y)
```

#### 📊 输出文件

- `output/cn_intraday_high_002202.pkl`: 训练好的模型
- `output/trades_002202.csv`: 交易记录
- `output/portfolio_002202.csv`: 投资组合历史

#### ⚠️ 已知限制

1. **分钟线数据**：AkShare的分钟线数据可能不稳定，建议使用付费数据源
2. **T+1限制**：策略假设次日开盘买入、30分钟后卖出，需要融券工具
3. **样本不平衡**：高涨幅样本较少，模型可能倾向预测中等涨幅

#### 🔜 下一步计划

- 优化样本不平衡问题（SMOTE等方法）
- 集成更多模型（LightGBM、XGBoost）
- 实现多股票池选股
- 添加可视化功能

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
