# Quant Project - 跨市场量化分析系统

一个支持 A股 (CN) 与 美股 (US) 的量化框架，具备处理日线数据到高频数据扩展的能力。

## 特性

### 数据引擎
- ✅ **自动市场识别**: 根据股票代码自动识别美股或A股
- ✅ **统一数据接口**: 统一的 OHLCV 数据格式
- ✅ **多数据源支持**: 
  - 美股: yfinance
  - A股: AkShare
- ✅ **数据缓存**: Parquet 格式本地缓存，加速数据访问
- ✅ **时区处理**: 自动处理不同市场的时区（US/Eastern, Asia/Shanghai）
- ✅ **数据集管理** 🆕: 保存/加载处理好的训练测试数据集，节省 60-80% 时间

### 因子库
- ✅ **30+ 基础算子**: Ref, MA, Std, Slope, RSI, MACD 等
- ✅ **Alpha158 因子库**: 158+ 个经典技术指标因子
- ✅ **Alpha360 因子库**: 360+ 个扩展因子
- ✅ **Qlib 风格**: 兼容 Qlib 的表达式语法
- ✅ **向量化计算**: 基于 Pandas 的高性能实现
- ✅ **多股票支持**: 支持单股票和多股票批量计算

### LLM 模块 (NEW! 🎉)
- ✅ **Gemini Deep Research**: 集成 Gemini AI 深度研究能力
- ✅ **自动化报告**: 生成股票、行业深度研究报告
- ✅ **报告管理**: 按日期自动组织和管理研究报告
- ✅ **批量研究**: 支持批量执行研究任务
- ✅ **智能搜索**: 关键词搜索历史报告

## 项目结构

```
quant_project/
├── docs/                           # 📚 文档中心
│   ├── claude.md                   # 开发指南（已优化）
│   ├── CHANGELOG.md                # 版本更新日志
│   ├── QUICKSTART.md               # 项目快速开始
│   ├── QUICKSTART_MAG7.md          # Mag7 策略快速开始
│   ├── QUICKSTART_FACTORS.md       # 因子库快速开始
│   ├── factors_guide.md            # 因子库详细指南
│   ├── DATASET_USAGE.md            # 数据集管理指南
│   ├── CN_DATA_MODULE.md           # A股数据模块文档
│   ├── CN_DATA_QUICKSTART.md       # A股数据快速开始
│   └── CN_DATA_IMPLEMENTATION.md   # A股数据实现总结
├── src/
│   ├── data_engine/                # ✅ 数据获取层
│   │   ├── base.py                 # 基础类定义
│   │   ├── us_fetcher.py           # 美股数据获取（yfinance）
│   │   ├── cn_fetcher.py           # A股数据获取（AkShare）
│   │   └── data_manager.py         # 统一数据管理器
│   ├── factors/                    # ✅ 因子计算层
│   │   ├── operators.py            # 基础算子（30+ 个）
│   │   ├── alpha158.py             # Alpha158 因子库
│   │   ├── alpha360.py             # Alpha360 因子库
│   │   └── __init__.py             # 模块导出
│   ├── models/                     # ✅ 机器学习模型
│   │   ├── rank_loss.py            # RankLoss 函数
│   │   ├── feature_engineering.py  # 特征工程
│   │   ├── predictor.py            # 预测模型
│   │   └── __init__.py
│   ├── backtester/                 # ✅ 回测引擎
│   │   ├── engine.py               # 回测引擎
│   │   ├── strategy.py             # 交易策略
│   │   ├── performance.py          # 性能分析
│   │   └── __init__.py
│   ├── llm/                        # ✅ LLM 模块（NEW! 🎉）
│   │   ├── gemini_client.py        # Gemini API 客户端
│   │   ├── report_manager.py       # 报告管理器
│   │   └── __init__.py
│   └── utils/                      # 🔧 工具函数
├── data/                           # 💾 本地数据缓存
│   ├── cn/                         # A股数据（Parquet）
│   ├── us/                         # 美股数据（Parquet）
│   ├── reports/                    # 📄 LLM 研究报告（NEW! 🎉）
│   └── metadata/                   # 元数据（JSON）
├── test/                           # 🧪 测试套件
│   ├── test_data_engine.py
│   ├── test_cn_fetcher.py
│   ├── test_factors.py
│   ├── test_models.py
│   └── test_backtester.py
├── output/                         # 📊 输出结果
│   ├── dataset.pkl                 # 保存的数据集
│   ├── portfolio.csv               # 投资组合历史
│   ├── trades.csv                  # 交易记录
│   └── model_*.pkl                 # 训练好的模型
├── main_mag7_strategy.py           # 🚀 Mag7 策略主脚本
├── example_*.py                    # 💡 使用示例
├── demo_*.py                       # 🎬 功能演示
├── requirements.txt                # 📦 依赖包
└── README.md                       # 📖 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行测试

```bash
python test_data_engine.py
```

### 3. 基本使用

#### 数据获取

```python
from src.data_engine import DataManager

# 创建数据管理器
dm = DataManager(data_dir="data")

# 获取美股数据（自动识别）
df_aapl = dm.fetch_data("AAPL")  # 默认获取最近一年数据

# 获取A股数据
df_maotai = dm.fetch_data("600519")  # 贵州茅台

# 批量获取 Mag7 数据
mag7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
data = dm.fetch_multiple(mag7)

# 自定义日期范围
from datetime import datetime
df = dm.fetch_data(
    "AAPL",
    start_date="2023-01-01",
    end_date="2024-01-01"
)
```

#### 因子计算 (NEW! 🎉)

```python
from factors import calculate_alpha158, calculate_alpha360
from factors import MA, RSI, MACD  # 基础算子

# 获取数据
df = dm.get_stock_data('AAPL', market='us')

# 方法 1: 使用基础算子
ma5 = MA(df['close'], 5)        # 5日均线
rsi14 = RSI(df['close'], 14)    # RSI指标
macd = MACD(df['close'])         # MACD指标

# 方法 2: 计算 Alpha158 因子（158+ 个因子）
factors_158 = calculate_alpha158(df)
print(f"生成因子数: {len(factors_158.columns)}")

# 方法 3: 计算 Alpha360 因子（360+ 个因子）
factors_360 = calculate_alpha360(df, include_alpha158=True)

# 保存因子
factors_158.to_parquet('data/factors/AAPL_alpha158.parquet')
```

#### 批量计算 Mag7 因子

```python
# 批量计算并保存
for symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']:
    df = dm.get_stock_data(symbol, market='us')
    factors = calculate_alpha158(df)
    factors.to_parquet(f'data/factors/{symbol}_alpha158.parquet')
```

📖 **详细文档**：
- [因子库使用指南](docs/factors_guide.md) - 完整的因子库使用文档
- [因子库快速开始](docs/QUICKSTART_FACTORS.md) - 5 分钟快速上手
- [示例代码](example_factors.py) - 实用代码示例

#### LLM 研究报告生成 (NEW! 🎉)

```python
from src.llm import GeminiDeepResearchClient, ReportManager

# 初始化（需要设置环境变量 GEMINI_API_KEY）
client = GeminiDeepResearchClient()
manager = ReportManager(base_dir='data/reports')

# 执行深度研究
result = client.deep_research(
    query="分析特斯拉 (TSLA) 2024年Q4的财务表现",
    metadata={'ticker': 'TSLA', 'quarter': 'Q4 2024'}
)

# 自动保存报告（按日期组织）
report_path = manager.save_report(
    report_data=result,
    filename='tsla_q4_2024_analysis'
)

# 搜索历史报告
reports = manager.search_reports(keyword='TSLA')

# 批量研究
queries = [
    "分析英伟达 (NVDA) 在AI芯片市场的竞争优势",
    "评估微软 (MSFT) 云计算业务的增长前景"
]
results = client.batch_research(queries)
```

**报告目录结构**：
```
data/reports/
├── 2024-12-25/
│   ├── tsla_q4_2024_analysis.txt        # 报告内容
│   ├── tsla_q4_2024_analysis.json       # 元数据
│   └── tsla_q4_2024_analysis_thinking.txt  # 思考过程
└── ...
```

📖 **详细文档**：
- [LLM 模块使用指南](docs/LLM_MODULE.md) - 完整使用文档
- [示例代码](example_llm.py) - 5 个实用示例

## 数据格式

所有获取的数据都是标准化的 Pandas DataFrame：

```
Index: date (timezone-aware DatetimeIndex)
Columns:
  - open: float64
  - high: float64
  - low: float64
  - close: float64
  - volume: int64/float64
  - market: str ('US' or 'CN')
  - symbol: str
```

## 市场识别规则

- **美股**: 1-5个大写字母（如 AAPL, MSFT, GOOGL）
- **A股**: 6位数字（如 600519, 000001）

### 模型和回测 (NEW! 🎉)

#### Mag7 每周轮动策略

完整实现了基于机器学习的股票择时策略：

```bash
# 运行 Mag7 策略
python main_mag7_strategy.py
```

**特性**：
- ✅ **RankLoss 函数**: RankMSE, PairwiseRank, ListNet
- ✅ **多种模型**: Random Forest, Ridge, LASSO, GBDT
- ✅ **特征工程**: 基于 Qlib 算子的量价因子
- ✅ **完整回测**: 包含交易成本、持仓管理、性能分析
- ✅ **每周调仓**: 每周一选择预测收益率最高的股票

**新功能：数据集保存和加载** 🆕
```bash
# 首次运行：保存数据集
python main_mag7_strategy.py --save_dataset --start_date 2022-01-01 --end_date 2024-12-31

# 后续快速运行：加载数据集（节省 60-80% 时间）
python main_mag7_strategy.py --load_dataset output/dataset.pkl --model_type gbdt
```

📖 **详细文档**：
- [Mag7 策略快速开始](docs/QUICKSTART_MAG7.md) - 完整使用指南
- [数据集管理指南](docs/DATASET_USAGE.md) - 数据集保存和加载 🆕
- [开发指南](docs/claude.md) - 架构设计和最佳实践

## 快速示例

### 完整策略示例

```python
from src.data_engine import DataManager
from src.models import FeatureEngineer, StockPredictor
from src.backtester import BacktestEngine, WeeklyRotationStrategy

# 1. 获取数据
dm = DataManager()
mag7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
data_dict = {symbol: dm.fetch_data(symbol) for symbol in mag7}

# 2. 特征工程
fe = FeatureEngineer()
X, y, dates, symbols = fe.prepare_dataset(data_dict, forward_days=5)

# 3. 训练模型
predictor = StockPredictor(model_type='random_forest', loss_type='rank_mse')
predictor.fit(X, y)

# 4. 回测
engine = BacktestEngine(initial_capital=100000)
strategy = WeeklyRotationStrategy(predictor, fe, top_k=1)

# 运行回测...
```

## 📖 文档导航

### 🚀 快速开始
- [项目快速开始](docs/QUICKSTART.md) - 5 分钟上手指南
- [Mag7 策略快速开始](docs/QUICKSTART_MAG7.md) - 美股轮动策略
- [因子库快速开始](docs/QUICKSTART_FACTORS.md) - 因子计算入门

### 📚 详细文档
- [开发指南](docs/claude.md) - 完整的开发指南和架构设计
- [因子库使用指南](docs/factors_guide.md) - 因子库详细文档
- [数据集管理指南](docs/DATASET_USAGE.md) - 数据集保存和加载
- [LLM 模块使用指南](docs/LLM_MODULE.md) - LLM 研究报告生成 (NEW! 🎉)

### 🇨🇳 A股数据模块
- [A股数据快速开始](docs/CN_DATA_QUICKSTART.md) - 5 分钟上手
- [A股数据完整文档](docs/CN_DATA_MODULE.md) - 完整功能说明
- [A股数据实现总结](docs/CN_DATA_IMPLEMENTATION.md) - 技术实现细节

### 📋 其他
- [更新日志](docs/CHANGELOG.md) - 版本历史和变更记录

---

## 🗺️ 开发路线图

### 已完成 ✅
- [x] 数据引擎（美股 + A股）
- [x] 因子计算库（Alpha158 + Alpha360）
- [x] RankLoss 函数
- [x] Mag7 每周轮动策略
- [x] 回测引擎
- [x] 数据集管理功能
- [x] LLM 模块（Gemini Deep Research）🎉
- [x] 完整测试套件
- [x] 文档体系

### 进行中 🔧
- [ ] 可视化模块
  - [ ] 收益曲线图
  - [ ] 持仓变化图
  - [ ] 因子分析图
  - [ ] 回测报告 HTML

### 计划中 📋
- [ ] A股次日高点预测策略
- [ ] 实时交易接口对接
- [ ] 策略组合管理
- [ ] 风险控制模块

## 依赖

- Python 3.10+
- pandas >= 2.0.0
- numpy >= 1.24.0
- yfinance >= 0.2.32
- akshare >= 1.12.0
- pyarrow >= 14.0.0
- scikit-learn >= 1.3.0 (机器学习)
- scipy >= 1.11.0 (科学计算)
- requests >= 2.31.0 (LLM API 调用)
- pytest >= 7.4.0 (测试)

## License

MIT

