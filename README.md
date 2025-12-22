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

### 因子库 (NEW! 🎉)
- ✅ **30+ 基础算子**: Ref, MA, Std, Slope, RSI, MACD 等
- ✅ **Alpha158 因子库**: 158+ 个经典技术指标因子
- ✅ **Alpha360 因子库**: 360+ 个扩展因子
- ✅ **Qlib 风格**: 兼容 Qlib 的表达式语法
- ✅ **向量化计算**: 基于 Pandas 的高性能实现
- ✅ **多股票支持**: 支持单股票和多股票批量计算

## 项目结构

```
quant_project/
├── docs/
│   ├── claude.md         # 开发指南
│   └── factors_guide.md  # 因子库使用指南 (NEW!)
├── src/
│   ├── data_engine/      # 数据获取层
│   │   ├── base.py       # 基础类定义
│   │   ├── us_fetcher.py # 美股数据获取
│   │   ├── cn_fetcher.py # A股数据获取
│   │   └── data_manager.py # 数据管理器
│   ├── factors/          # 因子计算层 (NEW! ✅)
│   │   ├── operators.py  # 基础算子（30+ 个）
│   │   ├── alpha158.py   # Alpha158 因子库
│   │   ├── alpha360.py   # Alpha360 因子库
│   │   └── __init__.py   # 模块导出
│   ├── models/           # 机器学习模型 (待实现)
│   ├── backtester/       # 回测引擎 (待实现)
│   └── utils/            # 工具函数 (待实现)
├── data/                 # 本地数据缓存
├── test/                 # 测试脚本
│   ├── test_data_engine.py
│   └── test_factors.py   # 因子测试 (NEW!)
├── example_factors.py    # 因子使用示例 (NEW!)
└── requirements.txt      # 依赖包
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

更多详细用法请参考：
- 📖 [因子库使用指南](docs/factors_guide.md)
- 💻 [示例代码](example_factors.py)

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

## 下一步计划

- [x] 实现因子计算模块（Qlib 风格算子）✅
  - [x] 30+ 基础算子
  - [x] Alpha158 因子库
  - [x] Alpha360 因子库
- [ ] 实现 Mag7 5日择股策略
  - [ ] 实现 RankMSE 损失函数
  - [ ] 训练预测模型
- [ ] 集成 Backtrader 回测引擎
- [ ] 添加 L2/L3 高频数据支持

## 依赖

- Python 3.10+
- pandas >= 2.0.0
- numpy >= 1.24.0
- yfinance >= 0.2.32
- akshare >= 1.12.0
- pyarrow >= 14.0.0
- pytest >= 7.4.0 (测试)

## License

MIT

