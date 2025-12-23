# 更新日志

## [1.2.0] - 2024-12-22

### 新增功能 ✨

#### 数据集管理系统 🆕
- **数据集保存/加载功能** (`main_mag7_strategy.py`)
  - ✅ 保存处理好的训练/测试数据集到 `.pkl` 文件
  - ✅ 快速加载已保存的数据集，跳过数据获取和特征工程
  - ✅ 节省 60-80% 的运行时间
  - ✅ 数据集包含完整元信息（样本数、特征数、日期范围等）
  - ✅ 方便测试不同模型和参数组合

#### 日期参数控制 🆕
- **灵活的日期范围控制**
  - ✅ `--start_date`: 指定数据开始日期 (YYYY-MM-DD)
  - ✅ `--end_date`: 指定数据结束日期 (YYYY-MM-DD)
  - ✅ 支持自定义时间范围进行回测
  - ✅ 向后兼容 `--years` 参数

### 新增参数
- `--start_date`: 数据开始日期 (YYYY-MM-DD)
- `--end_date`: 数据结束日期 (YYYY-MM-DD)
- `--save_dataset`: 保存处理好的数据集
- `--load_dataset`: 加载已保存的数据集路径
- `--dataset_path`: 数据集保存/加载路径（默认 output/dataset.pkl）

### 使用示例
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

### 文档 📚
- ✅ 新增 `DATASET_USAGE.md` - 数据集管理详细指南
- ✅ 新增 `test_dataset_save_load.py` - 数据集功能测试脚本
- ✅ 更新 `QUICKSTART_MAG7.md` - 添加数据集使用方法
- ✅ 更新 `README.md` - 添加新功能说明

### 性能优化 ⚡
- 首次运行（数据获取 + 特征工程 + 训练 + 回测）：~2-3 分钟
- 后续运行（加载数据集 + 训练 + 回测）：~30 秒
- **时间节省：60-80%**

---

## [1.1.0] - 2024-12-22

### 新增功能 ✨

#### 因子库模块
- **基础算子模块** (`src/factors/operators.py`)
  - ✅ 实现 30+ 个 Qlib 风格的时间序列算子
  - ✅ 包括：Ref, Delta, MA, EMA, WMA, Std, Var, Sum, Min, Max
  - ✅ 回归算子：Slope, Rsquare, Resi
  - ✅ 相关性算子：Corr, Cov
  - ✅ 技术指标：RSI, MACD, KDJ, ATR, 布林带
  - ✅ 高级算子：TSRank, TSMin, TSMax
  - ✅ 所有算子基于 Pandas 向量化运算，性能优异

- **Alpha158 因子库** (`src/factors/alpha158.py`)
  - ✅ 实现 158+ 个经典技术指标因子
  - ✅ KBAR 特征：OPEN/HIGH/LOW/CLOSE 相关特征（5 个时间窗口）
  - ✅ PRICE 特征：ROC, MA, STD, BETA, RSQR 等
  - ✅ VOLUME 特征：成交量统计和变化
  - ✅ 技术指标：RSI, MACD, KDJ, ATR, BOLL
  - ✅ 价格形态：振幅、上下影线、实体占比
  - ✅ 波动率特征：历史波动率、实现波动率
  - ✅ 交叉特征：均线交叉、量价关系
  - ✅ 量价特征：量价背离、资金流向

- **Alpha360 因子库** (`src/factors/alpha360.py`)
  - ✅ 实现 360+ 个扩展因子（包含 Alpha158）
  - ✅ 扩展时间窗口：3, 7, 14, 21, 40, 80, 120, 180 天
  - ✅ 高级技术指标：Williams %R, CCI, Stochastic, ADX
  - ✅ 波动率扩展：Parkinson, Garman-Klass, Vol of Vol
  - ✅ 价格形态识别：Doji, Hammer, Price Jump
  - ✅ 量能分析：OBV, Amihud 非流动性
  - ✅ 均值回归：Z-Score, Bias, Bollinger Position
  - ✅ 多因子组合：量价综合、动量-波动率比率、趋势质量
  - ✅ 高阶统计量：下偏标准差、上行标准差

#### 功能特性
- ✅ 支持单股票和多股票（MultiIndex）批量计算
- ✅ 自动处理缺失值和无穷大值
- ✅ Parquet 格式因子存储支持
- ✅ 完整的类型提示
- ✅ 向量化计算，性能优异

### 测试 🧪
- ✅ 新增 `test/test_factors.py`
  - 30 个测试用例
  - 覆盖所有基础算子
  - 覆盖 Alpha158 和 Alpha360
  - 测试单股票和多股票场景
  - 测试极端值和缺失值处理
- ✅ 所有测试通过（30/30）

### 文档 📚
- ✅ 新增 `docs/factors_guide.md` - 因子库详细使用指南
- ✅ 新增 `example_factors.py` - 5 个完整使用示例
- ✅ 更新 `README.md` - 添加因子库说明
- ✅ 更新 `requirements.txt` - 添加 pytest

### 示例代码
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

### 性能
- Alpha158：~3 秒（730 天数据）
- Alpha360：~67 秒（730 天数据）
- 所有算子均使用 Pandas 向量化运算

### 依赖更新
- 新增 `pytest >= 7.4.0`
- 新增 `pytest-cov >= 4.1.0`

---

## [1.0.0] - 2024-12-20

### 初始版本
- ✅ 实现数据引擎模块
- ✅ 支持美股（yfinance）和 A股（AkShare）
- ✅ 自动市场识别
- ✅ Parquet 格式数据缓存
- ✅ 时区自动处理

