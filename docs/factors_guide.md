# 因子库使用指南

## 概述

本项目实现了 Qlib 风格的因子计算库，包含：

1. **基础算子模块** (`operators.py`)：提供了 30+ 个通用的时间序列算子
2. **Alpha158 因子库** (`alpha158.py`)：包含 158+ 个经典技术指标因子
3. **Alpha360 因子库** (`alpha360.py`)：包含 360+ 个扩展因子（包括 Alpha158）

所有算子和因子计算均使用 Pandas 向量化运算，性能优异。

---

## 快速开始

### 1. 导入模块

```python
from factors import (
    # 基础算子
    Ref, MA, Std, Slope, RSI, MACD,
    # 因子库
    Alpha158, Alpha360,
    calculate_alpha158, calculate_alpha360
)
```

### 2. 准备数据

因子计算需要 OHLCV 格式的 DataFrame：

```python
# DataFrame 需要包含以下列
required_columns = ['open', 'high', 'low', 'close', 'volume']

# 示例：从数据管理器加载
from data_engine.data_manager import DataManager
dm = DataManager()
df = dm.get_stock_data('AAPL', market='us')
```

### 3. 计算因子

```python
# 方法 1: 使用便捷函数
factors = calculate_alpha158(df)

# 方法 2: 使用类
alpha158 = Alpha158()
factors = alpha158.calculate(df)

# 获取因子名称
factor_names = alpha158.get_factor_names()
```

---

## 基础算子详解

### 时间序列算子

| 算子 | 说明 | 示例 |
|------|------|------|
| `Ref(series, d)` | 引用 d 天前的值 | `Ref(close, 5)` |
| `Delta(series, d)` | 当前值与 d 天前的差值 | `Delta(close, 1)` |
| `Returns(series, d)` | d 期收益率 | `Returns(close, 1)` |
| `LogReturns(series, d)` | d 期对数收益率 | `LogReturns(close, 1)` |

### 统计算子

| 算子 | 说明 | 示例 |
|------|------|------|
| `Mean(series, d)` / `MA(series, d)` | d 天移动平均 | `MA(close, 5)` |
| `Std(series, d)` | d 天标准差 | `Std(close, 20)` |
| `Var(series, d)` | d 天方差 | `Var(close, 20)` |
| `Sum(series, d)` | d 天累加和 | `Sum(volume, 5)` |
| `Min(series, d)` | d 天最小值 | `Min(close, 20)` |
| `Max(series, d)` | d 天最大值 | `Max(close, 20)` |
| `Skewness(series, d)` | d 天偏度 | `Skewness(close, 20)` |
| `Kurtosis(series, d)` | d 天峰度 | `Kurtosis(close, 20)` |

### 回归算子

| 算子 | 说明 | 示例 |
|------|------|------|
| `Slope(series, d)` | d 天线性回归斜率 | `Slope(close, 10)` |
| `Rsquare(series, d)` | d 天线性回归 R² | `Rsquare(close, 10)` |
| `Resi(series, d)` | d 天线性回归残差 | `Resi(close, 10)` |

### 相关性算子

| 算子 | 说明 | 示例 |
|------|------|------|
| `Corr(s1, s2, d)` | d 天相关系数 | `Corr(close, volume, 20)` |
| `Cov(s1, s2, d)` | d 天协方差 | `Cov(close, volume, 20)` |

### 移动平均算子

| 算子 | 说明 | 示例 |
|------|------|------|
| `MA(series, d)` | 简单移动平均 | `MA(close, 5)` |
| `EMA(series, d)` | 指数移动平均 | `EMA(close, 12)` |
| `WMA(series, d)` | 加权移动平均 | `WMA(close, 10)` |

### 技术指标

| 算子 | 说明 | 示例 |
|------|------|------|
| `RSI(series, d)` | 相对强弱指标 | `RSI(close, 14)` |
| `MACD(series, f, s, sig)` | MACD 指标 | `MACD(close, 12, 26, 9)` |
| `KDJ(high, low, close, n, m1, m2)` | KDJ 指标 | `KDJ(high, low, close, 9, 3, 3)` |
| `ATR(high, low, close, d)` | 平均真实波幅 | `ATR(high, low, close, 14)` |
| `BBANDS_UPPER(series, d, n_std)` | 布林带上轨 | `BBANDS_UPPER(close, 20, 2)` |
| `BBANDS_LOWER(series, d, n_std)` | 布林带下轨 | `BBANDS_LOWER(close, 20, 2)` |

### 排名算子

| 算子 | 说明 | 示例 |
|------|------|------|
| `Rank(series)` | 横截面排名（0-1） | `Rank(close)` |
| `TSRank(series, d)` | 时间序列排名 | `TSRank(close, 10)` |
| `TSMin(series, d)` | 最小值出现位置 | `TSMin(close, 20)` |
| `TSMax(series, d)` | 最大值出现位置 | `TSMax(close, 20)` |

---

## Alpha158 因子库

Alpha158 包含以下几类因子：

### 1. KBAR 特征（价格形态）

- `OPEN_*`: 开盘价相关特征（相对值、均值、标准差）
- `HIGH_*`: 最高价相关特征
- `LOW_*`: 最低价相关特征
- `CLOSE_*`: 收盘价相关特征

时间窗口：5, 10, 20, 30, 60 天

### 2. PRICE 特征（价格动量）

- `ROC_*`: 收益率
- `MA_*`: 移动平均
- `STD_*`: 标准差
- `SKEW_*`: 偏度
- `KURT_*`: 峰度
- `MAX_*` / `MIN_*`: 最大最小值
- `CORR_*`: 量价相关性
- `BETA_*`: 线性回归斜率
- `RSQR_*`: R²
- `RESI_*`: 残差
- `TSRANK_*`: 时间序列排名
- `QTLU_*` / `QTLD_*`: 分位数

### 3. VOLUME 特征

- `VOLUME_mean_*`: 成交量均值
- `VOLUME_std_*`: 成交量标准差
- `VOLUME_delta_*`: 成交量变化
- `VR_*`: 成交量比率

### 4. 技术指标特征

- `RSI_*`: RSI 指标
- `MACD`, `MACD_signal`, `MACD_hist`: MACD 指标
- `KDJ_K`, `KDJ_D`, `KDJ_J`: KDJ 指标
- `ATR_*`: ATR 指标
- `BOLL_*`: 布林带指标

### 5. 价格形态特征

- `AMP_*`: 振幅
- `UPPER_SHADOW`: 上影线
- `LOWER_SHADOW`: 下影线
- `BODY_RATIO`: 实体占比

### 6. 波动率特征

- `VOLATILITY_*`: 历史波动率
- `REALIZED_VOL_*`: 实现波动率

### 7. 交叉特征

- `MA5_MA10`, `MA5_MA20`, etc.: 均线交叉
- `CLOSE_MA5`, `CLOSE_MA10`, etc.: 价格与均线的关系

### 8. 量价特征

- `PV_DIVERGENCE_*`: 量价背离
- `PV_CORR_*`: 量价相关性
- `MFI_*`: 资金流向指标

### 使用示例

```python
from factors import calculate_alpha158

# 计算 Alpha158 因子
factors = calculate_alpha158(df)

# 查看因子数量
print(f"因子数量: {len(factors.columns)}")

# 查看前几个因子
print(factors.iloc[:5, :10])

# 保存因子
factors.to_parquet('factors_alpha158.parquet')
```

---

## Alpha360 因子库

Alpha360 是 Alpha158 的扩展版本，包含以下额外特征：

### 扩展时间窗口

除了 Alpha158 的窗口，还增加了：3, 7, 14, 21, 40, 80, 120, 180 天

### 高级技术指标

- `WILLR_*`: Williams %R
- `CCI_*`: 商品通道指数
- `STOCH_K_*`, `STOCH_D_*`: 随机震荡指标
- `ADX_*`: 平均趋向指数
- `PLUS_DI_*`, `MINUS_DI_*`: 方向指标

### 波动率扩展

- `HV_ext_*`: 历史波动率
- `PARKINSON_VOL_*`: Parkinson 波动率
- `GK_VOL_*`: Garman-Klass 波动率
- `VOL_OF_VOL_*`: 波动率的波动率

### 价格形态识别

- `DOJI_PATTERN`: 十字星形态
- `HAMMER_PATTERN`: 锤子线形态
- `PRICE_JUMP_*`: 价格跳跃

### 量能分析

- `OBV_*`: 能量潮指标
- `AMIHUD_*`: Amihud 非流动性指标

### 均值回归指标

- `ZSCORE_*`: Z 分数
- `BIAS_*`: 乖离率
- `BOLL_POS_*`: 布林带位置

### 多因子组合

- `PV_COMPOSITE_*`: 量价综合指标
- `MOM_VOL_RATIO_*`: 动量-波动率比率
- `TREND_QUALITY_*`: 趋势质量

### 高阶统计量

- `DOWNSIDE_STD_*`: 下偏标准差（下行风险）
- `UPSIDE_STD_*`: 上行标准差

### 使用示例

```python
from factors import calculate_alpha360

# 计算 Alpha360（包含 Alpha158）
factors = calculate_alpha360(df, include_alpha158=True)

# 只计算 Alpha360 扩展因子
factors_ext = calculate_alpha360(df, include_alpha158=False)

# 查看因子数量
print(f"因子数量: {len(factors.columns)}")
```

---

## 批量处理示例

### 处理多只股票

```python
from factors import calculate_alpha158
from data_engine.data_manager import DataManager

mag7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
dm = DataManager()

all_factors = {}
for symbol in mag7:
    df = dm.get_stock_data(symbol, market='us')
    if df is not None:
        factors = calculate_alpha158(df)
        all_factors[symbol] = factors
        
        # 保存
        factors.to_parquet(f'data/factors/{symbol}_alpha158.parquet')
```

### 多股票 DataFrame（MultiIndex）

```python
# 如果你有多股票的 DataFrame（MultiIndex: [symbol, date]）
# 因子库会自动按股票分组计算

multi_stock_df = pd.concat([
    df1.assign(symbol='AAPL'),
    df2.assign(symbol='MSFT'),
]).set_index('symbol', append=True).swaplevel()

# 自动处理多股票
factors = calculate_alpha158(multi_stock_df)
```

---

## 因子分析示例

### 1. 计算因子与收益率的相关性（IC）

```python
# 计算因子
factors = calculate_alpha158(df)

# 计算未来收益率
df['return_5d'] = df['close'].pct_change(5).shift(-5)

# 合并
data = pd.concat([factors, df[['return_5d']]], axis=1).dropna()

# 计算相关性
correlations = data.corr()['return_5d'].sort_values(ascending=False)
print("相关性最高的因子:", correlations.head(20))
```

### 2. 因子分组测试

```python
# 选择一个因子
factor_name = 'MA5_MA20'

# 分成 5 组
data['quintile'] = pd.qcut(data[factor_name], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# 计算各组平均收益率
group_returns = data.groupby('quintile')['return_5d'].mean()
print(group_returns)
```

### 3. 因子去极值和标准化

```python
def winsorize(series, lower=0.01, upper=0.99):
    """去极值"""
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower_bound, upper_bound)

def standardize(series):
    """标准化"""
    return (series - series.mean()) / series.std()

# 应用到所有因子
for col in factors.columns:
    factors[col] = standardize(winsorize(factors[col]))
```

---

## 性能优化建议

### 1. 使用 Parquet 格式存储

```python
# 保存
factors.to_parquet('factors.parquet')

# 读取（速度远快于 CSV）
factors = pd.read_parquet('factors.parquet')
```

### 2. 避免重复计算

```python
# 如果需要多次使用，先实例化
alpha158 = Alpha158()

# 批量处理时复用实例
for symbol in symbols:
    factors = alpha158.calculate(data[symbol])
```

### 3. 处理大数据集

```python
# 对于超大数据集，可以分批处理
chunk_size = 10000
for chunk in pd.read_parquet('large_data.parquet', chunksize=chunk_size):
    factors = calculate_alpha158(chunk)
    # 处理或保存...
```

---

## 注意事项

1. **数据质量**：确保输入数据没有异常值，high >= low，high >= open/close，low <= open/close
2. **缺失值**：因子计算会自动处理缺失值，但建议事先清洗数据
3. **计算时间**：Alpha360 包含大量因子，计算时间较长，建议先使用 Alpha158
4. **内存占用**：大量股票 × 大量因子会占用较多内存，注意监控
5. **因子选择**：不是所有因子都有效，建议进行因子筛选和组合

---

## 常见问题

### Q: 如何添加自定义因子？

A: 可以直接使用基础算子组合，或者扩展 Alpha158/Alpha360 类：

```python
from factors import Alpha158

class MyAlpha(Alpha158):
    def _calculate_single_stock(self, df):
        result = super()._calculate_single_stock(df)
        
        # 添加自定义因子
        close = df['close']
        result['MY_FACTOR'] = close / MA(close, 10) - 1
        
        return result
```

### Q: 因子计算很慢怎么办？

A: 
1. 使用 Alpha158 而不是 Alpha360
2. 减少时间窗口
3. 只计算需要的因子
4. 使用多进程并行处理多只股票

### Q: 如何处理未来函数？

A: 所有算子都是基于历史数据计算的，不会使用未来信息。但要注意：
- 使用 `.shift(-n)` 会引入未来信息
- 计算收益率时注意时间对齐

---

## 参考资料

- Qlib 官方文档: https://qlib.readthedocs.io/
- Alpha158 论文: https://arxiv.org/abs/2010.15458
- 技术指标详解: https://www.investopedia.com/

---

## 更新日志

### v1.0.0 (2024-12-22)
- ✅ 实现基础算子 30+ 个
- ✅ 实现 Alpha158 因子库
- ✅ 实现 Alpha360 因子库
- ✅ 支持单股票和多股票计算
- ✅ 完整的单元测试覆盖

