# 因子库快速开始 🚀

## 10 分钟上手因子计算

### 1. 基础算子（1 分钟）

```python
from factors import MA, Std, RSI, MACD
from data_engine.data_manager import DataManager

# 获取数据
dm = DataManager()
df = dm.get_stock_data('AAPL', market='us')

# 计算指标
close = df['close']
ma5 = MA(close, 5)           # 5日均线
std20 = Std(close, 20)       # 20日标准差
rsi14 = RSI(close, 14)       # RSI指标
macd = MACD(close)           # MACD指标

print(f"最新MA5: {ma5.iloc[-1]:.2f}")
print(f"最新RSI: {rsi14.iloc[-1]:.2f}")
```

### 2. Alpha158 因子库（2 分钟）

```python
from factors import calculate_alpha158

# 一行代码计算 158+ 个因子
factors = calculate_alpha158(df)

print(f"生成因子数量: {len(factors.columns)}")
print(f"数据行数: {len(factors)}")

# 保存因子
factors.to_parquet('AAPL_alpha158.parquet')
```

### 3. Alpha360 因子库（3 分钟）

```python
from factors import calculate_alpha360

# 计算 360+ 个扩展因子
factors = calculate_alpha360(df, include_alpha158=True)

print(f"生成因子数量: {len(factors.columns)}")

# 查看部分因子
print(factors[['MA5_MA20', 'RSI_14', 'MACD', 'VOLATILITY_20']].tail())
```

### 4. 批量计算 Mag7（4 分钟）

```python
mag7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

for symbol in mag7:
    print(f"处理 {symbol}...")
    df = dm.get_stock_data(symbol, market='us')
    factors = calculate_alpha158(df)
    factors.to_parquet(f'data/factors/{symbol}_alpha158.parquet')
    print(f"  ✓ 生成 {len(factors.columns)} 个因子")
```

### 5. 因子分析示例（可选）

```python
# 计算因子与未来收益率的相关性
df['return_5d'] = df['close'].pct_change(5).shift(-5)
data = pd.concat([factors, df[['return_5d']]], axis=1).dropna()

# IC 分析
correlations = data.corr()['return_5d'].sort_values(ascending=False)
print("\n相关性最高的 10 个因子:")
print(correlations.head(10))

# 因子分组测试
factor_name = 'MA5_MA20'
data['quintile'] = pd.qcut(data[factor_name], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
group_returns = data.groupby('quintile')['return_5d'].mean()
print(f"\n{factor_name} 分组收益:")
print(group_returns)
```

---

## 可用的算子

### 基础算子
- **时间序列**: Ref, Delta, Returns, LogReturns
- **统计**: MA, EMA, WMA, Std, Var, Sum, Min, Max, Skewness, Kurtosis
- **回归**: Slope, Rsquare, Resi
- **相关性**: Corr, Cov
- **排名**: Rank, TSRank, TSMin, TSMax

### 技术指标
- **趋势**: MA, EMA, WMA, MACD
- **动量**: RSI, ROC, MOM
- **波动**: ATR, Bollinger Bands, Volatility
- **其他**: KDJ, Stochastic, Williams %R, CCI, ADX

---

## 因子库对比

| 特性 | Alpha158 | Alpha360 |
|------|----------|----------|
| 因子数量 | 158+ | 360+ |
| 计算时间 | ~3秒 | ~67秒 |
| 时间窗口 | 5,10,20,30,60 | +3,7,14,21,40,80,120,180 |
| 技术指标 | 基础 | 基础 + 高级 |
| 波动率特征 | 标准 | 标准 + Parkinson + GK |
| 适用场景 | 日常使用、快速迭代 | 深度分析、模型训练 |

---

## 测试

```bash
# 运行所有因子测试
pytest test/test_factors.py -v

# 运行特定测试
pytest test/test_factors.py::TestAlpha158 -v

# 查看测试覆盖率
pytest test/test_factors.py --cov=src/factors --cov-report=html
```

---

## 完整示例

运行完整示例：

```bash
python example_factors.py
```

示例包含：
1. 基础算子使用
2. Alpha158 因子计算
3. Alpha360 因子计算
4. Mag7 批量处理
5. 因子有效性分析

---

## 更多信息

- 📖 [详细使用指南](docs/factors_guide.md)
- 💻 [完整示例代码](example_factors.py)
- 🧪 [测试代码](test/test_factors.py)
- 📝 [更新日志](CHANGELOG.md)

---

## 常见问题

**Q: 因子计算很慢怎么办？**
- 使用 Alpha158 而不是 Alpha360
- 只计算需要的时间窗口
- 使用多进程并行处理

**Q: 如何添加自定义因子？**
- 直接使用基础算子组合
- 或扩展 Alpha158/Alpha360 类

**Q: 因子如何与策略结合？**
- 因子 → 特征工程 → 模型训练 → 策略回测
- 下一步将实现完整的策略回测框架

---

🎉 **恭喜！你已经掌握了因子库的基础用法！**

开始构建你的量化策略吧！

