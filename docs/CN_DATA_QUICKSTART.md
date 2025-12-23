# A股数据获取模块 - 快速开始指南

## 🎯 5分钟上手

### 1. 最简单的例子

```python
from src.data_engine.data_manager import DataManager
from datetime import datetime, timedelta

# 创建数据管理器
manager = DataManager(data_dir="data")

# 获取贵州茅台最近60天数据
df = manager.fetch_data("600519")  # 默认获取一年数据

print(f"获取了 {len(df)} 条记录")
print(df.tail())  # 查看最近5天
```

### 2. 指定日期范围

```python
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

df = manager.fetch_data(
    symbol="600519",
    start_date=start_date,
    end_date=end_date
)
```

### 3. 批量获取多只股票

```python
# A股：茅台、招行、五粮液
symbols = ["600519", "600036", "000858"]
results = manager.fetch_multiple(symbols)

for symbol, df in results.items():
    if df is not None:
        print(f"{symbol}: {len(df)} 条记录, 最新价 {df['close'].iloc[-1]:.2f}")
```

### 4. 增量更新（⭐推荐）

```python
# 第一次获取
df = manager.fetch_data("600519", use_cache=False)
print(f"初始数据: {len(df)} 条")

# 第二天收盘后，只获取新数据
df_updated = manager.fetch_data_incremental("600519")
print(f"更新后: {len(df_updated)} 条")
# 自动只下载新增的数据，快速且节省流量
```

## 📊 常用数据类型

### 日线数据（最常用）

```python
from src.data_engine.cn_fetcher import CNFetcher

fetcher = CNFetcher()
df = fetcher.fetch_daily_data("600519", start_date, end_date)

# 可用字段：
# - open, high, low, close, volume（基础OHLCV）
# - amount（成交额）
# - turnover（换手率）
# - pct_change（涨跌幅）
# - amplitude（振幅）
```

### 实时行情

```python
# 获取多只股票的实时价格
symbols = ["600519", "600036", "000858"]
df_realtime = fetcher.get_realtime_quotes(symbols)

print(df_realtime[['symbol', 'name', 'price', 'pct_change']])
# 输出：
# symbol  name    price  pct_change
# 600519  贵州茅台  1406.88  -0.10
# 600036  招商银行  41.89    0.29
```

### 行业信息

```python
industry_info = fetcher.fetch_industry_data("600519")
print(f"行业: {industry_info['industry']}")
# 输出: 行业: 酿酒行业
```

### 换手率分析

```python
quantile = fetcher.fetch_turnover_quantile(
    symbol="600519",
    current_date=datetime.now(),
    lookback_days=100
)

print(f"换手率分位数: {quantile:.2%}")
if quantile < 0.2:
    print("💡 地量区域")
elif quantile > 0.8:
    print("💡 放量区域")
```

## 🔧 进阶用法

### 分钟线数据

```python
# 获取5分钟K线
df_5min = manager.fetch_intraday_data(
    symbol="600519",
    start_date=datetime.now() - timedelta(days=5),
    end_date=datetime.now(),
    period="5"  # 可选: "1", "5", "15", "30", "60"
)
```

### 自定义缓存目录

```python
# 使用自定义数据目录
manager = DataManager(data_dir="/path/to/your/data")
```

### 清除缓存

```python
# 清除单个股票缓存
manager.clear_cache("600519")

# 清除所有缓存
manager.clear_cache()
```

## 💡 最佳实践

### 1. 日常更新策略

```python
def daily_update(symbols):
    """每日收盘后执行的更新脚本"""
    manager = DataManager()
    
    for symbol in symbols:
        try:
            df = manager.fetch_data_incremental(symbol)
            print(f"✓ {symbol} 更新完成，共 {len(df)} 条记录")
        except Exception as e:
            print(f"✗ {symbol} 更新失败: {e}")

# 在 crontab 或定时任务中调用
symbols = ["600519", "600036", "000858"]
daily_update(symbols)
```

### 2. 数据质量检查

```python
def check_data_quality(df, symbol):
    """检查数据质量"""
    print(f"\n数据质量报告 - {symbol}")
    print(f"总记录数: {len(df)}")
    print(f"日期范围: {df.index.min().date()} 到 {df.index.max().date()}")
    
    # 检查缺失值
    missing = df.isnull().sum()
    if missing.any():
        print("⚠️  发现缺失值:")
        print(missing[missing > 0])
    else:
        print("✓ 无缺失值")
    
    # 检查价格关系
    valid_price = (
        (df['high'] >= df['close']) & 
        (df['close'] >= df['low']) &
        (df['high'] >= df['open']) &
        (df['open'] >= df['low'])
    ).all()
    
    if valid_price:
        print("✓ 价格关系正常")
    else:
        print("⚠️  价格关系异常")

# 使用
df = manager.fetch_data("600519")
check_data_quality(df, "600519")
```

### 3. 批量下载并保存

```python
def download_stock_pool(symbols, days=365):
    """批量下载股票池数据"""
    manager = DataManager()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    results = manager.fetch_multiple(symbols, start_date, end_date)
    
    success_count = sum(1 for df in results.values() if df is not None)
    print(f"\n✓ 成功下载 {success_count}/{len(symbols)} 只股票")
    
    return results

# 下载整个股票池
my_pool = ["600519", "600036", "000858", "601318", "600887"]
data = download_stock_pool(my_pool, days=365)
```

## 🚨 常见问题

### Q1: 为什么获取数据很慢？
A: 首次下载会从网络获取，后续使用缓存会很快。使用`fetch_data_incremental`进行增量更新。

### Q2: 如何判断数据是从缓存还是网络？
A: 查看输出信息：
- "Using cached data for..." → 使用缓存
- "Fetching data for..." → 从网络获取

### Q3: 缓存数据存储在哪里？
A: 默认在 `data/cn/` 目录下，Parquet格式。

### Q4: 能获取多长时间的历史数据？
A: 取决于数据源，通常可获取上市以来的全部数据。

### Q5: 为什么有些股票获取失败？
A: 可能原因：
- 股票代码错误（A股必须是6位数字）
- 网络问题
- API限流（稍后重试）
- 股票已退市

## 📖 更多文档

- 完整功能文档: `docs/CN_DATA_MODULE.md`
- 测试用例: `test/test_cn_fetcher.py`
- 演示脚本: `demo_cn_data.py`

## 🎯 下一步

现在你已经可以：
1. ✅ 获取A股日线数据
2. ✅ 使用缓存和增量更新
3. ✅ 批量处理多只股票
4. ✅ 获取实时行情和行业数据

继续学习：
- 使用这些数据构建因子库（`src/factors/`）
- 开发交易策略（`src/backtester/`）
- 训练预测模型（`src/models/`）

