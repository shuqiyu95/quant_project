# 同花顺热股榜数据抓取与热度因子生成

## 功能概述

`hot_stock.py` 是一个完整的热股榜数据抓取和分析工具，提供以下功能：

1. **每日数据抓取**：自动抓取同花顺热股榜前N名（默认10名）
2. **多周期涨幅计算**：自动计算每只股票的1日、3日、5日、10日涨幅
3. **数据存储**：按日期命名保存到 `data/hot_stock/` 目录
4. **周度统计**：统计指定时间范围内的在榜次数和排名
5. **热度因子生成**：基于多种算法生成股票热度评分

## 目录结构

```
quant_project/
├── analysis/
│   └── hot_stock.py          # 主程序
└── data/
    └── hot_stock/            # 数据存储目录
        ├── 2024-12-25.csv    # 每日热股数据
        ├── 2024-12-26.csv
        └── heat_factor_2024-12-31.csv  # 热度因子
```

## 快速开始

### 1. 每日抓取任务

抓取当天的热股榜前10名：

```bash
cd /Users/yushuqi/Desktop/code/quant_project
python analysis/hot_stock.py --daily
```

抓取前20名：

```bash
python analysis/hot_stock.py --daily --top 20
```

### 2. 每周统计任务

统计最近7天的数据并生成热度因子：

```bash
python analysis/hot_stock.py --weekly
```

统计最近14天：

```bash
python analysis/hot_stock.py --weekly --days 14
```

### 3. 自定义日期范围

生成指定日期范围的热度因子：

```bash
python analysis/hot_stock.py --generate --start 2024-12-01 --end 2024-12-25
```

### 4. 使用不同的计算方法

```bash
# 加权方法（默认）：综合考虑在榜次数和排名
python analysis/hot_stock.py --weekly --method weighted

# 简单方法：只考虑在榜次数
python analysis/hot_stock.py --weekly --method simple

# 排名方法：重点考虑排名位置
python analysis/hot_stock.py --weekly --method rank_based
```

## 数据格式

### 每日数据 (YYYY-MM-DD.csv)

| 列名 | 说明 | 示例 |
|------|------|------|
| rank | 排名 | 1 |
| symbol | 股票代码 | 600519 |
| name | 股票名称 | 贵州茅台 |
| price | 最新价 (元) | 1680.50 |
| change_pct | 当日涨跌幅 (%) | 10.00 |
| return_1d | 1日涨幅 (%) | 10.00 |
| return_3d | 3日涨幅 (%) | 14.61 |
| return_5d | 5日涨幅 (%) | 29.16 |
| return_10d | 10日涨幅 (%) | 38.27 |

**说明**：
- 所有浮点数保留两位小数
- 涨幅数据基于前复权价格计算
- 如果历史数据不足，涨幅字段可能为空

### 热度因子 (heat_factor_YYYY-MM-DD.csv)

| 列名 | 说明 | 示例 |
|------|------|------|
| symbol | 股票代码 | 600519 |
| name | 股票名称 | 贵州茅台 |
| appearance_count | 在榜次数 | 7 |
| avg_rank | 平均排名 | 2.3 |
| min_rank | 最好排名 | 1 |
| max_rank | 最差排名 | 5 |
| heat_score | 热度得分 | 95.6 |
| heat_rank | 热度排名 | 1 |
| start_date | 统计开始日期 | 2024-12-19 |
| end_date | 统计结束日期 | 2024-12-25 |
| method | 计算方法 | weighted |
| generate_time | 生成时间 | 2024-12-25 16:00:00 |

## 热度因子计算方法

### 1. Weighted（加权方法，默认）

综合考虑在榜次数和排名的加权得分：

```
heat_score = normalized_appearance * 0.6 + normalized_rank_score * 0.4
```

- **在榜次数权重**: 60%
- **排名权重**: 40%
- **特点**: 平衡考虑出现频率和排名位置

### 2. Simple（简单方法）

只考虑在榜次数：

```
heat_score = appearance_count * 10
```

- **特点**: 简单直观，适合快速筛选
- **适用场景**: 只关注曝光度

### 3. Rank-based（排名方法）

重点考虑排名位置：

```
heat_score = appearance_count * (10.0 / avg_rank)
```

- **特点**: 排名越靠前得分越高
- **适用场景**: 关注头部热股

## Python API 使用

### 基础使用

```python
from analysis.hot_stock import HotStockTracker

# 创建跟踪器
tracker = HotStockTracker()

# 抓取热股数据
df = tracker.fetch_hot_stocks(top_n=10)
print(df)

# 保存数据
tracker.save_daily_data(df)
```

### 加载历史数据

```python
# 加载指定日期的数据
df = tracker.load_daily_data('2024-12-25')

# 加载日期范围的数据
df = tracker.load_date_range_data('2024-12-19', '2024-12-25')
```

### 生成热度因子

```python
# 计算周统计
stats_df = tracker.calculate_weekly_stats('2024-12-19', '2024-12-25')
print(stats_df)

# 生成热度因子
heat_df = tracker.generate_heat_factor(
    start_date='2024-12-19',
    end_date='2024-12-25',
    method='weighted'
)

# 保存热度因子
tracker.save_heat_factor(heat_df)
```

### 自动化任务

```python
# 每日任务
success = tracker.run_daily_task(top_n=10)

# 每周任务
success = tracker.run_weekly_task(days=7, method='weighted')
```

## 定时任务设置

### 使用 crontab（Linux/Mac）

编辑 crontab：

```bash
crontab -e
```

添加定时任务：

```bash
# 每天下午3点抓取热股榜
0 15 * * * cd /Users/yushuqi/Desktop/code/quant_project && python analysis/hot_stock.py --daily

# 每周一早上9点生成热度因子
0 9 * * 1 cd /Users/yushuqi/Desktop/code/quant_project && python analysis/hot_stock.py --weekly
```

### 使用 Windows 任务计划程序

1. 打开"任务计划程序"
2. 创建基本任务
3. 设置触发器（每天/每周）
4. 操作：启动程序
   - 程序：`python`
   - 参数：`analysis/hot_stock.py --daily`
   - 起始于：`C:\path\to\quant_project`

## 注意事项

1. **数据源**：使用 AkShare 的 `stock_hot_rank_wc()` 接口抓取同花顺热股榜
2. **网络要求**：需要网络连接才能抓取数据
3. **频率限制**：建议每天抓取1-2次，避免频繁请求
4. **数据完整性**：建议每天固定时间抓取，确保数据连续性
5. **排名说明**：排名数字越小表示越热（第1名最热）

## 应用场景

### 1. 市场情绪监控

通过热股榜追踪市场热点和资金流向：

```python
# 获取本周最热的股票
heat_df = tracker.generate_heat_factor('2024-12-19', '2024-12-25')
top_5 = heat_df.head(5)
print("本周最热股票：")
print(top_5[['symbol', 'name', 'heat_score', 'appearance_count']])
```

### 2. 涨幅分析

分析热股的短期和中期表现：

```python
# 加载今天的热股数据
df = tracker.load_daily_data('2024-12-25')

# 筛选短期强势股（3日涨幅 > 15%）
strong_stocks = df[df['return_3d'] > 15]
print("短期强势股：")
print(strong_stocks[['rank', 'symbol', 'name', 'return_1d', 'return_3d', 'return_5d']])

# 筛选中期强势股（10日涨幅 > 30%）
medium_strong = df[df['return_10d'] > 30]
print("\n中期强势股：")
print(medium_strong[['symbol', 'name', 'return_5d', 'return_10d']])

# 分析涨幅动能（3日涨幅 > 1日涨幅，说明还在加速）
momentum_stocks = df[df['return_3d'] > df['return_1d'] * 1.2]
print("\n有上涨动能的股票：")
print(momentum_stocks[['symbol', 'name', 'return_1d', 'return_3d']])
```

### 3. 选股策略

结合热度因子和涨幅进行选股：

```python
# 加载今日热股数据
df = tracker.load_daily_data('2024-12-25')

# 策略1：排名靠前且中期涨幅较大
strategy1 = df[
    (df['rank'] <= 5) & 
    (df['return_10d'] > 20)
]

# 策略2：近期涨幅适中，避免追高
strategy2 = df[
    (df['return_3d'] > 5) & 
    (df['return_3d'] < 20) &
    (df['rank'] <= 10)
]

# 策略3：结合热度因子
heat_df = tracker.generate_heat_factor('2024-12-19', '2024-12-25')
hot_stocks = heat_df[
    (heat_df['heat_score'] > 80) & 
    (heat_df['appearance_count'] >= 5)
]
```

### 4. 风险提示

识别过热股票和风险信号：

```python
# 加载数据
df = tracker.load_daily_data('2024-12-25')

# 警告1：连续大涨可能过热（10日涨幅 > 50%）
overheated = df[df['return_10d'] > 50]
print("警告：以下股票短期涨幅过大")
print(overheated[['symbol', 'name', 'return_5d', 'return_10d']])

# 警告2：涨幅衰减（1日涨幅 < 3日平均涨幅的一半）
df['avg_3d'] = df['return_3d'] / 3
losing_momentum = df[df['return_1d'] < df['avg_3d'] / 2]
print("\n警告：以下股票上涨动能减弱")
print(losing_momentum[['symbol', 'name', 'return_1d', 'return_3d']])

# 警告3：排名靠前但涨幅为负（可能是利空消息）
negative_hot = df[(df['rank'] <= 10) & (df['return_1d'] < 0)]
if not negative_hot.empty:
    print("\n警告：热股出现下跌")
    print(negative_hot[['symbol', 'name', 'return_1d', 'change_pct']])
```

## 常见问题

### Q1: 抓取失败怎么办？

**A**: 检查以下几点：
- 网络连接是否正常
- AkShare 是否安装正确：`pip install akshare --upgrade`
- 是否在交易时间段（数据可能更新不及时）

### Q2: 数据不完整怎么办？

**A**: 
- 确保每天定时抓取
- 如果缺失某天数据，可以手动运行补充
- 周统计会自动跳过缺失的日期

### Q3: 如何修改热度因子权重？

**A**: 编辑 `hot_stock.py` 中的 `generate_heat_factor` 方法：

```python
# 修改权重比例（当前是 60% 在榜次数 + 40% 排名）
stats_df['heat_score'] = (
    normalized_appearance * 0.6 +  # 可修改此处
    normalized_rank_score * 0.4    # 可修改此处
) * 100
```

### Q4: 如何集成到现有策略？

**A**: 
```python
from analysis.hot_stock import HotStockTracker

# 在策略中使用
tracker = HotStockTracker()
heat_df = tracker.generate_heat_factor('2024-12-19', '2024-12-25')

# 获取热度因子字典
heat_dict = dict(zip(heat_df['symbol'], heat_df['heat_score']))

# 在选股时参考热度
if symbol in heat_dict and heat_dict[symbol] > 80:
    # 高热度股票，可能需要特殊处理
    pass
```

## 更新日志

### v1.1.0 (2025-12-25)

- ✅ 新增多周期涨幅计算功能（1日、3日、5日、10日）
- ✅ 优化CSV存储格式，去除冗余字段
- ✅ 所有浮点数统一保留两位小数
- ✅ 涨幅数据基于前复权价格计算
- ✅ 增强应用场景示例和分析策略

### v1.0.0 (2024-12-25)

- ✅ 实现每日热股榜抓取功能
- ✅ 实现按日期存储数据
- ✅ 实现周度统计功能
- ✅ 实现三种热度因子计算方法
- ✅ 提供完整的命令行接口
- ✅ 提供 Python API

## 许可证

本项目为内部使用工具，请遵守数据源的使用条款。

## 联系方式

如有问题或建议，请联系项目维护者。

