# LLM 模块使用指南

## 📋 概述

LLM 模块提供 Gemini Deep Research API 集成，支持深度研究报告生成、保存和管理。

### 核心功能
- ✅ Gemini Deep Research API 调用
- ✅ 研究报告自动保存（按日期组织）
- ✅ 报告检索和加载
- ✅ 批量研究支持
- ✅ 元数据管理

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

设置环境变量：

```bash
export GEMINI_API_KEY='your_api_key_here'
```

或在代码中直接传入：

```python
from src.llm import GeminiDeepResearchClient

client = GeminiDeepResearchClient(api_key='your_api_key_here')
```

### 3. 基础使用

```python
from src.llm import GeminiDeepResearchClient, ReportManager

# 初始化客户端和管理器
client = GeminiDeepResearchClient()
manager = ReportManager(base_dir='data/reports')

# 执行研究
result = client.deep_research(
    query="分析特斯拉 (TSLA) 2024年Q4的财务表现"
)

# 保存报告
report_path = manager.save_report(
    report_data=result,
    filename='tsla_q4_2024_analysis'
)

print(f"报告已保存: {report_path}")
```

---

## 🔧 核心类和方法

### GeminiDeepResearchClient

#### 初始化

```python
client = GeminiDeepResearchClient(
    api_key=None,              # API Key（可选，从环境变量读取）
    base_url="https://...",    # API 基础 URL
    timeout=300,               # 请求超时时间（秒）
    max_retries=3             # 最大重试次数
)
```

#### deep_research()

执行单个深度研究查询。

```python
result = client.deep_research(
    query="研究问题或主题",
    model="gemini-2.0-flash-thinking-exp",  # 模型名称
    temperature=0.7,                        # 生成温度 (0.0-1.0)
    max_output_tokens=None,                 # 最大输出 token 数
    metadata={'ticker': 'AAPL'}            # 额外元数据
)
```

**返回格式：**
```python
{
    'content': '研究报告内容...',
    'query': '原始查询',
    'model': 'gemini-2.0-flash-thinking-exp',
    'timestamp': '2024-12-25T10:30:00',
    'elapsed_time': 12.5,
    'metadata': {...},
    'thinking_process': '思考过程...'
}
```

#### batch_research()

批量执行多个研究查询。

```python
results = client.batch_research(
    queries=['查询1', '查询2', '查询3'],
    model="gemini-2.0-flash-thinking-exp",
    temperature=0.7,
    delay=1.0  # 每次请求间隔（秒）
)
```

---

### ReportManager

#### 初始化

```python
manager = ReportManager(base_dir='data/reports')
```

#### save_report()

保存研究报告到本地。

```python
report_path = manager.save_report(
    report_data=result,           # 来自 deep_research() 的结果
    filename='tsla_analysis',     # 文件名（可选，自动生成）
    date='2024-12-25',           # 日期（可选，默认今天）
    save_metadata=True           # 是否保存元数据
)
```

**目录结构：**
```
data/reports/
├── 2024-12-25/
│   ├── tsla_analysis.txt        # 报告内容
│   ├── tsla_analysis.json       # 元数据
│   └── tsla_analysis_thinking.txt  # 思考过程（如有）
└── 2024-12-24/
    └── ...
```

#### load_report()

加载已保存的报告。

```python
report = manager.load_report(
    filename='tsla_analysis',
    date='2024-12-25',  # 可选，默认今天
    load_metadata=True
)
```

#### list_reports()

列出指定日期的所有报告。

```python
reports = manager.list_reports(
    date='2024-12-25',      # 可选，默认今天
    include_metadata=True   # 是否包含元数据
)

for report in reports:
    print(f"文件名: {report['filename']}")
    print(f"大小: {report['size']} 字节")
    print(f"查询: {report['metadata']['query']}")
```

#### list_dates()

列出所有有报告的日期。

```python
dates = manager.list_dates()
# 返回: ['2024-12-20', '2024-12-21', '2024-12-25']
```

#### search_reports()

搜索包含关键词的报告。

```python
results = manager.search_reports(
    keyword='TSLA',
    date=None,             # 可选，搜索所有日期
    search_content=True    # 是否搜索报告内容
)
```

#### delete_report()

删除报告。

```python
success = manager.delete_report(
    filename='tsla_analysis',
    date='2024-12-25'
)
```

---

## 💡 使用场景

### 场景 1: 股票深度研究

```python
from src.llm import GeminiDeepResearchClient, ReportManager

client = GeminiDeepResearchClient()
manager = ReportManager()

# 研究多只股票
tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']

for ticker in tickers:
    query = f"分析 {ticker} 的最新财务状况、业务前景和投资价值"
    
    result = client.deep_research(
        query=query,
        metadata={'ticker': ticker, 'type': 'fundamental_analysis'}
    )
    
    manager.save_report(
        report_data=result,
        filename=f"{ticker.lower()}_fundamental_analysis"
    )
    
    print(f"✅ {ticker} 分析完成")
```

### 场景 2: 行业研究

```python
# 研究特定行业
query = """
分析半导体行业的现状和未来趋势，包括：
1. 主要玩家和竞争格局
2. 技术发展趋势（如 AI 芯片、先进制程）
3. 供应链挑战和机遇
4. 投资机会分析
"""

result = client.deep_research(
    query=query,
    temperature=0.8,  # 更高的创造性
    metadata={'industry': 'semiconductor', 'type': 'industry_research'}
)

manager.save_report(
    report_data=result,
    filename='semiconductor_industry_analysis'
)
```

### 场景 3: 定期研究报告

```python
from datetime import datetime

# 每日市场分析
def daily_market_research():
    today = datetime.now().strftime('%Y-%m-%d')
    
    query = f"""
    生成 {today} 的市场研究报告：
    1. 美股主要指数表现
    2. 热点板块和个股
    3. 重要经济数据和事件
    4. 明日市场展望
    """
    
    result = client.deep_research(query=query)
    
    manager.save_report(
        report_data=result,
        filename=f'daily_market_report_{today}',
        date=today
    )
    
    return result

# 可以设置定时任务每日执行
```

### 场景 4: 批量竞品分析

```python
# 批量分析竞争对手
competitors = [
    "分析苹果 (AAPL) 在智能手机市场的竞争优势",
    "分析三星在智能手机市场的战略",
    "分析小米在智能手机市场的定位",
]

results = client.batch_research(
    queries=competitors,
    delay=2.0  # API 限流间隔
)

for i, result in enumerate(results, 1):
    if 'error' not in result:
        manager.save_report(result, filename=f'competitor_analysis_{i}')
```

---

## 🔍 报告检索示例

### 查看今天的所有报告

```python
reports = manager.list_reports()

for report in reports:
    print(f"\n{'='*60}")
    print(f"文件名: {report['filename']}")
    print(f"大小: {report['size']} 字节")
    print(f"修改时间: {report['modified']}")
```

### 搜索特定股票的报告

```python
# 搜索所有关于 TSLA 的报告
tsla_reports = manager.search_reports(
    keyword='TSLA',
    search_content=True
)

print(f"找到 {len(tsla_reports)} 个关于 TSLA 的报告")

for report in tsla_reports:
    print(f"- {report['filename']} ({report['date']})")
```

### 加载并分析历史报告

```python
# 加载上周的报告
report = manager.load_report(
    filename='weekly_market_report',
    date='2024-12-18'
)

print(f"报告内容:\n{report['content']}")
print(f"\n查询: {report['metadata']['query']}")
print(f"生成时间: {report['metadata']['elapsed_time']:.2f}s")
```

---

## ⚙️ 高级配置

### 自定义模型参数

```python
result = client.deep_research(
    query="深度分析问题",
    model="gemini-2.0-flash-thinking-exp",
    temperature=0.9,           # 更高的创造性
    max_output_tokens=8192,    # 更长的输出
    metadata={'priority': 'high'}
)
```

### 错误处理

```python
try:
    result = client.deep_research(query="分析查询")
    manager.save_report(result, filename='my_report')
    
except ValueError as e:
    print(f"参数错误: {e}")
    
except Exception as e:
    print(f"API 调用失败: {e}")
```

### 自定义重试逻辑

```python
client = GeminiDeepResearchClient(
    api_key='your_key',
    timeout=600,      # 10分钟超时
    max_retries=5     # 最多重试5次
)
```

---

## 📊 与 Quant Project 集成

### 结合量化分析

```python
from src.data_engine import DataManager
from src.llm import GeminiDeepResearchClient, ReportManager

# 获取股票数据
dm = DataManager()
df = dm.fetch_data('AAPL', start_date='2024-01-01')

# 计算收益率
returns = df['close'].pct_change()
annual_return = returns.mean() * 252
volatility = returns.std() * (252 ** 0.5)

# 生成分析查询
query = f"""
基于以下数据分析 AAPL 的投资价值：
- 年化收益率: {annual_return:.2%}
- 年化波动率: {volatility:.2%}
- 当前价格: ${df['close'].iloc[-1]:.2f}

请提供深度分析和投资建议。
"""

# 执行研究
client = GeminiDeepResearchClient()
result = client.deep_research(query)

# 保存报告
manager = ReportManager()
manager.save_report(result, filename='aapl_quant_analysis')
```

### 自动化投研流程

```python
def automated_research_pipeline(ticker):
    """自动化投研流程"""
    
    # 1. 获取数据
    dm = DataManager()
    df = dm.fetch_data(ticker, start_date='2023-01-01')
    
    # 2. 计算因子
    from src.factors import calculate_alpha158
    factors = calculate_alpha158(df)
    
    # 3. 生成研究查询
    query = f"""
    请基于以下信息深度分析 {ticker}：
    1. 技术分析（价格趋势、动量指标）
    2. 基本面分析（财务健康度、业务前景）
    3. 风险评估
    4. 投资建议
    """
    
    # 4. 执行研究
    client = GeminiDeepResearchClient()
    result = client.deep_research(
        query=query,
        metadata={'ticker': ticker, 'data_points': len(df)}
    )
    
    # 5. 保存报告
    manager = ReportManager()
    report_path = manager.save_report(
        result,
        filename=f'{ticker.lower()}_comprehensive_analysis'
    )
    
    return report_path

# 使用
report_path = automated_research_pipeline('TSLA')
print(f"研究报告已生成: {report_path}")
```

---

## 🧪 测试

运行测试套件：

```bash
# 运行所有 LLM 模块测试
pytest test/test_llm.py -v

# 运行特定测试
pytest test/test_llm.py::TestGeminiDeepResearchClient::test_deep_research_success -v

# 查看覆盖率
pytest test/test_llm.py --cov=src.llm --cov-report=html
```

---

## 📝 注意事项

### API 限流

- Gemini API 有速率限制
- 批量请求时使用 `delay` 参数控制间隔
- 建议设置 `delay >= 1.0` 秒

### 成本控制

- 每次调用消耗 API 额度
- 使用 `max_output_tokens` 限制输出长度
- 缓存常用查询结果

### 数据安全

- 不要将 API Key 硬编码在代码中
- 使用环境变量或配置文件
- 敏感报告注意加密存储

### 存储管理

- 报告按日期自动组织
- 定期清理旧报告释放空间
- 考虑使用数据库存储元数据

---

## 🔗 相关文档

- [项目开发指南](claude.md)
- [快速开始](QUICKSTART.md)
- [数据引擎文档](CN_DATA_MODULE.md)
- [因子库指南](factors_guide.md)

---

**最后更新**: 2024-12-25  
**模块版本**: v1.0.0  
**维护者**: Quant Team

