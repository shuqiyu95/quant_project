"""
LLM 模块使用示例

演示如何使用 Gemini Deep Research API 进行研究并保存报告
"""

import os
import logging
from src.llm import GeminiDeepResearchClient, ReportManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_basic_research():
    """基础研究示例"""
    print("\n" + "="*60)
    print("示例 1: 基础深度研究")
    print("="*60)
    
    # 初始化客户端（需要设置环境变量 GEMINI_API_KEY）
    client = GeminiDeepResearchClient()
    
    # 初始化报告管理器
    manager = ReportManager(base_dir='data/reports')
    
    # 执行研究
    query = "分析特斯拉 (TSLA) 2024年Q4的财务表现，包括收入、利润和关键业务指标"
    print(f"\n查询: {query}\n")
    
    result = client.deep_research(
        query=query,
        temperature=0.7,
        metadata={'ticker': 'TSLA', 'quarter': 'Q4 2024'}
    )
    
    # 打印结果摘要
    print(f"模型: {result['model']}")
    print(f"生成时间: {result['elapsed_time']:.2f}s")
    print(f"内容长度: {len(result['content'])} 字符")
    print(f"\n内容预览:\n{result['content'][:500]}...\n")
    
    # 保存报告
    report_path = manager.save_report(
        report_data=result,
        filename='tsla_q4_2024_analysis'
    )
    
    print(f"✅ 报告已保存: {report_path}")


def example_batch_research():
    """批量研究示例"""
    print("\n" + "="*60)
    print("示例 2: 批量深度研究")
    print("="*60)
    
    client = GeminiDeepResearchClient()
    manager = ReportManager(base_dir='data/reports')
    
    # 批量查询
    queries = [
        "分析英伟达 (NVDA) 在AI芯片市场的竞争优势",
        "比较苹果 (AAPL) 和三星在智能手机市场的策略差异",
        "评估微软 (MSFT) 云计算业务的增长前景",
    ]
    
    print(f"\n执行 {len(queries)} 个查询...\n")
    
    results = client.batch_research(
        queries=queries,
        temperature=0.7,
        delay=1.0  # 每次请求间隔1秒
    )
    
    # 保存所有报告
    for i, result in enumerate(results, 1):
        if 'error' not in result:
            filename = f"batch_research_{i}"
            report_path = manager.save_report(result, filename=filename)
            print(f"✅ 报告 {i} 已保存: {report_path}")
        else:
            print(f"❌ 报告 {i} 生成失败: {result['error']}")


def example_report_management():
    """报告管理示例"""
    print("\n" + "="*60)
    print("示例 3: 报告管理")
    print("="*60)
    
    manager = ReportManager(base_dir='data/reports')
    
    # 列出所有日期
    dates = manager.list_dates()
    print(f"\n有报告的日期: {dates}")
    
    # 列出今天的所有报告
    today_reports = manager.list_reports(include_metadata=True)
    print(f"\n今天的报告数量: {len(today_reports)}")
    
    for report in today_reports:
        print(f"\n文件名: {report['filename']}")
        print(f"大小: {report['size']} 字节")
        print(f"修改时间: {report['modified']}")
        
        if 'metadata' in report:
            metadata = report['metadata']
            print(f"查询: {metadata.get('query', 'N/A')}")
            print(f"模型: {metadata.get('model', 'N/A')}")
    
    # 搜索报告
    print("\n" + "-"*60)
    print("搜索包含 'TSLA' 的报告...")
    
    matched_reports = manager.search_reports(
        keyword='TSLA',
        search_content=False  # 只搜索元数据
    )
    
    print(f"找到 {len(matched_reports)} 个匹配的报告")


def example_load_report():
    """加载报告示例"""
    print("\n" + "="*60)
    print("示例 4: 加载和读取报告")
    print("="*60)
    
    manager = ReportManager(base_dir='data/reports')
    
    # 列出今天的报告
    reports = manager.list_reports()
    
    if not reports:
        print("没有可用的报告")
        return
    
    # 加载第一个报告
    first_report = reports[0]
    print(f"\n加载报告: {first_report['filename']}")
    
    report_data = manager.load_report(
        filename=first_report['filename'],
        load_metadata=True
    )
    
    print(f"\n内容长度: {len(report_data['content'])} 字符")
    print(f"文件路径: {report_data['path']}")
    
    if 'metadata' in report_data:
        metadata = report_data['metadata']
        print(f"\n查询: {metadata.get('query', 'N/A')}")
        print(f"模型: {metadata.get('model', 'N/A')}")
        print(f"生成时间: {metadata.get('elapsed_time', 0):.2f}s")
    
    print(f"\n内容预览:\n{report_data['content'][:500]}...")


def example_custom_date():
    """自定义日期保存示例"""
    print("\n" + "="*60)
    print("示例 5: 指定日期保存报告")
    print("="*60)
    
    client = GeminiDeepResearchClient()
    manager = ReportManager(base_dir='data/reports')
    
    # 执行研究
    query = "分析亚马逊 (AMZN) 的云计算和电商双引擎战略"
    result = client.deep_research(query=query)
    
    # 保存到指定日期
    custom_date = "2024-12-20"
    report_path = manager.save_report(
        report_data=result,
        filename='amzn_strategy_analysis',
        date=custom_date
    )
    
    print(f"✅ 报告已保存到 {custom_date}: {report_path}")
    
    # 加载该报告
    loaded_report = manager.load_report(
        filename='amzn_strategy_analysis',
        date=custom_date
    )
    
    print(f"✅ 报告已加载，内容长度: {len(loaded_report['content'])} 字符")


def main():
    """运行所有示例"""
    
    # 检查 API Key
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  警告: 未设置环境变量 GEMINI_API_KEY")
        print("请设置后再运行示例:")
        print("  export GEMINI_API_KEY='your_api_key_here'")
        return
    
    try:
        # 运行示例（根据需要注释/取消注释）
        
        # 示例 1: 基础研究
        example_basic_research()
        
        # 示例 2: 批量研究（注意 API 限流）
        # example_batch_research()
        
        # 示例 3: 报告管理
        example_report_management()
        
        # 示例 4: 加载报告
        example_load_report()
        
        # 示例 5: 自定义日期
        # example_custom_date()
        
    except Exception as e:
        logger.error(f"示例执行失败: {str(e)}", exc_info=True)
        print(f"\n❌ 错误: {str(e)}")


if __name__ == '__main__':
    main()

