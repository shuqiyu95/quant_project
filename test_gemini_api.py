"""
测试 Gemini API 调用

验证 Gemini API 是否能正常工作
"""

import logging
from src.llm import GeminiDeepResearchClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_simple_chat():
    """测试简单的对话功能"""
    print("\n" + "="*60)
    print("测试 1: 简单对话")
    print("="*60)
    
    try:
        # 初始化客户端
        client = GeminiDeepResearchClient()
        
        # 简单问答
        prompt = "你好，请用一句话介绍你自己。"
        print(f"\n提示词: {prompt}\n")
        
        result = client.chat(prompt=prompt)
        
        print(f"✅ API 调用成功!")
        print(f"模型: {result['model']}")
        print(f"用时: {result['elapsed_time']:.2f}s")
        
        if result.get('usage'):
            usage = result['usage']
            print(f"Token 使用: {usage.get('total_tokens', 'N/A')} "
                  f"(提示: {usage.get('prompt_tokens', 'N/A')}, "
                  f"回复: {usage.get('candidates_tokens', 'N/A')})")
        
        print(f"\n回复内容:\n{result['content']}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ API 调用失败: {str(e)}")
        logger.error("测试失败", exc_info=True)
        return False


def test_stock_analysis():
    """测试股票分析查询"""
    print("\n" + "="*60)
    print("测试 2: 股票分析")
    print("="*60)
    
    try:
        client = GeminiDeepResearchClient()
        
        prompt = """
        请简要分析以下信息：
        
        股票: 特斯拉 (TSLA)
        当前价格: $250
        近期趋势: 上涨 +15%
        
        请从以下角度分析：
        1. 价格走势评价（1-2句）
        2. 短期建议（1-2句）
        
        请保持简洁，总共不超过100字。
        """
        
        print(f"\n提示词: {prompt[:100]}...\n")
        
        result = client.chat(
            prompt=prompt,
            temperature=0.5,  # 较低温度以获得更确定的回答
            system_instruction="你是一个专业的股票分析师，擅长简洁清晰的分析。"
        )
        
        print(f"✅ API 调用成功!")
        print(f"模型: {result['model']}")
        print(f"用时: {result['elapsed_time']:.2f}s")
        
        print(f"\n分析结果:\n{result['content']}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ API 调用失败: {str(e)}")
        logger.error("测试失败", exc_info=True)
        return False


def test_different_models():
    """测试不同模型"""
    print("\n" + "="*60)
    print("测试 3: 不同模型")
    print("="*60)
    
    client = GeminiDeepResearchClient()
    
    models = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    
    prompt = "用一句话解释什么是量化交易。"
    
    for model in models:
        try:
            print(f"\n测试模型: {model}")
            result = client.chat(prompt=prompt, model=model)
            
            print(f"✅ {model} 调用成功")
            print(f"回复: {result['content'][:100]}...")
            
        except Exception as e:
            print(f"❌ {model} 调用失败: {str(e)}")


def test_token_limit():
    """测试 token 限制"""
    print("\n" + "="*60)
    print("测试 4: Token 限制")
    print("="*60)
    
    try:
        client = GeminiDeepResearchClient()
        
        prompt = "请用50个字以内介绍美国股市。"
        
        result = client.chat(
            prompt=prompt,
            max_output_tokens=100  # 限制输出长度
        )
        
        print(f"✅ API 调用成功!")
        print(f"回复长度: {len(result['content'])} 字符")
        print(f"回复: {result['content']}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ API 调用失败: {str(e)}")
        return False


def test_error_handling():
    """测试错误处理"""
    print("\n" + "="*60)
    print("测试 5: 错误处理")
    print("="*60)
    
    try:
        # 使用无效的 API Key
        client = GeminiDeepResearchClient(api_key="invalid_key_12345")
        result = client.chat("Hello")
        
        print(f"❌ 应该抛出异常但没有")
        return False
        
    except Exception as e:
        print(f"✅ 正确捕获异常: {type(e).__name__}")
        print(f"错误信息: {str(e)[:100]}...")
        return True


def main():
    """运行所有测试"""
    print("\n" + "🚀 "*20)
    print("Gemini API 连通性测试")
    print("🚀 "*20)
    
    tests = [
        ("简单对话", test_simple_chat),
        ("股票分析", test_stock_analysis),
        ("不同模型", test_different_models),
        ("Token 限制", test_token_limit),
        ("错误处理", test_error_handling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result if result is not None else True))
        except Exception as e:
            logger.error(f"测试 '{test_name}' 执行失败", exc_info=True)
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status} - {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\n总计: {total_passed}/{total_tests} 个测试通过")
    
    if total_passed == total_tests:
        print("\n🎉 所有测试通过！Gemini API 工作正常。")
    else:
        print(f"\n⚠️  有 {total_tests - total_passed} 个测试失败。")


if __name__ == '__main__':
    main()

