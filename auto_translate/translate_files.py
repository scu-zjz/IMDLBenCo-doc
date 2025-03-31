# 第三步：翻译处理 translate_files.py
import os
import json
import time
import openai
from openai import OpenAI
import argparse
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

import requests

def estimate_token_count(api_key, model, messages):
    url = "https://api.moonshot.cn/v1/tokenizers/estimate-token-count"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": messages
    }
    # print(f"Payload: {payload}")
    print(len(str(messages)))

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # 检查HTTP错误状态码
        
        result = response.json()
        # print(result)
        return result.get('data', {}).get('total_tokens', 0)
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None
    except ValueError as e:
        print(f"解析响应失败: {e}")
        return None

TRANSLATION_PROMPT = """你是一位专业的技术文档翻译员。请将以下Markdown内容精确翻译为英文，要求：
1. 保持原始格式和Markdown语法不变
2. 保留代码块、表格等特殊格式
3. 技术术语需要准确翻译
4. 中文专有名词保留原文"""
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type(openai._exceptions.RateLimitError),
    reraise=True
)
def translate_text_with_retry(content, api_key):
    client = OpenAI(
            api_key = api_key,
            base_url = "https://api.moonshot.cn/v1",
    )

    messages=[
                {"role": "system", "content": TRANSLATION_PROMPT},
                {"role": "user", "content": content}
    ]
    token_count = estimate_token_count(api_key, model="moonshot-v1-8k", messages=messages)
    print(f"Estimated Token count: {token_count}")
    select_model = "moonshot-v1-8k" if token_count <= 8192 / 2 else "moonshot-v1-32k" if token_count <= 128000 / 2 else "moonshot-v1-128k"
    MAX_TOKEN_LEN = 8192 if select_model == "moonshot-v1-8k" else 32768 if select_model == "moonshot-v1-32k" else 128000
    print(f"Selected Model: {select_model}")
    print(f"Max Token Length: {MAX_TOKEN_LEN}")

    max_tokens = MAX_TOKEN_LEN - token_count - 200  # 留出200个token用于响应
    print(max_tokens)
    response = client.chat.completions.create(
        model=select_model,
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens
        )
    print("Finish Reason:", response.choices[0].finish_reason)
    if response.choices[0].finish_reason == "length":  # <-- 当内容被截断时，finish_reason 的值为 length
        # 计算token count，选择合适的模型进行续写
        prefix = response.choices[0].message.content
        print(prefix, end="")  # <-- 在这里，你将看到被截断的部分输出内容
        response = client.chat.completions.create(
            model="moonshot-v1-128k",
            messages=[
                {"role": "system", "content": TRANSLATION_PROMPT},
                {"role": "user", "content": content},
                {"role": "assistant", "content": prefix, "partial": True},
            ],
            temperature=0.3,
            max_tokens=128000,  # <-- 注意这里，我们将 max_tokens 的值设置为一个较大的值，以确保 Kimi 大模型能完整输出内容
        )
        print(response.choices[0].message.content)  # <-- 在这里，你将看到 Kimi 大模型顺着之前已经输出的内容，继续将输出内容补全完整



    if response.choices[0].finish_reason != "stop":
        raise ValueError(f"Unexpected finish reason: {response.choices[0].finish_reason}")

    time.sleep(1)  # 强制请求间隔（根据您的RPM限制计算）
    return response.choices[0].message.content

def translate_text(content, api_key):
    try:
        return translate_text_with_retry(content, api_key)
    except openai._exceptions.RateLimitError as e:
        print(f"Rate limit exceeded even after retries: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected API Error: {str(e)}")
        return None
    
def process_files(config):
    with open(config['path_list'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    source_dir = data['source_dir']
    target_dir = config['target_dir']
    success = 0
    failed = []

    for rel_path in data['files']:
        src_path = os.path.join(source_dir, rel_path)
        dest_path = os.path.join(target_dir, rel_path)
        
        try:
            # 创建目标目录
            Path(os.path.dirname(dest_path)).mkdir(parents=True, exist_ok=True)
            
            # 读取原始内容
            with open(src_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 翻译内容
            translated = translate_text(content, config['api_key'])
            if not translated:
                failed.append(rel_path)
                continue
            
            # 写入翻译结果
            with open(dest_path, 'w', encoding='utf-8') as f:
                f.write(translated)
            
            success += 1
            print(f"Processed: {rel_path}")
        except Exception as e:
            print(f"Error processing {rel_path}: {str(e)}")
            failed.append(rel_path)

    print(f"\nProcess completed: {success} succeeded, {len(failed)} failed")
    if failed:
        print("Failed files:")
        print('\n'.join(failed))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file path')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    process_files(config)

if __name__ == '__main__':
    main()