import numpy as np
import re
from openai import OpenAI
from time import time, sleep
import pickle
from concurrent.futures import ThreadPoolExecutor
import torch

LLM_DICT = {
    'deepseek-coder': {
        'name': 'deepseek-coder',
        'source': 'deepseek'
    }}

def extract_float_result(content):
    try:
        pattern = r'"vacc":\s*([-\d\.]+)'
        match = re.search(pattern, content)
        if match:
            return float(match.group(1))
        else:
            return None
    except Exception as e:
        print(e)
        print(content)
        return None

def extract_list_result(content):
    pattern = r'"vacc":\s*(\[[-\d\.\,\s]+\])'
    match = re.search(pattern, content)
    if match:
        return eval(match.group(1))
    else:
        return None

def llm_infer(user_msg, model, system_msg, extract_result_func, verify_func=None, num_infer=1, num_iter=1, temperature=0, seed=10):
    source = LLM_DICT[model]['source']
    if source == 'main':
        client = OpenAI(api_key='sk-OsNcGo4GdRYjhn4D1554622d5b1443A593AeF1788c44E1D9',
                        base_url='https://aigc456.top/v1/')
    elif source == '35':
        client = OpenAI(api_key='sk-OixveBFDaew2JD3N70C5E9E8946a4383A9E63f3831517d5a',
                        base_url='https://35us.aigcbest.top/v1/')
    elif source == 'deepseek':
        client = OpenAI(api_key='sk-070007d5096342d59986e5e26302bb4f',
                        base_url='https://api.deepseek.com')
    elif source == 'glm':
        client = OpenAI(api_key='c6239cc2b32377951b7463dde1ca7008.1ZBjnsRHtNywWHwg',
                        base_url='https://open.bigmodel.cn/api/paas/v4/')
    elif source == 'qwen':
        client = OpenAI(api_key='sk-075f04173e9c4b6eafc36ea351d35d3c',
                        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1/')

    if type(user_msg) == str:
        sample_id = int(hash(user_msg) % 1e6)
    else:
        sample_id, user_msg = user_msg
    if verify_func is None:
        verify_func = lambda x: max(num_infer, num_iter) // 2

    result_list, content_list, logprobs_list = [], [], []
    for i in range(num_iter):
        result, cnt, content, logprobs = None, 0, None, None
        while result is None and cnt < 3:
            signal = False
            start = time()
            completion = None
            while not signal and time() - start <= 300:
                try:
                    completion = client.chat.completions.create(model=LLM_DICT[model]['name'],
                                                                messages=[{'role': 'system', 'content': system_msg},
                                                                          {'role': 'user', 'content': user_msg}],
                                                                n=num_infer, temperature=temperature, logprobs=True, seed=seed+sample_id+cnt)
                    signal = True
                except Exception as err:
                    error_code = 'sample_id: %s, error type: %s, error: %s' % (sample_id, type(err), err)
                    print(error_code)
                    sleep(30)
                    continue
            if not signal:
                print('Unexpected signal')
                continue
            if completion is not None and completion.choices is not None:
                content = completion.choices[0].message.content
                logprobs = []
                if completion.choices[0].logprobs is not None:
                    logprobs = [token.logprob for token in completion.choices[0].logprobs.content]
                result = extract_result_func(content)
            cnt += 1
        result_list.append(result)
        content_list.append(content)
        logprobs_list.append(logprobs)
    index = verify_func(result_list)
    result, content, logprobs = result_list[index], content_list[index], logprobs_list[index]
    if result is None:
        print('No completion found for sample_id %s.\nThe content is %s' % (sample_id, content))
    return sample_id, result, content, logprobs