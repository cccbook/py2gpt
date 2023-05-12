# OpenAi的API

* [只要會寫 Python 的人，都能簡單上手的 ChatGPT Python API](https://pecutsai.medium.com/%E5%8F%AA%E8%A6%81%E6%9C%83%E5%AF%AB-python-%E7%9A%84%E4%BA%BA-%E9%83%BD%E8%83%BD%E7%B0%A1%E5%96%AE%E4%B8%8A%E6%89%8B%E7%9A%84-chatgpt-python-api-cc4d3ed2234a)
	* https://platform.openai.com/docs/guides/chat/introduction

## ChatGPT:

範例：

* https://github.com/cycleapple/GPTTools/blob/master/email_generation.py
    * https://www.facebook.com/groups/2152027081656284/user/100000299381678/

## ccc: 請用 python 呼叫 openAI 的 ChatGPT

* https://platform.openai.com/docs/libraries

要使用 Python 调用 OpenAI 的 ChatGPT，您需要先安装 openai 包并获取您的 API 密钥。在安装包之前，您需要确保已安装 Python 环境。下面是使用 pip 安装 openai 包的示例命令：

```
pip install openai
```

在安装完成后，您可以使用以下代码调用 ChatGPT：

```
import openai
openai.api_key = "YOUR_API_KEY" # 将 YOUR_API_KEY 替换为您的 API 密钥

prompt = "Hello, how are you?"

# 调用 GPT-3 API
response = openai.Completion.create(
  engine="davinci", # 使用 davinci 引擎
  prompt=prompt,
  max_tokens=60, # 最大标记数
  n=1, # 生成一条完整的回复
  stop=None,
  temperature=0.5 # 控制生成文本的随机性。范围从 0（完全确定）到 1（完全随机）
)

print(response.choices[0].text.strip())

```

## 嵌入 Embedding 向量

https://platform.openai.com/docs/guides/embeddings/what-are-embeddings


1. How to get embeddings

    To get an embedding, send your text string to the embeddings API endpoint along with a choice of embedding model ID (e.g., text-embedding-ada-002). The response will contain an embedding, which you can extract, save, and use.

```
curl https://api.openai.com/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "input": "Your text string goes here",
    "model": "text-embedding-ada-002"
  }'
```

Text search using embeddings

* https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb

```py
from openai.embeddings_utils import get_embedding, cosine_similarity
 
df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
 
def search_functions(df, code_query, n=3, pprint=True, n_lines=7):
   embedding = get_embedding(code_query, model='text-embedding-ada-002')
   df['similarities'] = df.code_embedding.apply(lambda x: cosine_similarity(x, embedding))
 
   res = df.sort_values('similarities', ascending=False).head(n)
   return res
res = search_functions(df, 'Completions API tests', n=3)
```

