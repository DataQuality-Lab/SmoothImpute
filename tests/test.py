import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from smoothimpute.imputers import Imputer
from smoothimpute.simulators import Simulator
from smoothimpute.evaluator import Evaluator
from smoothimpute.advisor import Advisor

# data = [[0, np.nan, 1],
#         [np.nan, 1, 0]]

data = np.random.rand(5, 3)
data = pd.DataFrame(data)

print(data)
simulator = Simulator(mcar_p=0.2, mar_p=0.1, mnar_p=0.1)
xmiss = simulator.simulate(data)
print(xmiss)

imputer = Imputer("mice")
data_filled = imputer.impute(xmiss)
print(data_filled)

evaluator = Evaluator()
result = evaluator.evaluate(xmiss, data_filled, data)
print(result)

evaluator = Evaluator("text")
result = evaluator.evaluate(xmiss, data_filled, data)
print(result)

advisor_c = Advisor()
result = advisor_c.advise("What is the main component in SmoothImpute")
print(result)

# data = np.random.rand(5, 3)
# mask = np.random.rand(5, 3) < 0.2  # 10% missing values
# data[mask] = np.nan


# data_pd = pd.DataFrame(data)

# unif_random_matrix = np.random.uniform(0., 1., size=10)
# binary_random_matrix = 1 * (unif_random_matrix < (1 - 0.2))
# mask = torch.FloatTensor(binary_random_matrix) == 1
# print(mask)

# print(data_pd.to_numpy())
# imputer = Imputer("unimp")
# data_filled = imputer.impute(data_pd)
# print(data_filled)

# llm_path = "/data1/jianweiw/LLM/Imputation/models_hf/llama2_7b/"
# print(data_pd.to_numpy())
# imputer = Imputer("table_gpt", llm_path)
# data_filled = imputer.impute(data_pd)
# print(data_filled)

# llm_path = "/data1/jianweiw/LLM/Imputation/models_hf/Jellyfish-7B/"
# print(data_pd.to_numpy())
# imputer = Imputer("jellyfish", llm_path)
# data_filled = imputer.impute(data_pd)
# print(data_filled)


# import numpy as np
# import random
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from gensim.models import KeyedVectors
# import openai
# # from openai import OpenAI



# # Generate vocabulary using the new API
# def generate_vocabulary_with_llm(column_names, vocab_size=50):
#     """
#     Use the new OpenAI API to generate vocabulary for given column names.

#     Args:
#         column_names (list): List of column names.
#         vocab_size (int): Number of vocabulary words per column.

#     Returns:
#         dict: A dictionary of generated vocabulary for each column.
#     """
#     # API key
#     openai.api_key = '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'
#     client = openai.OpenAI()
#     prompt = (
#         f"Generate a vocabulary of {vocab_size} meaningful words for the following column names in a table. "
#         f"The vocabulary should be semantically relevant to the column names. "
#         f"Here are the column names: {', '.join(column_names)}. \n"
#         f"Return the vocabulary as a list of words for each column name."
#     )

#     # Request to generate vocabulary based on the column names
#     # response = openai.Completion.create(
#     #     model="gpt-4o-mini",  # Make sure to select the correct model
#     #     prompt=prompt,
#     #     max_tokens=300  # Limit the response size
#     # )
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=300
#     )

#     # Parse the generated text into a dictionary (adjust the format if necessary)
#     generated_text = response.choices[0].message.content
#     print(generated_text)
    
#     # Parse the generated text into a dictionary
#     vocab_dict = {}
#     current_column = None
#     words = []
    
#     for line in generated_text.split('\n'):
#         line = line.strip()
#         if not line:
#             continue
            
#         if line.endswith(':'):
#             # Save previous column's words if they exist
#             if current_column is not None:
#                 vocab_dict[current_column] = words
            
#             # Start new column
#             current_column = line[:-1]
#             words = []
#         else:
#             # Add words for current column
#             words.extend([w.strip() for w in line.split(',') if w.strip()])
    
#     # Add the last column
#     if current_column is not None:
#         vocab_dict[current_column] = words
        
#     return vocab_dict

# # Step 2: 构建表格数据生成逻辑
# def generate_tabular_data(column_names, vocab_dict, row_shape, zipf_s=1.5, ngram_order=2, lda_topics=3):
#     """
#     Generate a tabular dataset of textual data using Zipf's Law, n-grams, LDA topic distribution, and word embeddings.

#     Args:
#         column_names (list): 列名列表。
#         vocab_dict (dict): 每列的词汇表，格式为 {column_name: [word1, word2, ...]}。
#         row_shape (int): 表格的行数。
#         zipf_s (float): Zipf's Law 的指数参数。
#         ngram_order (int): n-gram 的阶数。
#         lda_topics (int): LDA 模型的主题数量。

#     Returns:
#         list: 表格数据，以列表形式返回，每行是一个列表。
#     """
#     table = []

#     # Step 2.1: 根据 Zipf's Law 分布生成单词
#     def generate_zipf_vocab(vocab, s, size):
#         word_probs = np.random.zipf(s, len(vocab))
#         word_probs = word_probs / word_probs.sum()  # 归一化
#         zipf_vocab = np.random.choice(vocab, size=size, p=word_probs)
#         return zipf_vocab

#     # Step 2.2: 生成 n-grams
#     def generate_ngrams(vocab, n):
#         ngrams = []
#         for i in range(len(vocab) - n + 1):
#             ngram = " ".join(vocab[i:i+n])
#             ngrams.append(ngram)
#         return ngrams

#     # Step 2.3: LDA 主题分布
#     def generate_lda_topics(vocab, n_topics):
#         vectorizer = CountVectorizer(max_features=500)
#         X_counts = vectorizer.fit_transform([" ".join(vocab)] * row_shape)
#         lda = LatentDirichletAllocation(n_components=int(n_topics), random_state=42)
#         lda.fit(X_counts)
#         topic_distributions = lda.transform(X_counts)
#         topics = np.argmax(topic_distributions, axis=1)
#         return topics

#     # Step 2.4: 生成 word embeddings
#     def generate_word_embeddings(vocab):
#         # 随机生成高维向量表示
#         embedding_size = 50
#         embeddings = {word: np.random.normal(0, 1, embedding_size) for word in vocab}
#         return embeddings

#     # Step 3: 生成表格数据
#     for _ in range(row_shape):
#         row = []
#         for col in column_names:
#             vocab = vocab_dict[col]  # 获取列对应的词汇表
#             zipf_vocab = generate_zipf_vocab(vocab, zipf_s, len(vocab))
#             ngram_vocab = generate_ngrams(zipf_vocab, ngram_order)
#             topics = generate_lda_topics(zipf_vocab, lda_topics)
#             word_embeddings = generate_word_embeddings(zipf_vocab)

#             # 从生成的词汇中随机选择一个单词或短语
#             word = random.choice(zipf_vocab)
#             row.append(f"{word} (topic {random.choice(topics)})")
#         table.append(row)

#     return table

# # Step 4: 调用函数生成数据
# column_names = ["Product", "Review", "Rating"]
# vocab_size = 20
# row_shape = 5

# # Step 4.1: 生成词汇表
# vocab_dict = generate_vocabulary_with_llm(column_names, vocab_size=vocab_size)

# # Step 4.2: 基于词汇表生成表格数据
# generated_table = generate_tabular_data(column_names, vocab_dict, row_shape)

# # 输出结果
# for row in generated_table:
#     print(row)



# from dsrag.create_kb import create_kb_from_file
# from dsrag.llm import OpenAIChatAPI
# from dsrag.reranker import NoReranker


# os.environ["OPENAI_API_KEY"] = "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"
# os.environ["CO_API_KEY"] = "pkkNnTRU9yi8P6iXzk4Rz8GDNTobQM4icZGlywsV"
# llm = OpenAIChatAPI(model='gpt-4o-mini')
# reranker = NoReranker()
# file_path = "/data1/jianweiw/LLM/Imputation/SmoothImpute/src/smoothimpute/data/data.txt"


# search_queries = ["What is the main components in SmoothImpute", "What is SmoothImpute used for"]


# import os
# from lightrag import LightRAG, QueryParam
# from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

# WORKING_DIR = "./datasets"


# if not os.path.exists(WORKING_DIR):
#     os.mkdir(WORKING_DIR)

# rag = LightRAG(
#     working_dir=WORKING_DIR,
#     llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
#     # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
# )

# with open(file_path) as f:
#     rag.insert(f.read())

# Perform naive search
# print(rag.query("What is the main components in SmoothImpute?", param=QueryParam(mode="naive")))

# # Perform local search
# print(rag.query("What is the main components in SmoothImpute?", param=QueryParam(mode="local")))

# # Perform global search
# print(rag.query("What is the main components in SmoothImpute?", param=QueryParam(mode="global")))

# # Perform hybrid search
# print(rag.query("What is the main components in SmoothImpute?", param=QueryParam(mode="hybrid")))

