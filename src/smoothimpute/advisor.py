import numpy as np
import pandas as pd
import torch
from scipy import optimize

import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import KeyedVectors
import openai

import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete

import sys
import os
import inspect

class Advisor:
    def __init__(self):
        self.file_path = inspect.getfile(self.__class__)
        self.file_directory = os.path.dirname(self.file_path)
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/smoothimpute/data')))

        WORKING_DIR = self.file_directory+"/data/memory_bank"


        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)

        self.rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
            # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
        )
        current_directory = os.getcwd()

        file_path = self.file_directory+"/data/data.txt"
        with open(file_path) as f:
            self.rag.insert(f.read())
    
    def advise(self, prompt):

        return self.rag.query(prompt, param=QueryParam(mode="naive"))
