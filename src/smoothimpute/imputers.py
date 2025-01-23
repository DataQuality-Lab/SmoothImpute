import numpy as np
import pandas as pd

class Imputer:
    def __init__(self, method, cuda=None, llm_path="/data1/jianweiw/LLM/Imputation/models_hf/llama2_7b/"):
        self.method = method
        self.cuda = cuda
        self.llm_path = llm_path

    def impute(self, data):
        if self.method == 'mean':
            from .imputer_methods.mean import mean_imputation
            return mean_imputation(data.to_numpy())
        elif self.method == 'mice':
            from .imputer_methods.mice import mice_imputation
            return mice_imputation(data.to_numpy())
        elif self.method == 'knn':
            from .imputer_methods.knn import knn_imputation
            return knn_imputation(data.to_numpy())
        elif self.method == 'gain':
            from .imputer_methods.gain import gain_imputation
            return gain_imputation(data.to_numpy(), self.cuda)
        elif self.method == 'vgain':
            from .imputer_methods.vgain import vgain_imputation
            return vgain_imputation(data.to_numpy(), self.cuda)
        elif self.method == 'tdm':
            from .imputer_methods.tdm import tdm_imputation
            return tdm_imputation(data.to_numpy(), self.cuda)
        elif self.method == 'ginn':
            from .imputer_methods.ginn import ginn_imputation
            return ginn_imputation(data.to_numpy())
        elif self.method == 'miracle':
            from .imputer_methods.miracle import miracle_imputation
            return miracle_imputation(data.to_numpy())
        elif self.method == 'missforest':
            from .imputer_methods.missforest import missforest_imputation
            return missforest_imputation(data.to_numpy())
        elif self.method == 'xgboost':
            from .imputer_methods.xgboost import xgb_imputation
            return xgb_imputation(data.to_numpy())
        elif self.method == 'matrix_factorization':
            from .imputer_methods.matrix_factorization import mf_imputation
            return mf_imputation(data.to_numpy())
        elif self.method == 'softimpute':
            from .imputer_methods.softimpute import si_imputation
            return si_imputation(data.to_numpy())
        elif self.method == 'miwae':
            from .imputer_methods.miwae import miwae_imputation
            return miwae_imputation(data.to_numpy())
        elif self.method == 'em':
            from .imputer_methods.em import em_imputation
            return em_imputation(data.to_numpy())
        elif self.method == 'grape':
            from .imputer_methods.grape import grape_imputation
            return grape_imputation(data.to_numpy())
        elif self.method == 'igrm':
            from .imputer_methods.igrm import igrm_imputation
            return igrm_imputation(data.to_numpy())
        elif self.method == 'nomi':
            from .imputer_methods.nomi import nomi_imputation
            return nomi_imputation(data.to_numpy())
        elif self.method == 'remasker':
            from .imputer_methods.remasker import remasker_imputation
            return remasker_imputation(data.to_numpy())
        elif self.method == 'dfms':
            from .imputer_methods.dfms import dfms_imputation
            return dfms_imputation(data, self.llm_path)
        elif self.method == 'jellyfish':
            from .imputer_methods.jellyfish import jellyfish_imputation
            return jellyfish_imputation(data, self.llm_path)
        elif self.method == 'table_gpt':
            from .imputer_methods.table_gpt import table_gpt_imputation
            return table_gpt_imputation(data, self.llm_path)
        elif self.method == 'unimp':
            from .imputer_methods.unimp import unimp_imputation
            return unimp_imputation(data.to_numpy())
        else:
            raise ValueError(f"Unknown imputation method: {self.method}")
