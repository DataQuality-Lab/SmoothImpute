import numpy as np
import pandas as pd
import torch

from transformers import BertTokenizer, BertModel
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import cosine
from Levenshtein import distance as levenshtein_distance
import numpy as np
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import argparse
import pandas as pd
import time


def binary_sampler(p, rows, cols):
    '''Sample binary random variables.
    
    Args:
        - p: probability of 1
        - rows: the number of rows
        - cols: the number of columns
        
    Returns:
        - binary_random_matrix: generated binary random matrix.
    '''
    unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
    binary_random_matrix = 1*(unif_random_matrix < p)
    
    return binary_random_matrix

def skip_bigrams(text, k=2):
    tokens = text.split()
    skip_bigrams_set = set()
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + k + 1, len(tokens))):
            skip_bigrams_set.add((tokens[i], tokens[j]))
    return skip_bigrams_set

def compute_rouge_s(label, pred):
    pred_bigrams = skip_bigrams(pred)
    label_bigrams = skip_bigrams(label)

    intersection = pred_bigrams.intersection(label_bigrams)
    
    precision = len(intersection) / len(pred_bigrams) if pred_bigrams else 0.0
    recall = len(intersection) / len(label_bigrams) if label_bigrams else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def lcs_length(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    
    return L[m][n]

def compute_rouge_w(label, pred):
    pred_tokens = pred.split()
    label_tokens = label.split()

    lcs_len = lcs_length(pred_tokens, label_tokens)
    
    precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs_len / len(label_tokens) if label_tokens else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def jaccard_sim(str1, str2):
    # 将两个文本分词并转换为集合
    set1 = set(str1.split())
    set2 = set(str2.split())
    
    # 计算交集
    intersection = set1.intersection(set2)
    
    # 计算并集
    union = set1.union(set2)
    
    # 计算Jaccard相似度
    jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0.0
    
    return jaccard_similarity

def cosine_sim_tf(text1, text2):
    # 将文本转换为词频向量
    # print(text1, text2)
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b').fit_transform([text1, text2])
    vectors = vectorizer.toarray()

    # 计算余弦相似度
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

def cosine_sim_tfidf(text1, text2):
    # 将文本转换为TF-IDF向量
    vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b').fit_transform([text1, text2])
    vectors = vectorizer.toarray()

    # 计算余弦相似度
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

def cosine_sim_word_embeddings(text1, text2, model, tokenizer):
    # Tokenize and encode the texts
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
    
    # Use the [CLS] token embedding as the sentence representation
    embedding1 = outputs1.last_hidden_state[:, 0, :].numpy()
    embedding2 = outputs2.last_hidden_state[:, 0, :].numpy()
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(embedding1, embedding2)
    return cosine_sim[0][0]

def compute_LLM_generation_metrics(pred_test_all, label_test_all):
    # Initialize scorers
    rouge_scorer_ins = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    rouge = Rouge()

    bleu_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    rouge_lsum_scores = []
    rouge_w_scores = []
    rouge_s_scores = []
    jaccard_sims = []
    lev_distances = []
    cos_sims = []
    cos_sims_tf = []
    cos_sims_tfidf = []
    cos_sims_word_embeddings = []

    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    for preds, labels in zip(pred_test_all, label_test_all):
        for pred, label in zip(preds, labels):
            # Remove "<eos> " prefix if present
            pred = pred.replace(" <eos>", "").strip()
            # Ensure pred is not an empty string
            if not pred:
                pred = "<eos>"
            pred = pred.replace(" Question", "").strip()
            

            label = label[0] if isinstance(label, list) else label

            label = str(label)
            pred = str(pred)

            chencherry = SmoothingFunction()
            bleu_score = sentence_bleu([label.split()], pred.split(), smoothing_function=chencherry.method1)
            bleu_scores.append(bleu_score)

            # ROUGE Scores
            rouge_scores = rouge_scorer_ins.score(label, pred)
            rouge_1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge_2_scores.append(rouge_scores['rouge2'].fmeasure)
            rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)
            rouge_lsum_scores.append(rouge_scores['rougeLsum'].fmeasure)
            rouge_w_scores.append(compute_rouge_w(label, pred)['f1_score'])
            rouge_s_scores.append(compute_rouge_s(label, pred)['f1_score'])

            # Jaccard Similarity
            # Ensure that both label and pred are lists for Jaccard calculation
            if isinstance(label, str):
                label = [label]
            if isinstance(pred, str):
                pred = [pred]
            
            # Convert labels to binary format for Jaccard calculation
            label_binary = [1 if word in label else 0 for word in pred]
            pred_binary = [1] * len(pred)
            
            jaccard_sims.append(jaccard_sim(label[0], pred[0]))

            # Levenshtein Distance
            lev_distance = levenshtein_distance(label[0], pred[0])  # Use the first element for distance calculation
            lev_distances.append(lev_distance)

            # Cosine Similarity
            label_vec = np.array([1 if char in label[0] else 0 for char in set(label[0] + pred[0])])
            pred_vec = np.array([1 if char in pred[0] else 0 for char in set(label[0] + pred[0])])
            cos_sim = 1 - cosine(label_vec, pred_vec)
            cos_sims.append(cos_sim)

            cos_sims_tf.append(cosine_sim_tf(label[0], pred[0]))
            cos_sims_tfidf.append(cosine_sim_tfidf(label[0], pred[0]))
            cos_sims_word_embeddings.append(cosine_sim_word_embeddings(label[0], pred[0], bert_model, tokenizer))

    # Calculate averages
    avg_bleu = np.mean(bleu_scores)
    avg_rouge_1 = np.mean(rouge_1_scores)
    avg_rouge_l = np.mean(rouge_l_scores)
    avg_rouge_lsum = np.mean(rouge_lsum_scores)
    avg_rouge_w = np.mean(rouge_w_scores)
    avg_rouge_s = np.mean(rouge_s_scores)
    avg_jaccard = np.mean(jaccard_sims)
    avg_levenshtein = np.mean(lev_distances)
    avg_cosine = np.mean(cos_sims)
    avg_cosine_tf = np.mean(cos_sims_tf)
    avg_cosine_tfidf = np.mean(cos_sims_tfidf)
    avg_cosine_word_embeddings = np.mean(cos_sims_word_embeddings)

    return avg_bleu, avg_rouge_1, avg_rouge_l, avg_rouge_lsum, avg_rouge_w, avg_rouge_s, avg_jaccard, avg_levenshtein, avg_cosine, avg_cosine_tf, avg_cosine_tfidf, avg_cosine_word_embeddings


def normalization (data, parameters=None):
    '''Normalize data in [0, 1] range.
    
    Args:
        - data: original data
    
    Returns:
        - norm_data: normalized data
        - norm_parameters: min_val, max_val for each feature for renormalization
    '''

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()
    
    if parameters is None:
    
        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        
        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
            max_val[i] = np.nanmax(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
        
        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                        'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
        
        # For each dimension
        for i in range(dim):
            norm_data[:,i] = norm_data[:,i] - min_val[i]
            norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
        
        norm_parameters = parameters    
    
    return norm_data, norm_parameters


def renormalization (norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.
    
    Args:
        - norm_data: normalized data
        - norm_parameters: min_val, max_val for each feature for renormalization
    
    Returns:
        - renorm_data: renormalized original data
    '''
    
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()
    
    for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
    return renorm_data

#### Accuracy Metrics ####
def MAE(X, X_true, mask):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.
    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.
    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)
    Returns
    -------
        MAE : float
    """
    X_true, norm_parameters = normalization(X_true)
    X, _ = normalization(X, norm_parameters)
    if torch.is_tensor(mask):
        # print("MAE using torch")
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else: # should be an ndarray
        # print("MAE using numpy")
        # mask_ = mask.astype(bool)
        mask_ = ~mask.astype(bool)
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()



def RMSE(X, X_true, mask):
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.
    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.
    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)
    Returns
    -------
        RMSE : float
    """
    X_true, norm_parameters = normalization(X_true)
    X, _ = normalization(X, norm_parameters)

    if torch.is_tensor(mask):
        # print("RMSE using torch")
        mask_ = mask.bool()
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else: # should be an ndarray
        # print("RMSE using numpy")
        mask_ = ~mask.astype(bool)
        # mask_ = mask
        return np.sqrt(((X[mask_] - X_true[mask_])**2).sum() / mask_.sum())



class Evaluator:
    def __init__(self, data_type="number"):
        
        self.data_type = data_type

    def evaluate(self, xmiss, imputed_data, ground_truth):
        data_m = (~xmiss.isna()).to_numpy()

        all_labels = []
        all_preds = []
        if self.data_type == "text":
            for i in range(xmiss.shape[0]):
                for j in range(xmiss.shape[1]):
                    if ~data_m[i][j]:
                        all_preds.append([str(imputed_data.iloc[i, j])])
                        all_labels.append([str(ground_truth.iloc[i, j])])
            avg_bleu, avg_rouge_1, avg_rouge_l, avg_rouge_lsum, avg_rouge_w, avg_rouge_s, avg_jaccard, avg_levenshtein, avg_cosine, avg_cosine_tf, avg_cosine_tfidf, avg_cosine_word_embeddings = compute_LLM_generation_metrics(all_preds, all_labels)
            result = {"bleu": avg_bleu, "rouge_1": avg_rouge_1, "rouge_l": avg_rouge_l, "rouge_lsum": avg_rouge_lsum, "rouge_w": avg_rouge_w, "rouge_s": avg_rouge_s, "jaccard": avg_jaccard, "levenshtein": avg_levenshtein, "cosine": avg_cosine, "cosine_tf": avg_cosine_tf, "cosine_tfidf": avg_cosine_tfidf, "cosine_word_embeddings": avg_cosine_word_embeddings}

        else:
            rmse_result = RMSE(imputed_data.values, ground_truth.values, data_m)
            mae_result = MAE(imputed_data.values, ground_truth.values, data_m)
            result = {"rmse_result": rmse_result, "mae_result": mae_result}

        return result