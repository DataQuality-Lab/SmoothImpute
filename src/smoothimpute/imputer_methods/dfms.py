import numpy as np
# from sklearn.impute import KNNImputer

"""Run inference."""
import pandas as pd
import numpy as np
import random
from typing import List

from transformers import AutoTokenizer
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time

class LLaMAWithCustomHead(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.custom_head = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.vocab_size)
        # self.custom_head.to(torch.float16)

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.custom_head.weight)
        nn.init.zeros_(self.custom_head.bias)

    def forward(self, input_ids, attention_mask=None, labels=None, original=False):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        logits = outputs.logits

        return logits, 0

def generate_translation(model, tokenizer, source_text, max_length=1024, max_new_tokens=32, original=False):
    inputs = tokenizer(source_text, return_tensors='pt', max_length=max_length, truncation=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        generated = inputs['input_ids']
        
        for _ in range(max_new_tokens):
            logits, _ = model(generated, attention_mask=inputs['attention_mask'], original=original)
            
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones((1, 1), device=device)], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Only decode the newly generated tokens
    new_tokens = generated[0, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def load_custom_llama_model(model_directory):
    base_model = AutoModelForCausalLM.from_pretrained(model_directory, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_directory, device_map="auto")

    model = LLaMAWithCustomHead(base_model)
    

    return model, tokenizer

def serialize_table(
    data_table,
    data_m,
    impute_cols: int = 2,
    sep_tok: str = ","
):  
    num_rows, num_cols = data_table.shape
    headers = data_table.columns
    # manual_prompt_flag = False
    # manual_prompt = ""
    serialized_r = []
    manual_prompt = []
    missing_index = []

    # construct the manual prompt
    k_shot = 2
    selected_indices = np.random.choice(np.where(data_m[:, impute_cols] == 1)[0], k_shot, replace=False)
    for i in selected_indices:
        serialized_one_row = []
        for j in range(num_cols):
            if j != impute_cols and data_m[i][j] == 1:
                serialized_one_row.append(f"{headers[j]}: {data_table.iloc[i, j]}")
        serialized_one_row.append(f"what is {headers[impute_cols]}? => {data_table.iloc[i, impute_cols]}")
        manual_prompt.append(". ".join(serialized_one_row))
    
    context = "\n\n ".join(manual_prompt)

    for i in range(num_rows):
       
        if data_m[i][impute_cols] == 1:
            continue

        missing_index.append(i)

        # serialized_one_row = ["Here is a new test sample. Please directly output the value of prediction without additional other words like 'prediction'. The test sample is as : "]
        serialized_one_row = []
        for j in range(num_cols):
            if j == impute_cols:
                continue
            if data_m[i][j] == 1:
                # serialized_one_row.append(f"( {headers[j]} : {data_table.iloc[i, j]} )".lstrip())
                serialized_one_row.append(f"{headers[j]}: {data_table.iloc[i, j]}")
            # else:
                # serialized_one_row.append(f"( {headers[j]} : NAN )".lstrip())
                # serialized_one_row.append(f"{headers[j]}: NAN")
        
        res = f"{sep_tok} ".join(serialized_one_row)
        serialized_r.append(f"{context}\n\n{res}{sep_tok} what is {headers[impute_cols]}? =>")
        
    
    return serialized_r, manual_prompt, missing_index

def dfms_imputation(xmiss, llm_path):
    
    loaded_model, loaded_tokenizer = load_custom_llama_model(llm_path)

    table_current = xmiss.copy()
    data_m = (~xmiss.isna()).to_numpy()
    # print(data_m)


    for column_index in range(table_current.shape[1]):
        if np.sum(data_m[:, column_index]) == data_m.shape[0]:
            continue

        queries, manual_prompt, missing_index = serialize_table(
            table_current, 
            data_m,
            impute_cols = column_index, 
            sep_tok=','
        )

        for j in range(len(queries)):
            output_imputation_ori = generate_translation(loaded_model, loaded_tokenizer, queries[j], original=True)
            xmiss.iloc[missing_index[j], column_index] = output_imputation_ori
            # print(output_imputation_ori)

    return xmiss