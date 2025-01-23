# Some modules are from https://github.com/maxiaoba/GRAPE
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.gen_imp import get_gen_imp
from models.imputaion_model import LinearHead, LLMHead
from utils import produce_NA, get_main_device, compute_LLM_generation_metrics
import time
import matplotlib.pyplot as plt
from data_loader import load_data
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import copy
from tqdm import tqdm

class HyperBatch:
    def __init__(self, train_hyper_node, hyperedge, train_ve_affiliation, train_labels, batch, train_tokens_emb):
        self.train_hyper_node = train_hyper_node
        self.hyperedge = hyperedge
        self.train_ve_affiliation = train_ve_affiliation
        self.train_labels = train_labels
        self.batch = batch
        self.train_tokens_emb = train_tokens_emb

    @staticmethod
    def from_data_list(ids, train_hyper_node_all, hyperedge_all, train_ve_affiliation_all, train_labels_all, train_tokens_emb_all):
        batch_train_hyper_node = []
        batch_hyperedge = []
        batch_train_ve_affiliation = []
        batch_train_labels = []
        batch_indicator = []
        batch_train_tokens_emb = []

        cumulative_edge = 0

        for i in range(len(ids)):

            num_edge = hyperedge_all[ids[i]].size(0)
            num_node = train_hyper_node_all[ids[i]].size(0)
            # hyper_node
            batch_train_hyper_node.append(train_hyper_node_all[ids[i]][:int(num_node/2)])
            # batch_train_hyper_node.append(train_hyper_node_all[ids[i]])

            # train_tokens_emb, LinearHead does not have train_tokens_emb
            if len(train_tokens_emb_all)>0:
                batch_train_tokens_emb.append(train_tokens_emb_all[ids[i]])

            # hyper_node
            batch_hyperedge.append(hyperedge_all[ids[i]])

            train_ve_affiliation = train_ve_affiliation_all[ids[i]][:, :int(num_node/2)] + cumulative_edge
            # train_ve_affiliation = train_ve_affiliation_all[ids[i]]+ cumulative_edge
            
            batch_train_ve_affiliation.append(train_ve_affiliation)

            batch_train_labels.append(train_labels_all[ids[i]])

            batch_indicator.append(torch.full((num_edge,), i, dtype=torch.long))

            cumulative_edge += num_edge


        train_hyper_node = torch.cat(batch_train_hyper_node, dim=0)
        hyperedge = torch.cat(batch_hyperedge, dim=0)
        train_ve_affiliation = torch.cat(batch_train_ve_affiliation, dim=1)
        train_labels = torch.cat(batch_train_labels, dim=0)
        batch = torch.cat(batch_indicator)
        if len(batch_train_tokens_emb) > 0:
            train_tokens_emb = torch.cat(batch_train_tokens_emb, dim=0)
        else:
            train_tokens_emb = []

        # undirected
        train_ve_affiliation_reverse = train_ve_affiliation[[1, 0], :]
        train_ve_affiliation = torch.cat([train_ve_affiliation, train_ve_affiliation_reverse], dim=1)
        train_hyper_node = torch.cat([train_hyper_node, train_hyper_node], dim=0)

        return HyperBatch(train_hyper_node, hyperedge, train_ve_affiliation, train_labels, batch, train_tokens_emb)

    def to(self, device):
        self.train_hyper_node = self.train_hyper_node.to(device)
        self.hyperedge = self.hyperedge.to(device)
        self.train_ve_affiliation = self.train_ve_affiliation.to(device)
        self.train_labels = self.train_labels.to(device)
        self.batch = self.batch.to(device)
        return self


# Generate Imputation Through Auto-Regressive
def generate_impute(args, embedding, impute_model, test_ve_affiliation, lm_model, tokenizer, x_text_test, max_new_tokens=16):

    impute_model.eval()
    lm_model.eval()

    batch_size = len(x_text_test)

    if batch_size == 0:
        return [], []

    inputs = tokenizer(x_text_test, padding=True, truncation=True, return_tensors="pt")
    device = get_main_device(lm_model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    new_text = []

    with torch.no_grad():
        generated = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        for _ in range(max_new_tokens):
            generated = generated.to(device)
            attention_mask = attention_mask.to(device)

            outputs = lm_model.model(
                input_ids=generated,
                attention_mask=attention_mask,
                return_dict=True
            )

            hidden_states = outputs.last_hidden_state.to(f'cuda:{args.device}')
            
            logits = impute_model([embedding[test_ve_affiliation[0]], embedding[test_ve_affiliation[1]]], hidden_states)

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            generated = torch.cat([generated, next_token.unsqueeze(-1).to(device)], dim=-1)
            new_text.append(next_token.unsqueeze(-1).to(device))
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device)], dim=-1)
            
            if all(next_token == tokenizer.eos_token_id):
                break
    
    # Concatenate new_text tensors along the sequence dimension
    new_text_tensor = torch.cat(new_text, dim=1)
    
    return [tokenizer.decode(gen.squeeze(), skip_special_tokens=True) for gen in new_text_tensor], [tokenizer.decode(gen, skip_special_tokens=True) for gen in generated]

def plot(data, figure_name):

    plt.figure(figsize=(10, 6))
    plt.plot(data)

    min_value = min(data)
    last_value = data[-1]
    min_index = data.index(min_value)
    last_index = len(data) - 1

    plt.annotate(f'Min: {min_value}', 
                 xy=(min_index, min_value), 
                 xytext=(0.05, 0.95), 
                 textcoords='axes fraction',
                 arrowprops=dict(facecolor='blue', shrink=0.05))

    plt.annotate(f'Last: {last_value}', 
                 xy=(last_index, last_value), 
                 xytext=(0.95, 0.95), 
                 textcoords='axes fraction',
                 ha='right',
                 arrowprops=dict(facecolor='red', shrink=0.05))

    plt.title(figure_name)
    plt.xlabel('Index')
    plt.ylabel('Value')

    plt.savefig("./figures/"+figure_name+".png")
    plt.close()



def train_model(args, xmiss, device=torch.device('cpu')):
    
    lm_model, tokenizer, hyperedge, train_hyper_node, train_ve_affiliation, train_labels, test_hyper_node, test_ve_affiliation, test_labels, dataset, chunk_map, train_tokens_emb_LLM, test_node_text = load_data(args, xmiss)
    
    model = get_gen_imp(hyperedge[0].shape[1], train_hyper_node[0].shape[1], args).to(device)
    
    if args.header_type == "Linear":
        # impute_hiddens = list(map(int,args.impute_hiddens.split('_')))
        impute_hiddens = hyperedge[0].shape[1]
        input_dim = args.hyperedge_dim_hidden * 2
        output_dim = 1
        impute_model = LinearHead(input_dim, output_dim,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout).to(device)
        test_labels_all = [item.clone().detach() for item in test_labels]

        if args.load_model_name != "None":
            model.load_state_dict(torch.load(f"./saved_models/llm_gnn_model_{args.load_model_name}.pth"))
            impute_model.load_state_dict(torch.load(f"./saved_models/llm_impute_model_{args.load_model_name}.pth"))

    elif args.header_type == "LLM":
        impute_hiddens = hyperedge[0].shape[1]
        input_dim = args.hyperedge_dim_hidden * 2
        output_dim = args.vocab_size # vocab_size for LlamaLite
        impute_model = LLMHead(input_dim, output_dim,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout,
                            relation_type=args.relation_type)
        
        impute_model.lm_head.weight.data = lm_model.lm_head.weight.data.clone()
        if lm_model.lm_head.bias is not None:
            impute_model.lm_head.bias.data = lm_model.lm_head.bias.data.clone()
        else:
            impute_model.lm_head.bias.data = torch.zeros_like(impute_model.lm_head.bias.data)
        impute_model = impute_model.to(device)
        test_labels_all = [copy.deepcopy(item) for item in test_labels]

        if args.load_model_name != "None":
            model.load_state_dict(torch.load(f"./saved_models/llm_gnn_model_{args.load_model_name}.pth"))
            impute_model.load_state_dict(torch.load(f"./saved_models/llm_impute_model_{args.load_model_name}.pth"))
    
    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters())
    
    # print(model)
    # print(impute_model)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params_impute = [p for p in impute_model.parameters() if p.requires_grad]

    
    # print('total trainable params in GNN model:', sum(p.numel() for p in trainable_params))
    # print('total trainable params in impute model:', sum(p.numel() for p in trainable_params_impute))
        

    filter_fn = filter(lambda p : p.requires_grad, trainable_parameters)
    # optimizer = torch.optim.AdamW(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    
    train_hyper_node_all = [item.clone().detach() for item in train_hyper_node]
    hyperedge_all = [item.clone().detach() for item in hyperedge]
    train_ve_affiliation_all = [item.clone().detach() for item in train_ve_affiliation]
    train_labels_all = [item.clone().detach() for item in train_labels]
    test_hyper_node_all = [item.clone().detach() for item in test_hyper_node]
    test_ve_affiliation_all = [item.clone().detach() for item in test_ve_affiliation]
    train_tokens_emb_all = [item.clone().detach() for item in train_tokens_emb_LLM]

    start_time = time.time()
    loss_all = [[] for i in range(len(dataset))]
    rmse_all = [[] for i in range(len(dataset))]
    mae_all = [[] for i in range(len(dataset))]

    # print(chunk_map)
    train_ids = [i for i in range(len(chunk_map))]
    train_loader = DataLoader(train_ids, batch_size=args.chunk_batch, shuffle=False)
    train_loss = 0

    for epoch in tqdm(range(args.epochs)):
    # for epoch in range(args.epochs):
    
        for ids in train_loader:
            batch = HyperBatch.from_data_list(ids, 
                        train_hyper_node_all, hyperedge_all, 
                        train_ve_affiliation_all, train_labels_all, train_tokens_emb_all)
            train_hyper_node = batch.train_hyper_node.to(device)
            hyperedge = batch.hyperedge.to(device)
            train_ve_affiliation = batch.train_ve_affiliation.to(device)
            train_labels = batch.train_labels.to(device)
            
            model.train()
            impute_model.train()
            optimizer.zero_grad()
            
            known_mask = produce_NA(train_hyper_node[:int(train_hyper_node.shape[0]/2)], p_miss=1-args.known, mecha="Random")
            known_mask_dup = torch.cat((known_mask, known_mask), dim=0)
            known_hyper_node = train_hyper_node.clone().detach()
            known_ve_affiliation = train_ve_affiliation.clone().detach()
            known_hyper_node = known_hyper_node[known_mask_dup]
            known_ve_affiliation = known_ve_affiliation[:,known_mask_dup]

            embedding, hyper_node = model(hyperedge, known_hyper_node, known_ve_affiliation)
            

            if args.header_type == "Linear":
                train_tokens_emb = batch.train_tokens_emb
                pred = impute_model([embedding[train_ve_affiliation[0, :int(train_hyper_node.shape[0]/2)]], embedding[train_ve_affiliation[1, :int(train_hyper_node.shape[0]/2)]]], train_tokens_emb)
                pred_train = pred[:int(train_hyper_node.shape[0] / 2),0]
                label_train = train_labels
                huber_loss = torch.nn.HuberLoss(delta=1)  
                loss = huber_loss(pred_train, label_train)
            elif args.header_type == "LLM":
                train_tokens_emb = batch.train_tokens_emb.to(device)
                # print(f"the shape of train_tokens_emb is : {train_tokens_emb.shape}, the shape of embedding is : {embedding.shape}, the shape of train_ve_affiliation is : {train_ve_affiliation.shape}, the shape of train_labels is : {train_labels.shape}")
                pred = impute_model([embedding[train_ve_affiliation[0, :int(train_hyper_node.shape[0]/2)]], embedding[train_ve_affiliation[1, :int(train_hyper_node.shape[0]/2)]]], train_tokens_emb)
                pred_train = pred[:int(train_hyper_node.shape[0] / 2)]
                label_train = train_labels
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pred_train.view(-1, pred_train.size(-1)), label_train.view(-1))
            
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

        
        if (epoch+1) % args.eval_epoch_gap ==0: 
            model.eval()
            impute_model.eval()
            
            with torch.no_grad():
                for k in range(len(dataset)):
                    dataset_chunk = [item==k for item in chunk_map]
                    pred_test_all = []
                    label_test_all = []
                    query_test_all = []
                    affiliation_all = []

                    for i in range(len(chunk_map)):
                        if not dataset_chunk[i]:
                            continue
                        train_hyper_node = train_hyper_node_all[i].to(device)
                        hyperedge = hyperedge_all[i].to(device)
                        train_ve_affiliation = train_ve_affiliation_all[i].to(device)
                        
                        test_hyper_node = test_hyper_node_all[i].to(device)
                        test_ve_affiliation = test_ve_affiliation_all[i].to(device)
                        test_labels = test_labels_all[i]

                        embedding, hyper_node = model(hyperedge, train_hyper_node, train_ve_affiliation)
                        if args.header_type == "Linear":
                            pred = impute_model([embedding[test_ve_affiliation[0], :], embedding[test_ve_affiliation[1], :]], token_emb=[])

                            affiliation_all.append(test_ve_affiliation)

                            pred_test_all.append(pred[:int(test_hyper_node.shape[0] / 2),0])
                            label_test_all.append(test_labels.to(device))

                    if args.header_type == "Linear":
                        pred_test = torch.cat(pred_test_all)
                        affiliation = torch.cat(affiliation_all, dim=1)
                        label_test = torch.cat(label_test_all)
                        mse = F.mse_loss(pred_test, label_test)
                        test_rmse = np.sqrt(mse.item())
                        l1 = F.l1_loss(pred_test, label_test)
                        test_l1 = l1.item()
                        
                        
                        loss_all[k].append(train_loss)
                        rmse_all[k].append(test_rmse)
                        mae_all[k].append(test_l1)

    cnt = 0
    # print(affiliation.shape, pred_test.shape)
    for i in range(xmiss.shape[0]):
        for j in range(xmiss.shape[1]):
            if np.isnan(xmiss.iloc[i, j]):
                if affiliation[0][cnt] == i and affiliation[1][cnt]-xmiss.shape[0] == j:
                    xmiss.iloc[i, j] = pred_test[cnt].item()
                cnt+=1
    

    return xmiss             
    

    