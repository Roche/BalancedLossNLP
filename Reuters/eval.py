import os
import torch
import pickle
import numpy as np
from torch import nn
from transformers import * 
from tqdm import trange
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

########## Configuration Part 1 ###########
source_dir = './'
prefix = 'reuters' 
suffix = 'rand123'
model_name = 'bert_base'

if model_name == 'bert_base':
    model_checkpoint = os.path.join(source_dir, 'berts', 'bert-base-uncased')
    
data_val=pickle.load(open(os.path.join(source_dir, 'data', 'data_val.'+suffix),'rb'))
data_test=pickle.load(open(os.path.join(source_dir, 'data', 'data_test.'+suffix),'rb'))
labels_ref=pickle.load(open(os.path.join(source_dir, 'data', 'labels_ref.'+suffix),'rb'))
class_freq=pickle.load(open(os.path.join(source_dir, 'data', 'class_freq.'+suffix),'rb'))
num_labels = len(labels_ref)
max_len = 512
batch_size = 32


########## set up ###########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, max_len=max_len)#, use_fast=True)
model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)).to(device)

########## data preprocessing (one-off configuration based on the input data) ###########
from torch.utils.data import Dataset, DataLoader

def preprocess_function(docu):
    labels = [1 if x in docu['labels'] else 0 for x in labels_ref] 
    encodings = tokenizer(docu['text'], truncation=True, padding='max_length')    
    return (torch.tensor(encodings['input_ids']), torch.tensor(encodings['attention_mask']), torch.tensor(labels))

class CustomDataset(Dataset):
    '''Characterizes a dataset for PyTorch'''
    def __init__(self, documents):
        '''Initialization'''
        self.documents = documents

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.documents)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        return preprocess_function(self.documents[index])

validation_dataloader = DataLoader(CustomDataset(data_val), shuffle=False, batch_size=batch_size)
test_dataloader = DataLoader(CustomDataset(data_test), shuffle=False, batch_size=batch_size)  

group_head = [i for i, x in enumerate(class_freq) if x>=35]
group_med = [i for i, x in enumerate(class_freq) if x<35 and x>8]
group_tail = [i for i, x in enumerate(class_freq) if x<=8]
print('Label count for head, med and tail groups', len(group_head), len(group_med), len(group_tail))

model_paras = []
model_dir = os.path.join(source_dir, 'models')
for j, fname in enumerate(os.listdir(model_dir)):
    if not fname.startswith(".") and not os.path.isdir(os.path.join(model_dir, fname)):
        loss_func_name = fname.split("_")[3]
        model_paras.append( (loss_func_name, os.path.join(model_dir, fname)) )
        
res = []
for loss_func_name, model_para in model_paras:
    res_model = []
    model.load_state_dict(torch.load(model_para))
    ########## start training + val ###########
    
    model.eval()
    
    true_labels,pred_labels = [],[]   
    for i, batch in enumerate(validation_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
        # Forward pass
            outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)
    
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        true_labels.append(b_labels)
        pred_labels.append(pred_label)
    
    true_labels_val = [item for sublist in true_labels for item in sublist]
    pred_labels_val = [item for sublist in pred_labels for item in sublist]

    true_labels,pred_labels = [],[]
    for i, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
        # Forward pass
            outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)
    
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        true_labels.append(b_labels)
        pred_labels.append(pred_label)
    
    true_labels_test = [item for sublist in true_labels for item in sublist]
    pred_labels_test = [item for sublist in pred_labels for item in sublist]
    
    
    true_bools = [tl==1 for tl in true_labels_val]
    best_med_th = 0.5
    thresholds = (np.array(range(-10,11))/100)+best_med_th # can be larger if out of boundary
    f1_results_micro = []
    for th in thresholds:
        pred_bools = [pl>th for pl in np.array(pred_labels_val)]
        f1_results_micro.append(f1_score(true_bools,pred_bools,average='micro', zero_division=0))
    best_micro_f1_th = thresholds[np.argmax(f1_results_micro)]
    
    miF_val = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_val)],average='micro', zero_division=0)
    maF_val = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_val)],average='macro', zero_division=0)
    
    true_bools = [tl==1 for tl in true_labels_test]
    miF_test = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_test)],average='micro', zero_division=0)
    maF_test = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_test)],average='macro', zero_division=0)
    res_model.extend([loss_func_name, best_micro_f1_th, miF_val, maF_val, miF_test, maF_test])
    
    # evaluation with same cutoff
    for group_name, group in [('Head',group_head), ('Medium', group_med), ('Tail',group_tail)]:
        true_bools = [[tl[i]==1 for i in group] for tl in true_labels_val]
        pred_labels_sub = [[pl[i] for i in group] for pl in pred_labels_val]
        miF_val = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_sub)],average='micro', zero_division=0)
        maF_val = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_sub)],average='macro', zero_division=0)
        
        true_bools = [[tl[i]==1 for i in group] for tl in true_labels_test]
        pred_labels_sub = [[pl[i] for i in group] for pl in pred_labels_test]
        miF_test = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_sub)],average='micro', zero_division=0)
        maF_test = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_sub)],average='macro', zero_division=0)
        res_model.extend([miF_val, maF_val, miF_test, maF_test])

    print(res_model)
    res.append(tuple(res_model.copy()))
    
import pandas as pd
df_res = pd.DataFrame(res)
df_res.columns = ["Loss Function Name", "Threshold", 
                  "Micro-F1-val-All", "Macro-F1-val-All", "Micro-F1-test-All", "Macro-F1-test-All", 
                  "Micro-F1-val-Head", "Macro-F1-val-Head", "Micro-F1-test-Head", "Macro-F1-test-Head", 
                  "Micro-F1-val-Medium", "Macro-F1-val-Medium", "Micro-F1-test-Medium", "Macro-F1-test-Medium", 
                  "Micro-F1-val-Tail", "Macro-F1-val-Tail", "Micro-F1-test-Tail", "Macro-F1-test-Tail"]
df_res.to_excel(prefix + '_eval.xlsx',index=False)