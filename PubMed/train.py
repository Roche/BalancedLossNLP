import sys
import os
import torch
import pickle
import json
import numpy as np
from torch import nn
from transformers import *
from tqdm import trange
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

########## Configuration Part 1 ###########
source_dir = './'
prefix = 'pubmed' 
suffix = 'gt2020.rand123'
model_name = 'biobert_base'
loss_func_name = str(sys.argv[1]) # The loss function name will be given as first argument

if model_name == 'biobert_base':
    model_checkpoint = os.path.join(source_dir, 'berts', 'biobert-base-cased-v1.1')
    
data_train=pickle.load(open(os.path.join(source_dir, 'data', 'data_train.'+suffix),'rb'))
data_val=pickle.load(open(os.path.join(source_dir, 'data', 'data_val.'+suffix),'rb'))
labels_ref=pickle.load(open(os.path.join(source_dir, 'data', 'labels_ref.'+suffix),'rb'))
class_freq=pickle.load(open(os.path.join(source_dir, 'data', 'class_freq.'+suffix),'rb'))
train_num=pickle.load(open(os.path.join(source_dir, 'data', 'train_num.'+suffix),'rb'))
num_labels = len(labels_ref)
max_len = 512
lr = 4e-4
epochs = 50
batch_size = 32

########## set up ###########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, max_len=max_len)
model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)).to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight'] 
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,lr=lr) # consider the scale of loss function

########## Configuration Part 2 ###########

from util_loss import ResampleLoss

if loss_func_name == 'BCE':
    loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
                             focal=dict(focal=False, alpha=0.5, gamma=2),
                             logit_reg=dict(),
                             class_freq=class_freq, train_num=train_num)
    
if loss_func_name == 'FL':
    loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(),
                             class_freq=class_freq, train_num=train_num) 
    
if loss_func_name == 'CBloss': #CB
    loss_func = ResampleLoss(reweight_func='CB', loss_weight=5.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(),
                             CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                             class_freq=class_freq, train_num=train_num) 
    
if loss_func_name == 'R-BCE-Focal': # R-FL
    loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                             class_freq=class_freq, train_num=train_num)
    
if loss_func_name == 'NTR-Focal': # NTR-FL
    loss_func = ResampleLoss(reweight_func=None, loss_weight=0.5,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             class_freq=class_freq, train_num=train_num)  

if loss_func_name == 'DBloss-noFocal': # DB-0FL
    loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=0.5,
                             focal=dict(focal=False, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                             class_freq=class_freq, train_num=train_num)
    
if loss_func_name == 'CBloss-ntr': # CB-NTR
    loss_func = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                             class_freq=class_freq, train_num=train_num) 
    
if loss_func_name == 'DBloss': # DB
    loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                             class_freq=class_freq, train_num=train_num)

        
########## data preprocessing (one-off configuration based on the input data) ###########
from torch.utils.data import Dataset, DataLoader

def preprocess_function(docu):
    labels = [1 if x in docu['meshMajor'] else 0 for x in labels_ref] 
    encodings = tokenizer('%s %s' % (docu['title'],docu['abstractText']), truncation=True, padding='max_length')    
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
    
########## start training + val ###########
train_dataloader = DataLoader(CustomDataset(data_train), shuffle=True, batch_size=batch_size)
validation_dataloader = DataLoader(CustomDataset(data_val), shuffle=False, batch_size=batch_size)

best_f1_for_epoch = 0
epochs_without_improvement = 0

for epoch in trange(epochs, desc="Epoch"):
    # Training
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
  
    for _, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        loss = loss_func(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels))
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    # Validation
    model.eval()
    val_loss = 0
    nb_val_steps = 0
    true_labels,pred_labels = [],[]
    
    for _, batch in enumerate(validation_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            # Forward pass
            outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)
            loss = loss_func(b_logit_pred.view(-1,num_labels),b_labels.type_as(b_logit_pred).view(-1,num_labels))
            val_loss += loss.item()
            nb_val_steps += 1
    
            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        true_labels.append(b_labels)
        pred_labels.append(pred_label)
    
    print("Validation loss: {}".format(val_loss/nb_val_steps))

    # Flatten outputs
    true_labels = [item for sublist in true_labels for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]

    # Calculate Accuracy
    threshold = 0.5
    true_bools = [tl==1 for tl in true_labels]
    pred_bools = [pl>threshold for pl in pred_labels]
    val_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')
    val_precision_accuracy = precision_score(true_bools, pred_bools,average='micro')
    val_recall_accuracy = recall_score(true_bools, pred_bools,average='micro')
    
    print('F1 Validation Accuracy: ', val_f1_accuracy)
    print('Precision Validation Accuracy: ', val_precision_accuracy)
    print('Recall Validation Accuracy: ', val_recall_accuracy)

    # Calculate AUC as well
    val_auc_score = roc_auc_score(true_bools, pred_labels, average='micro')
    print('AUC Validation: ', val_auc_score)
    
    # Search best threshold for F1
    best_med_th = 0.5
    micro_thresholds = (np.array(range(-10,11))/100)+best_med_th
    f1_results, prec_results, recall_results = [], [], []
    for th in micro_thresholds:
        pred_bools = [pl>th for pl in pred_labels]
        test_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')
        test_precision_accuracy = precision_score(true_bools, pred_bools,average='micro')
        test_recall_accuracy = recall_score(true_bools, pred_bools,average='micro')
        f1_results.append(test_f1_accuracy)
        prec_results.append(test_precision_accuracy)
        recall_results.append(test_recall_accuracy)

    best_f1_idx = np.argmax(f1_results) #best threshold value

    # Print and save classification report
    print('Best Threshold: ', micro_thresholds[best_f1_idx])
    print('Test F1 Accuracy: ', f1_results[best_f1_idx])

    # Save the model if this epoch gives the best f1 score in validation set
    if f1_results[best_f1_idx] > (best_f1_for_epoch * 0.995):
        best_f1_for_epoch = f1_results[best_f1_idx]
        epochs_without_improvement = 0
        model_dir = os.path.join(source_dir, 'models')
        for fname in os.listdir(model_dir):
            if fname.startswith('_'.join([prefix,model_name,loss_func_name,suffix])):
                os.remove(os.path.join(model_dir, fname))
        torch.save(model.state_dict(), os.path.join(model_dir, '_'.join([prefix,model_name,loss_func_name,suffix,'epoch'])+str(epoch+1)+'para'))
    else:
        epochs_without_improvement += 1
    
    log_dir = os.path.join(source_dir, 'logs')
    # Log all results in validation set with different thresholds
    with open(os.path.join(log_dir, '_'.join([prefix,model_name,loss_func_name,suffix,'epoch'])+str(epoch+1)+'.json'),'w') as f:
        d = {}
        d["f1_accuracy_default"] =  val_f1_accuracy
        d["pr_accuracy_default"] =  val_precision_accuracy
        d["rec_accuracy_default"] =  val_recall_accuracy
        d["auc_score_default"] =  val_auc_score
        d["thresholds"] =  list(micro_thresholds)
        d["threshold_f1s"] =  f1_results
        d["threshold_precs"] =  prec_results
        d["threshold_recalls"] =  recall_results
        json.dump(d, f)
    
    open(os.path.join(log_dir, '_'.join([prefix,model_name,loss_func_name,suffix,'epoch'])+str(epoch+1)+'.tmp'),'w').write('%s %s' % (micro_thresholds[best_f1_idx], f1_results[best_f1_idx]))

    # If 5 epochs pass without improvement consider the model as saturated and exit
    if epochs_without_improvement > 4:
        break
