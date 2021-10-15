import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

source_dir = './'
prefix = 'pubmed' 
suffix = 'gt2020.rand123'

## preprocess
data_train=pickle.load(open(os.path.join(source_dir, 'data', 'data_train.'+suffix),'rb'))
data_val=pickle.load(open(os.path.join(source_dir, 'data', 'data_val.'+suffix),'rb'))
data_test=pickle.load(open(os.path.join(source_dir, 'data', 'data_test.'+suffix),'rb'))
labels_ref=pickle.load(open(os.path.join(source_dir, 'data', 'labels_ref.'+suffix),'rb'))

corpus_train = ['%s %s' % (docu['title'],docu['abstractText']) for docu in data_train]
pipe = Pipeline([('count', CountVectorizer(max_features=10000)), # to reduce the matrix size
                 ('tfidf', TfidfTransformer())]).fit(corpus_train)

X_all = pipe.transform(['%s %s' % (docu['title'],docu['abstractText']) for docu in data_train]).toarray()
y_all = np.array([[1 if x in docu['meshMajor'] else 0 for x in labels_ref] for docu in data_train])
pickle.dump((X_all, y_all), open(os.path.join(source_dir, 'data', 'data_train.'+suffix+'.tfidf'),'wb'), protocol=4)

X_all = pipe.transform(['%s %s' % (docu['title'],docu['abstractText']) for docu in data_val]).toarray()
y_all = np.array([[1 if x in docu['meshMajor'] else 0 for x in labels_ref] for docu in data_val])
pickle.dump((X_all, y_all), open(os.path.join(source_dir, 'data', 'data_val.'+suffix+'.tfidf'),'wb'), protocol=4)

X_all = pipe.transform(['%s %s' % (docu['title'],docu['abstractText']) for docu in data_test]).toarray()
y_all = np.array([[1 if x in docu['meshMajor'] else 0 for x in labels_ref] for docu in data_test])
pickle.dump((X_all, y_all), open(os.path.join(source_dir, 'data', 'data_test.'+suffix+'.tfidf'),'wb'), protocol=4)


## train and validate
class_freq=pickle.load(open(os.path.join(source_dir, 'data', 'class_freq.'+suffix),'rb'))
X_train, y_train=pickle.load(open(os.path.join(source_dir, 'data', 'data_train.'+suffix+'.tfidf'),'rb'))
X_val, y_val=pickle.load(open(os.path.join(source_dir, 'data', 'data_val.'+suffix+'.tfidf'),'rb'))
X_test, y_test=pickle.load(open(os.path.join(source_dir, 'data', 'data_test.'+suffix+'.tfidf'),'rb'))

group_head = [i for i, x in enumerate(class_freq) if x>=50]
group_med = [i for i, x in enumerate(class_freq) if x<50 and x>15]
group_tail = [i for i, x in enumerate(class_freq) if x<=15]

res = []

### linear kernel, same class weight
classifier = OneVsRestClassifier(
    LinearSVC(C=1.0, max_iter=10000, random_state=123),
    n_jobs=8)
classifier.fit(X_train, y_train)
y_val_score = classifier.decision_function(X_val)
y_test_score = classifier.decision_function(X_test)


#### no hyper-plane shifting
res_model = []

best_micro_f1_th = 0

miF_val = f1_score(y_val, y_val_score>best_micro_f1_th, average='micro', zero_division=0)
maF_val = f1_score(y_val, y_val_score>best_micro_f1_th, average='macro', zero_division=0)
miF_test = f1_score(y_test, y_test_score>best_micro_f1_th, average='micro', zero_division=0)
maF_test = f1_score(y_test, y_test_score>best_micro_f1_th, average='macro', zero_division=0)
res_model.extend(['one-weight', best_micro_f1_th])
res_model.extend([miF_val, maF_val, miF_test, maF_test])

for group_name, group in [('Head',group_head), ('Medium', group_med), ('Tail',group_tail)]:
    y_val_sub = [[tl[i]==1 for i in group] for tl in y_val]
    y_val_score_sub = [[pl[i] for i in group] for pl in y_val_score]
    miF_val = f1_score(y_val_sub,np.array(y_val_score_sub)>best_micro_f1_th,average='micro', zero_division=0)
    maF_val = f1_score(y_val_sub,np.array(y_val_score_sub)>best_micro_f1_th,average='macro', zero_division=0)

    y_test_sub = [[tl[i]==1 for i in group] for tl in y_test]
    y_test_score_sub = [[pl[i] for i in group] for pl in y_test_score]
    miF_test = f1_score(y_test_sub,np.array(y_test_score_sub)>best_micro_f1_th,average='micro', zero_division=0)
    maF_test = f1_score(y_test_sub,np.array(y_test_score_sub)>best_micro_f1_th,average='macro', zero_division=0)
    
    res_model.extend([miF_val, maF_val, miF_test, maF_test])
    
res.append(tuple(res_model.copy()))

#### hyper-plane shifting optimized on validation data
res_model = []

best_med_th = 0
thresholds = (np.array(range(-10,11))/10)+best_med_th # can be larger if out of boundary
f1_results_micro = [f1_score(y_val, y_val_score>th, average='micro', zero_division=0) for th in thresholds]
best_micro_f1_th = thresholds[np.argmax(f1_results_micro)]

miF_val = f1_score(y_val, y_val_score>best_micro_f1_th, average='micro', zero_division=0)
maF_val = f1_score(y_val, y_val_score>best_micro_f1_th, average='macro', zero_division=0)
miF_test = f1_score(y_test, y_test_score>best_micro_f1_th, average='micro', zero_division=0)
maF_test = f1_score(y_test, y_test_score>best_micro_f1_th, average='macro', zero_division=0)
res_model.extend(['one-weight-shift-hyper-plane', best_micro_f1_th])
res_model.extend([miF_val, maF_val, miF_test, maF_test])

for group_name, group in [('Head',group_head), ('Medium', group_med), ('Tail',group_tail)]:
    y_val_sub = [[tl[i]==1 for i in group] for tl in y_val]
    y_val_score_sub = [[pl[i] for i in group] for pl in y_val_score]
    miF_val = f1_score(y_val_sub,np.array(y_val_score_sub)>best_micro_f1_th,average='micro', zero_division=0)
    maF_val = f1_score(y_val_sub,np.array(y_val_score_sub)>best_micro_f1_th,average='macro', zero_division=0)

    y_test_sub = [[tl[i]==1 for i in group] for tl in y_test]
    y_test_score_sub = [[pl[i] for i in group] for pl in y_test_score]
    miF_test = f1_score(y_test_sub,np.array(y_test_score_sub)>best_micro_f1_th,average='micro', zero_division=0)
    maF_test = f1_score(y_test_sub,np.array(y_test_score_sub)>best_micro_f1_th,average='macro', zero_division=0)
    
    res_model.extend([miF_val, maF_val, miF_test, maF_test])
    
res.append(tuple(res_model.copy()))


### linear kernel, balanced class weight
classifier = OneVsRestClassifier(
    LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=123),
    n_jobs=8)
classifier.fit(X_train, y_train)
y_val_score = classifier.decision_function(X_val)
y_test_score = classifier.decision_function(X_test)

#### no hyper-plane shifting
res_model = []

best_micro_f1_th = 0

miF_val = f1_score(y_val, y_val_score>best_micro_f1_th, average='micro', zero_division=0)
maF_val = f1_score(y_val, y_val_score>best_micro_f1_th, average='macro', zero_division=0)
miF_test = f1_score(y_test, y_test_score>best_micro_f1_th, average='micro', zero_division=0)
maF_test = f1_score(y_test, y_test_score>best_micro_f1_th, average='macro', zero_division=0)
res_model.extend(['balanced-weight', best_micro_f1_th])
res_model.extend([miF_val, maF_val, miF_test, maF_test])

for group_name, group in [('Head',group_head), ('Medium', group_med), ('Tail',group_tail)]:
    y_val_sub = [[tl[i]==1 for i in group] for tl in y_val]
    y_val_score_sub = [[pl[i] for i in group] for pl in y_val_score]
    miF_val = f1_score(y_val_sub,np.array(y_val_score_sub)>best_micro_f1_th,average='micro', zero_division=0)
    maF_val = f1_score(y_val_sub,np.array(y_val_score_sub)>best_micro_f1_th,average='macro', zero_division=0)

    y_test_sub = [[tl[i]==1 for i in group] for tl in y_test]
    y_test_score_sub = [[pl[i] for i in group] for pl in y_test_score]
    miF_test = f1_score(y_test_sub,np.array(y_test_score_sub)>best_micro_f1_th,average='micro', zero_division=0)
    maF_test = f1_score(y_test_sub,np.array(y_test_score_sub)>best_micro_f1_th,average='macro', zero_division=0)
    
    res_model.extend([miF_val, maF_val, miF_test, maF_test])
    
res.append(tuple(res_model.copy()))

#### hyper-plane shifting optimized on validation data
res_model = []

best_med_th = 0
thresholds = (np.array(range(-10,11))/10)+best_med_th # can be larger if out of boundary
f1_results_micro = [f1_score(y_val, y_val_score>th, average='micro', zero_division=0) for th in thresholds]
best_micro_f1_th = thresholds[np.argmax(f1_results_micro)]

miF_val = f1_score(y_val, y_val_score>best_micro_f1_th, average='micro', zero_division=0)
maF_val = f1_score(y_val, y_val_score>best_micro_f1_th, average='macro', zero_division=0)
miF_test = f1_score(y_test, y_test_score>best_micro_f1_th, average='micro', zero_division=0)
maF_test = f1_score(y_test, y_test_score>best_micro_f1_th, average='macro', zero_division=0)
res_model.extend(['balanced-weight-shift-hyper-plane', best_micro_f1_th])
res_model.extend([miF_val, maF_val, miF_test, maF_test])

for group_name, group in [('Head',group_head), ('Medium', group_med), ('Tail',group_tail)]:
    y_val_sub = [[tl[i]==1 for i in group] for tl in y_val]
    y_val_score_sub = [[pl[i] for i in group] for pl in y_val_score]
    miF_val = f1_score(y_val_sub,np.array(y_val_score_sub)>best_micro_f1_th,average='micro', zero_division=0)
    maF_val = f1_score(y_val_sub,np.array(y_val_score_sub)>best_micro_f1_th,average='macro', zero_division=0)

    y_test_sub = [[tl[i]==1 for i in group] for tl in y_test]
    y_test_score_sub = [[pl[i] for i in group] for pl in y_test_score]
    miF_test = f1_score(y_test_sub,np.array(y_test_score_sub)>best_micro_f1_th,average='micro', zero_division=0)
    maF_test = f1_score(y_test_sub,np.array(y_test_score_sub)>best_micro_f1_th,average='macro', zero_division=0)
    
    res_model.extend([miF_val, maF_val, miF_test, maF_test])
    
res.append(tuple(res_model.copy()))


import pandas as pd
df_res = pd.DataFrame(res)
df_res.columns = ["Method", "Threshold", 
                  "Micro-F1-val-All", "Macro-F1-val-All", "Micro-F1-test-All", "Macro-F1-test-All", 
                  "Micro-F1-val-Head", "Macro-F1-val-Head", "Micro-F1-test-Head", "Macro-F1-test-Head", 
                  "Micro-F1-val-Medium", "Macro-F1-val-Medium", "Micro-F1-test-Medium", "Macro-F1-test-Medium", 
                  "Micro-F1-val-Tail", "Macro-F1-val-Tail", "Micro-F1-test-Tail", "Macro-F1-test-Tail"]
df_res.to_excel(prefix + '_eval_svm_tfidf.xlsx',index=False)