# BalancedLossNLP

The Pytorch implementation for paper 

"*Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution*"

by Yi Huang, Buse Giledereli, Abdullatif Koksal, Arzucan Ozgur and Elif Ozkirimli.

[[Paper (ACL Anthology)]](https://aclanthology.org/2021.emnlp-main.643/)
[[Paper (arXiv)]](https://arxiv.org/abs/2109.04712)

## Datasets

### Reuters-21578

The Aptemod version of the Reuters-21578 benchmark corpus is used for the experiments. This version can be downloaded from https://www.kaggle.com/nltkdata/reuters, which is a simplified format of the dataset that can be downloaded from http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html. This data was pre-processed by converting each news document to a JSON list element with the properties: "labels" and "text". The original train and test split was used, there are 7769 documents in the training set and 3019 documents in the test set. 

The dataset has 90 labels and a long-tailed distribution of the labels. The most common label occurs in 3964 documents whereas the least common label occurs in only 5 documents. There are approximately 1.24 labels per document on average.

### PubMed-BioASQ

The PubMed dataset comes from the [BioASQ Challenge](http://participants-area.bioasq.org/datasets/) (License Code: 8283NLM123) providing PubMed articles with titles and abstracts, that have been manually labelled for Medical Subject Headings (MeSH). This version can be downloaded from the Training v.2021 (txt version). The 224,897 articles published during 2020 and 2021 are used. The data was re-formatted so that the "title" and "abstractText" fields are used as the input , and "meshMajor" field that contains a list of MeSH terms is used as the output.

The dataset has 18211 labels. There are approximately 12.3 labels per document on average.

## Quick start

### Environment setup

- Python >=3.7.6
- Packages
  ```
  pip3 install -r requirements.txt
  ```

### Data preparation and descriptive analysis

#### Reuters-21578
```
cd Reuters
mkdir data berts logs models

# download train_data.json and test_data.json
# run dataset_prep.ipynb
# run dataset_analysis.ipynb
```

#### PubMed-BioASQ
```
cd PubMed
mkdir data berts logs models

# download data2020.json and data2021.json
# run dataset_prep.ipynb
# run dataset_analysis.ipynb
```


### Training

For each dataset and each loss function
```
python train.py {loss_function_name}

# For example, to train a model on Reuters with DBloss
cd Reuters
python train.py DBloss
# or if you would keep it running in the background
nohup python train.py DBloss >> logs/Reuters-DBloss.log 2>&1&
```

The best model will be saved in the `models` folder and messages will be logged in the `logs` folder.

### Testing

For each dataset, after all the necessary models have been generated

```
python eval.py
```

This will generate a `*._eval.xlsx` file with F1 scores.


### SVM experiments

For each dataset, there is also a Python script for SVM model performance, concatenating Preprocessing/Training/Testing steps. Please note it may require a high CPU usage and you can adjust `n_jobs` to fit the environment.
```
python svm-ovr-tfidf.py
# or if you would keep it running in the background
nohup python svm-ovr-tfidf.py >> logs/Reuters-SVM.log 2>&1&
```

## Citation
```
@inproceedings{huang2021balancing,
  title={Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution},
  author={Huang, Yi and Giledereli, Buse and Koksal, Abdullatif and Ozgur, Arzucan and Ozkirimli, Elif},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2021}
}
```

## License 
The usage is under the [CC-BY-NC license](https://creativecommons.org/licenses/by-nc/4.0/) and restricted to non-commercial research and educational purposes.
