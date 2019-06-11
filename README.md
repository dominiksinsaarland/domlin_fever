# domlin_fever

Our hand in for the FEVER 2.0 shared task

# requirements


### Requirements
* Python 3.6
* AllenNLP
* TensorFlow
* BERT


# installation

* Download and install Anaconda (https://www.anaconda.com/)
* Create a Python Environment and activate it:
```bash 
    conda create -n domlin_fever python=3.6
    source activate domlin_fever
```

* download BERT cased English model
```bash 
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
unzip cased_L-12_H-768_A-12.zip
rm cased_L-12_H-768_A-12.zip
```

* install requirements
```bash 
pip install requirments.txt
```

* get relevant fever data (that is train, dev and test set)
```bash 
wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
unzip wiki-pages.zip -d wiki_pages
mkdir fever_data
wget -O fever-data/train.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
wget -O fever-data/dev.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl
wget -O fever-data/test.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl 
```



