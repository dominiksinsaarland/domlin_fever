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

* Download NLTK Punkt Tokenizer
```bash
    python -c "import nltk; nltk.download('punkt')"
```


* get relevant fever data (that is train, dev and test set)
```bash 
wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
unzip wiki-pages.zip -d fever_data/wiki_pages
mkdir fever_data
mkdir fever_models
wget -O fever_data/train.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
wget -O fever_data/dev.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl
wget -O fever_data/test.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl 
```


* document retrieval module

We simply used the document retrieval module from team athene in the last fever shared task (https://github.com/UKPLab/fever-2018-team-athene/blob/master/README.md).

I implemented a small wrapper around that module because we don't need the whole codebase.

First, run document retrieval on train, dev and test. Careful, this takes a while!


```bash 
PYTHONPATH=src python src/scripts/athene/doc_retrieval_athene.py --database none --infile fever_data/train.jsonl --outfile fever_data/train.documents_retrieved.jsonl --path_wiki_titles fever_data/wiki_pages
PYTHONPATH=src python src/scripts/athene/doc_retrieval_athene.py --database none --infile fever_data/dev.jsonl --outfile fever_data/dev.documents_retrieved.jsonl --path_wiki_titles fever_data/wiki_pages
PYTHONPATH=src python src/scripts/athene/doc_retrieval_athene.py --database none --infile fever_data/test.jsonl --outfile fever_data/test.documents_retrieved.jsonl --path_wiki_titles fever_data/wiki_pages
```

* first sentence retrieval module

We implemented a hierarchical sentence retrieval approach where we try to find evidence in the documents retrieved by the athene module.

to train the model:

```bash 


# generate the training set
python src/generate_training_data/generate_sentence_retrieval_part_1_data.py --infile fever_data/train.documents_retrieved.jsonl --outfile fever_data/sentence_retrieval_1_training_set.tsv --path_wiki_titles fever_data/wiki_pages

# generate dev set sentences
python src/domlin/sentence_retrieval_part_1.py --infile fever_data/dev.documents_retrieved.jsonl --outfile fever_data/sentence_retrieval_1_dev_set.tsv --path_wiki_titles fever_data/wiki_pages


# train the model (maybe set CUDA_VISIBLE_DEVICES and nohup, takes a while)
python src/domlin/run_fever.py --task_name=ir --do_train=true --do_eval=false --do_predict=true --path_to_train_file=fever_data/sentence_retrieval_1_training_set.tsv --vocab_file=cased_L-12_H-768_A-12/vocab.txt --bert_config_file=cased_L-12_H-768_A-12/bert_config.json --output_dir=fever_models/sentence_retrieval_part_1 --max_seq_length=128 --do_lower_case=False --learning_rate=2e-5 --train_batch_size=32 --num_train_epochs=2 --init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt --use_hingeloss=yes --negative_samples=4 --file_test_results=train_sentences_hinge_loss.tsv --prediction_file=train_sentences_for_hingeloss.tsv


```






