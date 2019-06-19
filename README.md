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
python src/domlin/run_fever.py --task_name=ir --do_train=true --do_eval=false --do_predict=true \
--path_to_train_file=fever_data/sentence_retrieval_1_training_set.tsv --vocab_file=cased_L-12_H-768_A-12/vocab.txt\
 --bert_config_file=cased_L-12_H-768_A-12/bert_config.json --output_dir=fever_models/sentence_retrieval_part_1 --max_seq_length=128\
 --do_lower_case=False --learning_rate=2e-5 --train_batch_size=32 --num_train_epochs=2 \
--init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt --use_hingeloss=yes --negative_samples=4 \
--file_test_results=fever_data/dev_set_sentences_predicted_part_1.tsv --prediction_file=fever_data/sentence_retrieval_1_dev_set.tsv
```


* second sentence retrieval module

Some of the combined evidence examples in the fever dataset have evidence sentences which are not only conditioned on the claim itself but also previously retrieved evidence.

Consider the example

```bash 
claim: Ryan Gosling has been to a country in Africa.
Evidence 1: He [...] has traveled to Chad , Uganda and eastern Congo [...].
Evidence 2: Chad [...] is a landlocked country in Central Africa
```

The second evidence is hard to retrieve given the claim only because it is only valid evidence since Ryan Gosling has traveled to Chad, whereas a very similar sentence like 
"Sudan is a landlocked country in Africa" would not be valid evidence for the claim. 

Therefore, we trained a second sentence retrieval module which is trained on all train examples containing two evidences (there are not too many examples which have more than two evidence sentences) where a positive example is "claim [SEP] evidence 1 + evidence 2 [SEP]" and a negative example is "claim [SEP] evidence 1 + random sentence [SEP]".

If we retrieve a positive sentence evid_1 in a wiki doc X, we collect all outgoing links in X and classify all sentences "claim [SEP] evid_1 sentence [SEP] for every sentence in the collected documents

```bash 
# generate the training set
python src/generate_training_data/generate_sentence_retrieval_part_2_data.py --infile fever_data/train.documents_retrieved.jsonl --outfile fever_data/sentence_retrieval_2_training_set.tsv --path_wiki_titles fever_data/wiki_pages

# generate dev set sentences
python src/domlin/sentence_retrieval_part_2.py --infile fever_data/train.documents_retrieved.jsonl --outfile fever_data/sentence_retrieval_2_dev_set.tsv --path_wiki_titles fever_data/wiki_pages/ --file_with_sentences_to_be_predicted fever_data/sentence_retrieval_1_dev_set.tsv --predicted_evidence fever_data/dev_set_sentences_predicted_part_1.tsv


# train the model (maybe set CUDA_VISIBLE_DEVICES and nohup, takes a while)
python src/domlin/run_fever.py --task_name=combined_evidence --do_train=true --do_eval=false --do_predict=true \
--path_to_train_file=fever_data/sentence_retrieval_2_training_set.tsv --vocab_file=cased_L-12_H-768_A-12/vocab.txt\
 --bert_config_file=cased_L-12_H-768_A-12/bert_config.json --output_dir=fever_models/sentence_retrieval_part_2 --max_seq_length=256\
 --do_lower_case=False --learning_rate=2e-5 --train_batch_size=16 --num_train_epochs=2 \
--init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt --use_hingeloss=yes --negative_samples=2 \
--file_test_results=fever_data/dev_set_sentences_predicted_part_2.tsv --prediction_file=fever_data/sentence_retrieval_2_dev_set.tsv
```

* RTE (recognize textual entailment) module

We found in preliminary experiments that BERT performs very well on the RTE part of the FEVER dataset and if we take only gold evidence (for the "SUPPORT/REFUTES" and ignoring the "NOT ENOUGH INFORMATION") classes, we get 90%+ accuracy. Therefore, we use our sentence retrieval modules to find evidence for the NEI class, concatenate all the evidence and then fine-tune another BERT checkpoint for the RTE module - this is very similar to the FEVER baseline in (Thorne et al., 2018).

First, we need to find evidence sentences for the NEI class:

```bash 
# generate NEI evidence
python src/domlin/sentence_retrieval_part_1.py --infile fever_data/train.documents_retrieved.jsonl --outfile fever_data/NEI_evidence_1.tsv --path_wiki_titles fever_data/wiki_pages --NEI_evidence True

# predict NEI evidence
CUDA_VISIBLE_DEVICES=0 python src/domlin/run_fever.py --task_name=ir --do_train=false --do_eval=false --do_predict=true \
 --vocab_file=cased_L-12_H-768_A-12/vocab.txt --bert_config_file=cased_L-12_H-768_A-12/bert_config.json \
 --output_dir=fever_models/sentence_retrieval_part_1 --max_seq_length=128 \
 --do_lower_case=False --use_hingeloss=yes --file_test_results=fever_data/NEI_evidence_predicted.tsv --prediction_file=fever_data/NEI_evidence_1.tsv

# generate the RTE training set
python src/generate_training_data/generate_RTE_data.py --infile fever_data/train.documents_retrieved.jsonl --outfile fever_data/RTE_train_set.tsv --path_wiki_titles fever_data/wiki_pages/ --NEI_evidence fever_data/NEI_evidence_1.tsv --NEI_predictions fever_data/NEI_evidence_predicted.tsv 

# generate the RTE dev set

python src/domlin/generate_rte.py --infile fever_data/dev.documents_retrieved.jsonl --outfile fever_data/RTE_dev_set.py --path_evid_1 fever_data/sentence_retrieval_1_dev_set.tsv --path_evid_1_predicted fever_data/dev_set_sentences_predicted_part_1.tsv --path_evid_2 fever_data/sentence_retrieval_2_dev_set.tsv --path_evid_2_predicted fever_data/dev_set_sentences_predicted_part_2.tsv --path_wiki_titles fever_data/wiki_pages

# actually train the model

CUDA_VISIBLE_DEVICES=6 python src/domlin/run_fever.py --task_name=fever --do_train=true --do_eval=true --do_predict=false --vocab_file=cased_L-12_H-768_A-12/vocab.txt --bert_config_file=cased_L-12_H-768_A-12/bert_config.json --max_seq_length=370 --do_lower_case=false --learning_rate=3e-5 --train_batch_size=12 --num_train_epochs=2 --output_dir=fever_models/rte_model --init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt --prediction_file=RTE_dev_set_new_try_v4.tsv --train_file=fever_data/RTE_train_set.tsv


# generate the dev set

# predict dev set
```

* run all at once on testset



