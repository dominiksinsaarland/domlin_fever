#!/bin/sh

path_wiki_title="fever_data/wiki_pages/"
default_cuda_device=0


mkdir tmp

echo "starting document retrieval"

PYTHONPATH=src python src/scripts/athene/doc_retrieval_athene.py --database none --infile $1 --outfile tmp/ir.$(basename $1) --path_wiki_titles $path_wiki_title

echo "starting sentence retrieval part 1"

python src/domlin/sentence_retrieval_part_1.py --infile tmp/ir.$(basename $1) --outfile tmp/sentences_1.$(basename $1) --path_wiki_titles $path_wiki_title


CUDA_VISIBLE_DEVICES=0 python src/domlin/run_fever.py --task_name=ir --do_train=false --do_eval=false --do_predict=true \
 --vocab_file=cased_L-12_H-768_A-12/vocab.txt --bert_config_file=cased_L-12_H-768_A-12/bert_config.json \
 --output_dir=fever_models/sentence_retrieval_part_1 --max_seq_length=128 \
 --do_lower_case=False --use_hingeloss=yes --file_test_results=tmp/sentences_1_predicted.$(basename $1) --prediction_file=tmp/sentences_1.$(basename $1)

echo "starting sentence retrieval part 2"

python src/domlin/sentence_retrieval_part_2.py --infile tmp/ir.$(basename $1) --outfile tmp/sentences_2.$(basename $1) --path_wiki_titles $path_wiki_title --file_with_sentences_to_be_predicted  tmp/sentences_1.$(basename $1) --predicted_evidence tmp/sentences_1_predicted.$(basename $1)


python src/domlin/run_fever.py --task_name=combined_evidence --do_train=false --do_eval=false --do_predict=true \
--vocab_file=cased_L-12_H-768_A-12/vocab.txt --bert_config_file=cased_L-12_H-768_A-12/bert_config.json --output_dir=fever_models/sentence_retrieval_part_2 --max_seq_length=256 \
--do_lower_case=False --use_hingeloss=yes \
--file_test_results=tmp/sentences_2_predicted.$(basename $1) --prediction_file=tmp/sentences_2.$(basename $1)


echo "starting RTE module"


python src/domlin/generate_rte.py --infile tmp/ir.$(basename $1) --outfile tmp/rte.$(basename $1) \
--path_evid_1 tmp/sentences_1.$(basename $1) --path_evid_1_predicted tmp/sentences_1_predicted.$(basename $1) \
--path_evid_2 tmp/sentences_2.$(basename $1) --path_evid_2_predicted tmp/sentences_2_predicted.$(basename $1) --path_wiki_titles $path_wiki_title


python src/domlin/run_fever.py --task_name=fever --do_train=false --do_eval=true --do_predict=false \
--vocab_file=cased_L-12_H-768_A-12/vocab.txt --bert_config_file=cased_L-12_H-768_A-12/bert_config.json \
--max_seq_length=370 --do_lower_case=false --output_dir=fever_models/rte_model --prediction_file=tmp/rte.$(basename $1) --file_test_results=tmp/rte_predicted.$(basename $1)


echo "generating submission"

python src/domlin/generate_submission.py --path_rte_file tmp/rte.$(basename $1) path_rte_predictions tmp/rte_predicted.$(basename $1) \
--outfile $2 --path_evid_1 tmp/sentences_1.$(basename $1) --path_evid_1_predicted tmp/sentences_1_predicted.$(basename $1) \
--path_evid_2 tmp/sentences_2.$(basename $1) --path_evid_2_predicted tmp/sentences_2_predicted.$(basename $1)
