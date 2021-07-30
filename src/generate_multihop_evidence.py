from FEVER_class import FEVER_Instance, Wiki_Docs, ALL_instances
from collections import defaultdict
import argparse
import json
import unicodedata
import random
import os
import re

def fetch_documents(db, pages):
	docs = defaultdict(lambda: [])
	for page, lines in db.get_all_doc_lines(pages):
		docs[page] = re.split("\n(?=\d+)", lines)
	return docs

def normalize(text):
	return unicodedata.normalize('NFD', text)

def normalize_text_to_title(text):
	return normalize(text.strip()\
		.replace("(","-LRB-")\
		.replace(")","-RRB-")\
		.replace(" ","_")\
		.replace(":","-COLON-")\
		.replace("[","-LSB-")\
		.replace("]","-RSB-"))


def get_sentences(doc):
	for sent in doc:
		if sent:
			yield sent.split("\t")[1]
		else:
			yield ""

def get_hyperlinks_from_sent(docs, page, sent_id):
	hyperlinks = set()
	for i, hyperlink in enumerate(docs[page][sent_id].split("\t")[2:]):
		if i % 2 == 1:
			item = normalize_text_to_title(hyperlink)
			hyperlinks.add(item)
	return hyperlinks


def load_evidence_first_pass(args):
	retrieved_evidence = defaultdict(lambda: defaultdict(float))
	with open(args.input_file) as infile:
		for line in infile:
			line = line.strip().split("\t")
			claim_id, claim, doc, sent_id, evid_sent, label_evid, pred= line
			pred = float(pred)
			if pred > 0:
				retrieved_evidence[int(claim_id)][(doc, int(sent_id))] = pred

	return retrieved_evidence


def expand_evidence_sentences(args):
	instances = ALL_instances(args.athene_document_retrieval_train, args.athene_document_retrieval_dev, args.db_file)
	retrieved_evidence = load_evidence_first_pass(args)
	if args.mode == "train":
		all_examples = instances.train_examples
	elif args.mode == "dev":
		all_examples = instances.dev_examples
	with open(os.path.join(args.outfile_name), "w") as outfile_IR_2:
		for example in all_examples:
			if example.id in retrieved_evidence:
				results = set()
				pages = set([x[0] for x in retrieved_evidence[example.id]])
				docs = fetch_documents(instances.db, pages)
				for (page, sent_ID) in retrieved_evidence[example.id]:
					sent = docs[page][sent_ID].split("\t")[1]
					# current
					current_sentence_A = (page, str(sent_ID), sent)
					hyperlinks = get_hyperlinks_from_sent(docs, page, sent_ID)
					hyperlink_docs = fetch_documents(instances.db, hyperlinks)
					for hyperlink, sents in hyperlink_docs.items():
						for sent_ID, sent in enumerate(get_sentences(sents)):
							if sent_ID > 2:
								break
							if not sent:
								continue
							if len(sent) <= 3:
								continue
							result = (str(example.id), example.claim, hyperlink, str(sent_ID), sent, "no_evidence") + current_sentence_A
							results.add(result)
				for result in results:
					outfile_IR_2.write("\t".join(result) + "\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--db_file", required=True, type=str, "path to wikipedia database")
	parser.add_argument("--athene_document_retrieval_train", required=True, type=str, help="path to athene document retrieval output train")
	parser.add_argument("--athene_document_retrieval_dev", required=True, type=str, help="path to athene document retrieval output dev")
	parser.add_argument("--input_file", default="", type=str, help="path to predicted sentences")
	parser.add_argument("--mode", default="dev", type=str)
	parser.add_argument("--outfile_name", default="", type=str)
	args = parser.parse_args()
	expand_evidence_sentences(args)


