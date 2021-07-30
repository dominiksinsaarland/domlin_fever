import json
import re
import unicodedata
import os
from collections import defaultdict
import numpy as np
import argparse

def load_wiki_pages(path, docs,return_all_titles=False):
	files = os.listdir(path)
	wiki_docs = {}
	if return_all_titles:
		all_wiki_titles = set()
	for f in files:
		with open(os.path.join(path, f)) as infile:
			for line in infile:
				data = json.loads(line)
				title = data["id"]
				title = unicodedata.normalize("NFC", title)
				if title in docs:
					wiki_docs[title] = data["lines"].split("\n")
				if return_all_titles:
					all_wiki_titles.add(title)
	if return_all_titles:
		return wiki_docs, all_wiki_titles
	else:
		return wiki_docs



def load_wiki_docs(path_to_infile, path_wiki_titles, return_all_titles=False):
	"""
	returns a dictionairy with keys being wiki_titles and values being the document's content
	makes one pass through the infile first to only store the wiki titles which are actually required by the infile
	"""

	# one pass to check which documents are actually required
	docs = set()
	with open(path_to_infile) as infile:
		for line in infile:
			data = json.loads(line)
			if data["verifiable"] ==  "NOT VERIFIABLE":
				continue

			claim = data["claim"]
			label = data["label"]
			evidence = data["evidence"]
			pred_pages = data["predicted_pages"]
			for evid in evidence:
				for item in evid:
					Annotation_ID, Evidence_ID, Wikipedia_URL, sentence_ID = item
					Wikipedia_URL = unicodedata.normalize("NFC", Wikipedia_URL)
					docs.add(Wikipedia_URL)
			docs.update(pred_pages)

	# fetch the documents
	if return_all_titles:
		docs, all_wiki_titles = load_wiki_pages(path_wiki_titles, docs, return_all_titles=True)
		return docs, all_wiki_titles
	else:
		docs = load_wiki_pages(path_wiki_titles, docs)
		return docs


def generate_RTE_train_set(path_infile, path_NEI_evidence, path_NEI_predictions, path_outfile, path_to_wiki):
	claim_labels = {}
	RTE_evidence = defaultdict(list)
	claim_set = set()
	docs = set()
	with open(path_infile) as infile:
		for line in infile:
			data = json.loads(line)
			claim = data["claim"]
			if claim in claim_set:
				continue
			claim_set.add(claim)
			label = data["label"]
			evidence = data["evidence"]
			claim_id = str(data["id"])
			claim_labels[claim_id] = (claim, label)
			#if data["id"] != 137334:
			#	continue
			if data["verifiable"] ==  "NOT VERIFIABLE":
				continue


			evidence = sorted(evidence, key=lambda x: len(x), reverse=True)

			for evid in evidence:
				for item in evid:
					Annotation_ID, Evidence_ID, Wikipedia_URL, sentence_ID = item
					Wikipedia_URL = unicodedata.normalize("NFC", Wikipedia_URL)
					if (Wikipedia_URL, sentence_ID) not in RTE_evidence[claim_id] and len(RTE_evidence[claim_id]) < 5:
						RTE_evidence[claim_id].append((Wikipedia_URL, sentence_ID))


	NEI_evidence = defaultdict(lambda: defaultdict(float))
	with open(path_NEI_evidence) as infile:
		with open(path_NEI_predictions) as preds:
			for pred, line in zip(preds, infile):
				pred = float(pred.strip())
				line = line.strip().split("\t")
				claim, doc, sent, sent_id, claim_id, rest = line[0], line[1], line[2], line[3], line[4], line[5:]
				NEI_evidence[claim_id][(doc, int(sent_id))] = pred

	for claim_id, pred in NEI_evidence.items():
		pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)[:5]
		#pred = [list(p[0]) for p in pred if p[1] > 0]
		pred = [list(p[0]) for p in pred if p[1] > 0]
		if pred:
			RTE_evidence[claim_id] = pred[:5]

	docs = set()
	for claim_id, evid in RTE_evidence.items():
		docs.update([x[0] for x in evid])

	docs = load_wiki_pages(path_to_wiki, docs)
	with open(path_outfile, "w") as outfile:
		for claim_id, evid in RTE_evidence.items():
			if claim_id not in claim_labels:
				continue
			evid_string = ""
			last_wiki_url = ""
			for item in evid:
				doc, sent_id = item
				try:
					sent = docs[doc][sent_id].split("\t")[1]
				except:
					continue
				if doc == last_wiki_url:
					evid_string = evid_string + " " + sent
				else:
					evid_string = evid_string + " " + sent
					#evid_string = evid_string + " " + doc + " : " + sent
				last_wiki_url = doc
			claim, label = claim_labels[claim_id]
			if evid_string:
				outfile.write(claim + "\t" + evid_string + "\t" + label + "\t" + claim_id + "\n")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--infile')
	parser.add_argument('--outfile')
	parser.add_argument('--path_wiki_titles')
	parser.add_argument('--NEI_evidence')
	parser.add_argument('--NEI_predictions')
	args = parser.parse_args()
	generate_RTE_train_set(args.infile, args.NEI_evidence, args.NEI_predictions, args.outfile, args.path_wiki_titles)


