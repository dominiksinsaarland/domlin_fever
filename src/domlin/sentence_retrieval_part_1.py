

import json
import re
import unicodedata
import os
from collections import defaultdict
import numpy as np
import argparse
import sys

def load_wiki_pages(path, docs,return_all_titles=False):
	files = os.listdir(path)
	wiki_docs = {}
	if return_all_titles:
		all_wiki_titles = set()
	for f in files:
		#print (f)
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
		print (len(all_wiki_titles))
		return wiki_docs, all_wiki_titles
	else:
		return wiki_docs

def sentence_retrieval(path_to_infile, path_to_outfile, path_to_wiki):
	docs = set()
	with open(path_to_infile) as infile:
		for line in infile:
			data = json.loads(line)
			claim = data["claim"]
			pred_pages = data["predicted_pages"]
			docs.update(pred_pages)
	docs = load_wiki_pages(path_to_wiki, docs)

	outfile = open(path_to_outfile, "w")
	with open(path_to_infile) as infile:
		for line in infile:
			data = json.loads(line)
			claim = data["claim"]
			pred_pages = data["predicted_pages"]

			for page in pred_pages:
				if page in docs:
					for i, sent in enumerate(docs[page]):
						if len(sent.split()) > 1:
							raw_sent = sent.split("\t")[1]
							outfile.write(claim + "\t" + page + "\t" + raw_sent + "\t" + str(i) + "\t" + str(data["id"]) + "\t" + sent + "\n")
							# claim, doc, sent, sent_id, claim_id, rest = line[0], line[1], line[2], line[3], line[4], line[5:]
	outfile.close()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--infile')
	parser.add_argument('--outfile')
	parser.add_argument('--path_wiki_titles')
	args = parser.parse_args()
	sentence_retrieval(args.infile, args.outfile, args.path_wiki_titles)

