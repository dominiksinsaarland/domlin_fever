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

def sample_negative_example(title, docs, block_set):
	examples = []

	for i, sent in enumerate(docs[title]):
		sent = sent.split("\t")
		try:
			sent_id = sent[0]
			sent = sent[1]
			if (title, i) not in block_set and len(sent.split()) > 5:
				examples.append((title, i))
		except Exception as e:
			print (str(e))
	if examples:
		return examples
	return -1


def generate_sentence_retrieval_training_set(path_to_infile, outfile, path_wiki_titles):

	"""
	generates the training set for the first sentence retrieval module
	params: path_to_infile: shared task train file with predicted documents, outfile: name of the outfile, path_wiki_titles: path to where the wiki documents are stored
	"""
	outfile = open(outfile, "w")
	docs = load_wiki_docs(path_to_infile, path_wiki_titles)

	
	with open(path_to_infile) as infile:
		for line in infile:
			data = json.loads(line)
			claim = data["claim"]
			label = data["label"]
			evidence = data["evidence"]
			pred_pages = data["predicted_pages"]

			# if not verifiable, we don't have evidence and just continue
			if data["verifiable"] ==  "NOT VERIFIABLE":
				continue

			positive_examples = set()
			negative_examples = set()
			good_docs = set()

			for evid in evidence:
				for i,item in enumerate(evid):
					Annotation_ID, Evidence_ID, Wikipedia_URL, sentence_ID = item
					Wikipedia_URL = unicodedata.normalize("NFC", Wikipedia_URL)

					# add positive example (only the first evidence)

					if i == 0:
						positive_examples.add((claim, Wikipedia_URL, sentence_ID, 0))

						# sample negative evidence:
						neg = sample_negative_example(Wikipedia_URL, docs, block_set)
						if neg != -1:
							#negative_examples.add((claim, neg[0], neg[1], 2))
							for n in neg:
								negative_examples.add((claim, n[0], n[1], 2))
						good_docs.add(Wikipedia_URL)
					
					# otherwise we just want to add the document so that we don't sample negative examples from a "good" document
					else:
						good_docs.add(Wikipedia_URL)



			# sample negative examples from other predicted pages which are not in good evidence
			for page in pred_pages:
				if page in docs:
					if page not in good_docs:
						neg = sample_negative_example(page, docs, block_set)

						if neg != -1:
							#negative_examples.add((claim, neg[0], neg[1], 2))
							# only add first three sentences (first few are most indicative given false positive wiki docs, especially the first sentence)
							for n in neg[:3]:
								negative_examples.add((claim, n[0], n[1], 2))
			# write positive and negative evidence to file
			for ex in positive_examples:
				sent = docs[ex[1]][ex[2]].split("\t")[1]
				outfile.write(ex[0] + "\t" + ex[1] + "\t" + sent + "\t" + str(ex[3]) + "\t" + str(ex[2]) + "\t" + label + "\n")
			for ex in negative_examples:
				try:
					sent = docs[ex[1]][ex[2]].split("\t")[1]
				#	print (ex[1], ex[2], "------",ex[0], "-------", sent, "------", ex[3])
					outfile.write(ex[0] + "\t" + ex[1] + "\t" + sent + "\t" + "2" + "\t" + str(ex[2]) + "\t" + label + "\n")
				except:
					pass
	outfile.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--infile')
	parser.add_argument('--outfile')
	parser.add_argument('--path_wiki_titles')
	args = parser.parse_args()
	generate_sentence_retrieval_training_set(args.infile, args.outfile, args.path_wiki_titles)

