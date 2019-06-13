import json
import re
import unicodedata
import os
from collections import defaultdict
import numpy as np
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--infile')
	parser.add_argument('--outfile')
	parser.add_argument('--path_wiki_titles')
	args = parser.parse_args()
	generate_sentence_retrieval_training_set(args.infile, args.outfile, args.path_wiki_titles)




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

def process_title_rev(title):
	# reverse engineer title to find the relevant document
	title = re.sub("\(", "-LRB-", title)
	title = re.sub("\)", "-RRB-", title)
	title = re.sub("\:", "-COLON-", title)
	title = re.sub(" ", "_", title)
	return title

def load_evidence_chain_docs(path_to_json, path_to_wiki, docs_first_pass):
	new_docs = set()
	with open(path_to_json) as infile:
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
					if len(item) > 1:
					
						Annotation_ID, Evidence_ID, Wikipedia_URL, sentence_ID = item
						Wikipedia_URL = unicodedata.normalize("NFC", Wikipedia_URL)

						sent = docs_first_pass[Wikipedia_URL][sentence_ID].split("\t")
						for page in sent[2:]:
							page = process_title_rev(page)
							if page not in docs_first_pass:
								new_docs.add(page)

	#docs = load_wiki_pages("/home/dominik/Documents/FEVER/wiki-pages", docs)
	new_docs = load_wiki_pages(path_to_wiki, new_docs)
	return new_docs



def generate_evidence_chains_train_new(path_to_infile, path_to_outfile, path_to_wiki):


	outfile = open(path_to_outfile, "w")

	# load relevant documents first

	docs, all_wiki_titles = load_wiki_docs(path_to_infile, path_to_wiki,return_all_titles=True)
	new_docs = load_evidence_chain_docs(path_to_infile, path_to_wiki, docs)
	docs = {**docs, **new_docs}

	missed, found = 0,0
	with open(path_to_infile) as infile:
		for line in infile:
			data = json.loads(line)
			claim = data["claim"]
			label = data["label"]
			evidence = data["evidence"]
			pred_pages = data["predicted_pages"]
			if data["verifiable"] ==  "NOT VERIFIABLE":
				continue
			block_set = set()
			positive_examples = set()
			negative_examples = set()
			positive_evidence = set()
			for evid in evidence:
				for i, item in enumerate(evid):
					if len(evid) == 1:
						block_set.add((unicodedata.normalize("NFC",item[2]), item[3]))
				for i, item in enumerate(evid):
					positive_evidence.add((unicodedata.normalize("NFC",item[2]), item[3]))

			for evid in evidence:
				altered_evid = ""
				pages_with_additional_evidence = set()
				num_docs = set()
				for i,item in enumerate(evid):
					num_docs.add(unicodedata.normalize("NFC",item[2]))
				for i,item in enumerate(evid):
					# only one evidence, we already covered that
					if len(evid) == 1:
						continue
					
					# we are not going to make more than one pass, therefore we can skip documents with evidence from more than 2 docs
					elif len(num_docs) > 2:
						continue

					Annotation_ID, Evidence_ID, Wikipedia_URL, sentence_ID = item
					Wikipedia_URL = unicodedata.normalize("NFC", Wikipedia_URL)

			
					# if we have two documents, we are good

					if i == 0:
						# first, concatenate Wikipedia_URL and the first evidence sentence (Wiki_URL for coreference resolution)
						altered_evid = Wikipedia_URL + " : " + docs[Wikipedia_URL][sentence_ID].split("\t")[1]
						# find additional pages
						additional_pages = docs[Wikipedia_URL][sentence_ID].split("\t")[2:]
						pages_with_additional_evidence.update([process_title_rev(page) for page in additional_pages])
						# iterate through the whole article to find more links to outgoing articles from the document
						for sent in docs[Wikipedia_URL]:
							sent = sent.split("\t")
							if len(sent) > 2:
								pages_with_additional_evidence.update([process_title_rev(page) for page in sent[2:]])
						continue

					else:
						positive_examples.add((claim, Wikipedia_URL, altered_evid, sentence_ID, 0))



					counter = 0
					neg = sample_negative_example(Wikipedia_URL, docs, positive_evidence)
					if neg != -1:
						for n in neg:
							negative_examples.add((claim, Wikipedia_URL, altered_evid, n[1], 2))

					for page in pages_with_additional_evidence:
						if counter > 10:
							break
						if len(pages_with_additional_evidence) > 10:
							# if we have lots of outgoing links, sample only first sentence as it is the most informative one
							if page in docs:
								neg = sample_negative_example(page, docs, positive_evidence)
								if neg != -1:
									#negative_examples.add((claim, neg[0], neg[1], 2))
									for n in neg[:1]:
										negative_examples.add((claim, page, altered_evid, n[1], 2))
						elif len(pages_with_additional_evidence) > 5:
							if page in docs:
							# if we have 5-10 outgoing links, sample first 3 sentences

								neg = sample_negative_example(page, docs, positive_evidence)
								if neg != -1:
									for n in neg[:3]:
										negative_examples.add((claim, page, altered_evid, n[1], 2))
						else:
							# otherwise, sample all the sentences
							if page in docs:
								neg = sample_negative_example(page, docs, positive_evidence)
								if neg != -1:
									for n in neg:
										negative_examples.add((claim, page, altered_evid, n[1], 2))


		
			for ex in positive_examples:
				claim, wiki_page, altered_evid, sent_id, ir_label = ex
				sent = docs[wiki_page][sent_id].split("\t")[1]
				sent = altered_evid + " " + wiki_page + " : " + sent
				outfile.write(ex[0] + "\t\t" + sent + "\t" + str(ir_label) + "\t" + str(sent_id) + "\t" + label + "\n")

			for ex in negative_examples:
				claim, wiki_page, altered_evid, sent_id, ir_label = ex
				if wiki_page in docs:
					sent = docs[wiki_page][sent_id].split("\t")
					if len(sent) >= 2:						
						sent = sent[1]
						if len(sent.split()) > 5:
							sent = altered_evid + " " + wiki_page + " : " + sent
							outfile.write(ex[0] + "\t\t" + sent + "\t" + str(ir_label) + "\t" + str(sent_id) + "\t" + label + "\n")

	outfile.close()


