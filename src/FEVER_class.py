import json
import re
import unicodedata
import os
from collections import Counter, defaultdict
import numpy as np
import argparse
from fever.scorer import fever_score
from fever_doc_db import FeverDocDB


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


	

class FEVER_Instance:
	"""
	a single FEVER training instance
	FEVER training instance
	"""
	def __init__(self, data, predicted_sentences=None):
		self.id = data["id"]
		# predict testset...
		self.claim = data["claim"]
		self.pages = set([unicodedata.normalize("NFD", page) for page in data["predicted_pages"]])
		#self.pages = set(data["predicted_pages"])

		try:
			self.label = data["label"]
			self.evidence = data["evidence"]
			self.verifiable = data["verifiable"]
			self.evidence_set = self.get_full_evidence_set(self.evidence)
			if predicted_sentences is not None:
				self.predicted_sentences = [(unicodedata.normalize("NFD", x[0]), x[1]) for x in predicted_sentences]
				#self.predicted_sentences = set([(unicodedata.normalize("NFD", x[0]), x[1]) for x in predicted_sentences])
				#sorted(self.predicted_sentences)
			else:
				self.predicted_sentences = []
		except:
			self.label = "NOT ENOUGH INFO"
			self.evidence = []
			self.verifiable = "NOT VERIFIABLE"
			self.evidence_set = set()
			if predicted_sentences is not None:
				self.predicted_sentences = [(unicodedata.normalize("NFD", x[0]), x[1]) for x in predicted_sentences]
				#self.predicted_sentences = set([(unicodedata.normalize("NFD", x[0]), x[1]) for x in predicted_sentences])
			else:
				#self.predicted_sentences = set()
				self.predicted_sentences = []



	def get_sentence_retrieval_examples(self, db, prob=0):
		"""
		get all sentence retrieval examples for an instance
		this is: sentences for the first sentence retrieval module (sentences_first_SR) and for the second (sentences_second_SR) 
		"""

		sentences_first_SR, sentences_second_SR = set(), set()

		# non verifiable case, ignore
		if self.verifiable == "NOT VERIFIABLE":
			return sentences_first_SR, sentences_second_SR			

		evid_set = self.get_evidence_set()
		if self.predicted_sentences:
			predicted_pages = set([p[0][0] for p in evid_set])
			predicted_sentences_pages = set([p[0] for p in self.predicted_sentences])
			predicted_sentences_docs = fetch_documents(db, predicted_sentences_pages)
		else:
			predicted_pages = set([p[0][0] for p in evid_set])
		docs = fetch_documents(db, predicted_pages.union(self.pages))

		for evid in evid_set:
			"""
			we don't have the conditioned evidence case if
				- only one sentence in an evidence set
				- all sentences in the evidence set can be found in the same document
				- all documents from which evidence sentences can come from are predicted by the document retrieval module
			"""

			for i, (evid_page, evid_sent_ID) in enumerate(evid):
				if i == 0:
					hyperlinks = get_hyperlinks_from_sent(docs, evid_page, evid_sent_ID)
					for page, sents in docs.items():
						for sent_ID, sent in enumerate(get_sentences(sents)):
							# some sentences are empty
							if not sent:
								continue
							if len(sent) <= 3:
								continue
							if "may refer to" in sent:
								continue
							# don't add negative example if it is in any evidence set of a fever example
							if (page, sent_ID) in self.evidence_set:
								continue
							else:
								result = (str(self.id), self.claim, page, str(sent_ID), sent, "no_evidence")
								sentences_first_SR.add(result)
					evid_sent = docs[evid_page][evid_sent_ID].split("\t")[1]
					result = (str(self.id), self.claim, evid_page, str(evid_sent_ID), evid_sent, "is_evidence")
					sentences_first_SR.add(result)
					current_sentence_A = (evid_page, str(evid_sent_ID), evid_sent)
					if self.predicted_sentences:
						for (page, sent_ID) in self.predicted_sentences:
							try:
								sent = predicted_sentences_docs[page][sent_ID].split("\t")[1]
							except:
								continue
							if not sent:
								continue
							if len(sent) <= 3:
								continue
							# don't add negative example if it is in any evidence set of a fever example
							if (page, sent_ID) in self.evidence_set:
								continue
							else:
								result = (str(self.id), self.claim, page, str(sent_ID), sent, "no_evidence")
								sentences_first_SR.add(result)
				else:
					if evid_page in predicted_pages:
						evid_sent = docs[evid_page][evid_sent_ID].split("\t")[1]
						result = (str(self.id), self.claim, evid_page, str(evid_sent_ID), evid_sent, "is_evidence")
						sentences_first_SR.add(result)
					elif self.predicted_sentences and evid_page in predicted_sentences_pages:
						evid_sent = predicted_sentences_docs[evid_page][evid_sent_ID].split("\t")[1]
						result = (str(self.id), self.claim, evid_page, str(evid_sent_ID), evid_sent, "is_evidence")
						sentences_first_SR.add(result)
					if evid_page not in hyperlinks:
						hyperlinks.add(evid_page)
					hyperlink_docs = fetch_documents(db, hyperlinks)
					for page, sents in hyperlink_docs.items():
						for sent_ID, sent in enumerate(get_sentences(sents)):
							#if len(sent_ID) > 2:
							#	continue
							# some sentences are empty
							if not sent:
								continue
							if len(sent) <= 3:
								continue
							if sent_ID > 20:
								continue
							if "may refer to" in sent:
								continue
							# don't add negative example if it is in any evidence set of a fever example
							if (page, sent_ID) in self.evidence_set:
								continue
							else:
								result = (str(self.id), self.claim, page, str(sent_ID), sent, "no_evidence") + current_sentence_A
								sentences_second_SR.add(result)
					evid_sent = hyperlink_docs[evid_page][evid_sent_ID].split("\t")[1]
					result = (str(self.id), self.claim, evid_page, str(evid_sent_ID), evid_sent, "is_evidence") + current_sentence_A
					sentences_second_SR.add(result)
		return sentences_first_SR, sentences_second_SR


	def get_full_evidence_set(self, evidence):
		# return set of all (page, sent_ID) in evidence_set (so that we do not randomly sample from one of these for tricky cases)
		evidence_set = set()
		if self.verifiable == "NOT VERIFIABLE":
			return evidence_set
		evidence = [[(unicodedata.normalize("NFD", i[2]) ,i[3]) for i in x] for x in self.evidence]
		for evid in evidence:
			evidence_set.update(evid)
		return evidence_set
		

	def get_wiki_sentences(self, page, wiki_data):
		return [sent.split("\t")[1] for sent in wiki_data[page]["lines"].split("\n")]

	def get_wiki_hyperlinks(self, page, wiki_data):
		pass
		# return [sent.split("\t")


	def get_evidence_set(self):
		# easy case, one evidence set:
		return [[(i[2] ,i[3]) for i in x] for x in self.evidence]



class ALL_instances:
	"""
	container for all FEVER instances
	"""
	def __init__(self, fn_train, fn_dev, db_file, fn_train_predicted_sentences="", fn_dev_predicted_sentences="", test=False, wiki_docs=""):
		if db_file:
			self.db = FeverDocDB(db_file)
		self.train_examples = []
		self.dev_examples = []
		claims = set()
		predicted_sentences = {}
		# load predicted sentencecs
		print (fn_train_predicted_sentences, fn_dev_predicted_sentences)
		if fn_train_predicted_sentences:
			with open(fn_train_predicted_sentences) as infile:
				for i, line in enumerate(infile):
					data = json.loads(line)
					claim_id = data["id"]
					predicted_sentences[claim_id] = data["predicted_sentences"]
		if fn_dev_predicted_sentences:
			with open(fn_dev_predicted_sentences) as infile:
				for i, line in enumerate(infile):
					data = json.loads(line)
					claim_id = data["id"]
					predicted_sentences[claim_id] = data["predicted_sentences"]
		print (len(predicted_sentences))
		# load fever instances
		with open(fn_train) as infile:
			for i, line in enumerate(infile):
				data = json.loads(line)
				claim = data["claim"]
				# some claims are double
				if claim in claims:
					continue
				claims.add(claim)
				claim_id = data["id"]
				if claim_id in predicted_sentences:
					self.train_examples.append(FEVER_Instance(data, predicted_sentences=predicted_sentences[claim_id]))
				else:
					self.train_examples.append(FEVER_Instance(data))
		with open(fn_dev) as infile:
			for i, line in enumerate(infile):
				data = json.loads(line)
				claim = data["claim"]
				if len(claim.split()) > 75:
					continue
				claim_id = data["id"]
				if claim_id in predicted_sentences:
					self.dev_examples.append(FEVER_Instance(data, predicted_sentences=predicted_sentences[claim_id]))
				else:
					self.dev_examples.append(FEVER_Instance(data))

	def write_sentence_retrieval_training_sets(self, path_outfiles):
		# counts of sentence IDs in conditioned case (sample of 1k train examples)
		# defaultdict(<class 'int'>, {0: 71, 1: 9, 8: 1, 10: 1, 4: 1, 3: 1}), 80 of 84 examples appear at sentence 1 or 2 in a doc

		y, n = 0,0
		#with open(os.path.join(args.path_outfiles, "train_data_first_SR.txt"), "w") as outfile_1:
		#	with open(os.path.join(args.path_outfiles, "train_data_second_SR.txt"), "w") as outfile_2:
		with open(os.path.join(args.path_outfiles, "train_data_SR.txt"), "w") as outfile:
			for example in instances.train_examples:
				results_1, results_2 = example.get_sentence_retrieval_examples(self.db)
				if results_1:
					for result in results_1:
						outfile.write("\t".join(result) + "\n")
				if results_2:
					#print (results_2)
					for result in results_2:
						outfile.write("\t".join(result) + "\n")

	def get_conditioned_dev_sentences(self):
		conditioned_dev_sentences = set()
		for example in self.dev_examples:
			results_1, results_2 = example.get_sentence_retrieval_examples(self.db)
			if results_2:
				#print (results_2)
				for result in results_2:
					conditioned_dev_sentences.add((result[0], result[2], result[3]))
		return conditioned_dev_sentences


	def get_upper_bound(self):
		predictions, actual = [], []
		with open(os.path.join(args.path_outfiles, "train_data_first_SR.txt"), "w") as outfile_1:
			with open(os.path.join(args.path_outfiles, "train_data_second_SR.txt"), "w") as outfile_2:
				for example in instances.train_examples:
					actual.append({"label":example.label, "evidence":list(example.get_full_evidence_set)})
					if label != "NOT ENOUGH INFO":
						predictions.append({"id": int(claim_id), "predicted_label":pred_label, "predicted_evidence":list(predicted_evidence)})
					else:
						predictions.append({"id":int(claim_id), "predicted_label":"NOT ENOUGH INFO", "predicted_evidence":[["Page", 0]]})

					#try:
					results_1, results_2 = example.get_sentence_retrieval_examples(self.db)
					if results_1:
						for result in results_1:
							outfile_1.write("\t".join(result) + "\n")
					if results_2:
						#print (results_2)
						for result in results_2:
							outfile_2.write("\t".join(result) + "\n")


	def sanity_check(self, sent):
		# some sentences are empty
		if not sent:
			return False
		# some are too short
		if len(sent) <= 3:
			return False
		# some are from disambiguation pages and thus not valuable
		if "may refer to" in sent:
			return False
		return True
		



	def write_all_train_sentences_to_outfile(self, outfile_name):
		with open(outfile_name, "w") as outfile:
			for example in self.train_examples:
				# all predicted pages
				docs = fetch_documents(self.db, example.pages)

				for page, sents in docs.items():
					for sent_ID, sent in enumerate(get_sentences(sents)):
						if not self.sanity_check(sent):
							continue
						# don't add negative example if it is in any evidence set of a fever example
						if (page, sent_ID) in example.evidence_set:			
							result = (str(example.id), example.claim, page, str(sent_ID), sent, "is_evidence")
						else:
							result = (str(example.id), example.claim, page, str(sent_ID), sent, "no_evidence")

						outfile.write("\t".join(result) + "\n")

				# all predicted sentences
				if example.predicted_sentences:
					docs = fetch_documents(self.db, set(i[0] for i in example.predicted_sentences))
					for (page, sent_ID) in example.predicted_sentences:
						if page not in example.pages:
							try:
								sent = docs[page][sent_ID].split("\t")[1]
							except:
								continue
							if not self.sanity_check(sent):
								continue
							if (page, sent_ID) in example.evidence_set:			
								result = (str(example.id), example.claim, page, str(sent_ID), sent, "is_evidence")
							else:
								result = (str(example.id), example.claim, page, str(sent_ID), sent, "no_evidence")
							outfile.write("\t".join(result) + "\n")

	def write_all_dev_sentences_to_outfile(self, outfile_name):
		with open(outfile_name, "w") as outfile:
			for example in self.dev_examples:
				# all predicted pages
				docs = fetch_documents(self.db, example.pages)

				for page, sents in docs.items():
					for sent_ID, sent in enumerate(get_sentences(sents)):
						if not self.sanity_check(sent):
							continue
						if (page, sent_ID) in example.evidence_set:			
							result = (str(example.id), example.claim, page, str(sent_ID), sent, "is_evidence")
						else:
							result = (str(example.id), example.claim, page, str(sent_ID), sent, "no_evidence")

						outfile.write("\t".join(result) + "\n")

				# all predicted sentences
				if example.predicted_sentences:
					docs = fetch_documents(self.db, set(i[0] for i in example.predicted_sentences))
					for (page, sent_ID) in example.predicted_sentences:
						if page not in example.pages:
							try:
								sent = docs[page][sent_ID].split("\t")[1]
							except:
								continue
							if not self.sanity_check(sent):
								continue
							if (page, sent_ID) in example.evidence_set:			
								result = (str(example.id), example.claim, page, str(sent_ID), sent, "is_evidence")
							else:
								result = (str(example.id), example.claim, page, str(sent_ID), sent, "no_evidence")
							outfile.write("\t".join(result) + "\n")

class Wiki_Docs:
	"""
	Wikipedia documents
	one doc looks the following:
	{"id": "Record", "text": "A recording , record or records may mean : ", "lines": "0\tA recording , record or records may mean :\n1\t"}
	"""

	def __init__(self, path_train, path_dev, path_wiki):
		self.path_train = path_train
		self.path_dev = path_dev
		self.path_wiki = path_wiki
		#self.wiki = {**self.wiki_train, **self.wiki_dev}
		self.wiki = self.load_all_wiki(path_wiki)
		self.all_pages, self.pages_lowercased = self.get_all_pages_map()

	def load_all_wiki(self, path):
		files = os.listdir(path)
		wiki_docs = {}
		print ("loading wikipedia files")
		for f in files:
			#print (f)
			with open(os.path.join(path, f)) as infile:
				for line in infile:
					data = json.loads(line)
					title = data["id"]
					title = unicodedata.normalize("NFD", title)
					wiki_docs[title] = data["lines"].split("\n")
		print ("wikipedia files loaded", len(wiki_docs))
		return wiki_docs


	def get_all_pages_map(self, fn="data/all_wiki_titles.txt"):
		pages_lowercased = defaultdict(list)
		all_pages = set()
		with open(fn) as infile:
			for line in infile:
				title = line.strip()
				title = unicodedata.normalize("NFD", title)
				pages_lowercased[title.lower()].append(title)
				all_pages.add(title)
		return all_pages, pages_lowercased


	def load_wiki_pages(self, path, docs):
		files = os.listdir(path)
		#wiki_docs = defaultdict(list)
		wiki_docs = {}
		print (len(docs))
		for f in files:
			print (f)
			with open(os.path.join(path, f)) as infile:
				for line in infile:
					data = json.loads(line)
					title = data["id"]
					title = unicodedata.normalize("NFD", title)
					if title in docs:
						wiki_docs[title] = data["lines"].split("\n")
		return wiki_docs

	def get_raw_sentence(self, page, sent_id):
		# fix here first
		return self.wiki[page][sent_id].split("\t")[1]

	def get_surface_hyperlinks(self, page, sent_id):
		surface_forms = []
		for i, item in enumerate(self.wiki[page][sent_id].split("\t")[2:]):
			if i % 2 == 0:
				item = self.normalize_text_to_title(item)
				surface_forms.append(item)
		return surface_forms
	


	def normalize(self, text):
		return unicodedata.normalize('NFD', text)

	def normalize_text_to_title(self, text):
		return self.normalize(text.strip()\
			.replace("(","-LRB-")\
			.replace(")","-RRB-")\
			.replace(" ","_")\
			.replace(":","-COLON-")\
			.replace("[","-LSB-")\
			.replace("]","-RSB-"))

	def get_hyperlinks_from_doc(self, page, to_skip=-1, target_page="", claim=""):
		hyperlinks = []
		for page_id, sent in enumerate(self.wiki[page]):
			if page_id == to_skip:
				continue
			for i, hyperlink in enumerate(sent.split("\t")[2:]):
				hyperlink = self.normalize_text_to_title(hyperlink)
				if hyperlink == target_page:
					print ("-----------------------------------")
					print ("claim", claim)
					print ("Page:", page, "recoverable sentence:", self.get_raw_sentence(page, page_id))
					print ("target page", hyperlink)
					input("")
				hyperlinks.append(hyperlink)
		return hyperlinks


	def get_hyperlinks_from_sent(self, page, sent_id):
		hyperlinks = set()
		for i, hyperlink in enumerate(self.wiki[page][sent_id].split("\t")[2:]):
			if i % 2 == 1:
				item = self.normalize_text_to_title(hyperlink)
				if item in self.all_pages:
					hyperlinks.add(item)
				elif item.lower() in self.pages_lowercased:
					hyperlinks.update(self.pages_lowercased[item.lower()])
		return hyperlinks

	def get_sentences_from_page(self, page):
		# return tuples of sentence, Wikipedia_URL, sentence_ID
		sentences = []
		if page in self.wiki:
			if self.wiki[page] == "":
				return None
			for sentence in self.wiki[page]:
				if sentence == "":
					continue
				sentences.append((sentence.split("\t")[1], page, int(sentence.split("\t")[0])))
		elif page in self.wiki_dev:
			if self.wiki_dev[page] == "":
				return None
			for sentence in self.wiki_dev[page]:
				if sentence == "":
					continue
				sentences.append((sentence.split("\t")[1], page, int(sentence.split("\t")[0])))
		if sentences:
			return sentences

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--db_file", required=True, type=str, "path to wikipedia database")
	parser.add_argument("--athene_document_retrieval_train", required=True, type=str, help="path to athene document retrieval output train")
	parser.add_argument("--athene_document_retrieval_dev", required=True, type=str, help="path to athene document retrieval output dev")
	parser.add_argument("--path_outfiles", default="data", type=str, help="path to pages of the Wikipedia dump")
	parser.add_argument("--all_train_sentences", default="all_sentences_train_set.txt", type=str, help=	"path to the outfile with all sentences from the training set")
	parser.add_argument("--all_dev_sentences", default="all_sentences_dev_set.txt", type=str, help="path to the outfile with all sentences from the training set")
	parser.add_argument("--fever_baseline_predicted_sentences_train", required=True, type=str, help="path to FEVER baseline document retrieval output train")
	parser.add_argument("--fever_baseline_predicted_sentences_dev", required=True, type=str, help="path to FEVER baseline document retrieval output dev") 
	args = parser.parse_args()

	# if testset, just replace paths to dev set with paths to test set
	try:
		os.mkdir(args.path_outfiles)
	except:
		pass

	instances = ALL_instances(args.athene_document_retrieval_train, args.athene_document_retrieval_dev, args.db_file, fn_train_predicted_sentences=args.fever_baseline_predicted_sentences_train, fn_dev_predicted_sentences=args.fever_baseline_predicted_sentences_dev)
	instances.write_sentence_retrieval_training_sets(args.path_outfiles)
	instances.write_all_train_sentences_to_outfile(os.path.join(args.path_outfiles, args.all_train_sentences))
	instances.write_all_dev_sentences_to_outfile(os.path.join(args.path_outfiles, args.all_dev_sentences))


