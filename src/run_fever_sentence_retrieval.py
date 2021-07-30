
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import numpy as np

import argparse
import os
import unicodedata
import re

def normalize(text):
	return unicodedata.normalize('NFD', text)


def normalize_title_to_text(text):
	return normalize(text.strip()\
		.replace("-LRB-", "(")\
		.replace("-RRB-", ")")\
		.replace("_", " ")\
		.replace("-COLON-", ":")\
		.replace("-LSB-", "[")\
		.replace("-RSB-", "]"))

def process_evid(sentence):
	sentence = re.sub(" -LSB-.*?-RSB-", " ", sentence)
	sentence = re.sub(" -LRB- -RRB- ", " ", sentence)
	sentence = re.sub("-LRB-", "(", sentence)
	sentence = re.sub("-RRB-", ")", sentence)
	sentence = re.sub("-COLON-", ":", sentence)
	sentence = re.sub("_", " ", sentence)
	sentence = re.sub("\( *\,? *\)", "", sentence)
	sentence = re.sub("\( *[;,]", "(", sentence)
	sentence = re.sub("--", "-", sentence)
	sentence = re.sub("``", '"', sentence)
	sentence = re.sub("''", '"', sentence)
	return sentence




class SequenceClassificationDataset(Dataset):
	def __init__(self, filename, tokenizer):
		self.examples = self.load_data(filename)
		self.tokenizer = tokenizer
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def __len__(self):
		return len(self.examples)
	def __getitem__(self, idx):
		return self.examples[idx]
	def collate_fn(self, batch):
		model_inputs = self.tokenizer([i[0] for i in batch], [i[1] for i in batch], return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
		return model_inputs


	def load_data(self, filename):
		claims, evidence_sentences = [], []
		with open(filename) as f:
			for line in f:
				line = line.strip().split("\t")
				if len(line) == 6:
					claims.append(line[1])
					title = normalize_title_to_text(line[2])
					evidence_sentences.append(title + ":" + process_evid(line[4])
				elif len(line) == 9:
					# result = (str(example.id), example.claim, hyperlink, str(sent_ID), sent, "no_evidence") + current_sentence_A
					claims.append(line[1] + ":" + process_evid(line[-1]))
					title = normalize_title_to_text(line[2])
					evidence_sentences.append(title + process_evid(line[4])
		return list(zip(claims, evidence_sentences))


def write_results(filename, outfile_name, predictions):
	with open(filename) as f, open(outfile_name, "w") as outfile:
		for line, pred in zip(f, predictions):
			line = line.strip() + "\t" + str(pred)
			outfile.write(line + "\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--path_input_file', default="climate-fever", help="The path to the abstract sentences")
	parser.add_argument('--batch_size', type=int, default=32, help='')
	parser.add_argument('--model_name', type=str, default='models/sentence-retrieval-model/')
	parser.add_argument('--path_outfile', default="sentence_retrieval_predictions.tsv", help="filename for results")
	args = parser.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"

	model = RobertaForSequenceClassification.from_pretrained(args.model_name)
	tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
	model.to(device)

	devset = SequenceClassificationDataset(args.path_input_file), tokenizer)
	predictions = []

	model.eval()	
	with torch.no_grad():
		for i, batch in enumerate(tqdm(DataLoader(devset, batch_size=args.batch_size, collate_fn=devset.collate_fn))):
			output = model(**batch)
			# if the last batch contains only a single element, extend won't work and we have to append instead
			try:
				predictions.extend(output.logits.squeeze().cpu().tolist())
			except:
				predictions.append(output.logits.squeeze().cpu().tolist())
	write_results(os.path.join(args.path_input_file), os.path.join(args.path_outfile), predictions)

