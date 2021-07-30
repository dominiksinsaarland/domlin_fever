from collections import defaultdict
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

import argparse
import os
from tqdm import tqdm
import numpy as np

def load_predicted_evidence_sentences(filename, filename_multihop):
	retrieved_evidence = defaultdict(lambda: defaultdict(float))
	with open(filename) as f:
		for line in f:
			line = line.strip().split("\t")
			claim_id, claim, doc, sent_id, evid_sent, label_evid, prediction= line
			claim_id, claim, sentence, title, prediction = line
			prediction = float(prediction)
			# key: (claim_id, claim), values: dictionairy of key:sentence, value:prediction
			if prediction > 0:
				retrieved_evidence[(line[0], line[1])][line[4]] = prediction
	with open(filename_multihop) as f:
		for line in f:
			line = line.strip().split("\t")
			claim_id, claim, doc, sent_id, evid_sent, label_evid, _, _ ,_, prediction= line
			prediction = float(prediction)
			if prediction > 0:
				prediction = 0.01
				if sentence not in retrieved_evidence[(claim_id, claim)]:
					retrieved_evidence[(line[0], line[1])][line[4]] = prediction
				# else pass, it's already been predicted from first sentence retrieval module


	return retrieved_evidence

class SequenceClassificationDataset(Dataset):
	def __init__(self, retrieved_evidence, tokenizer):
		self.examples = self.prepare_dataset(retrieved_evidence)
		self.tokenizer = tokenizer
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def __len__(self):
		return len(self.examples)
	def __getitem__(self, idx):
		return self.examples[idx]
	def collate_fn(self, batch):
		model_inputs = self.tokenizer([i["claim"] for i in batch], [i["evidence"] for i in batch], return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
		return model_inputs

	def prepare_dataset(self, retrieved_evidence):
		examples = []
		for (claim_id, claim), sentences in retrieved_evidence.items():
			# five highest scoring evidence sentences for this claim
			if not sentences:
				continue
			evidence = sorted(sentences.items(), key=lambda x:x[1], reverse=True)[:5]
			raw_sentences = [i[0] for i in evidence]
			scores = [i[1] for i in evidence]
			example = {"claim_id": claim_id, "claim": claim, "evidence": " ".join(raw_sentences), "evidence_list":evidence, "scores": scores}
			examples.append(example)
		return examples


def write_outfile(outfile, claims, scores, path_dataset):
	id2label = {0:"REFUTES", 1:"NOT ENOUGH INFO", 2:"SUPPORTS"}
	# ok, we rewrite here!

	claim_predictions = {}

	for claim, score in zip(claims, scores):
		idx = claim["claim_id"]
		claim_predictions[idx] = id2label[np.argmax(score)]

	with open(outfile, "w") as f, open(path_dataset) as dataset:
		for claim, line in zip(claims, dataset):
			line = json.loads(line)
			if "claim_id" in line:
				claim_id = str(line["claim_id"])
			elif "id" in line:
				claim_id = str(line["id"])
			if claim_id in claim_predictions:
				predicted_label = claim_predictions[claim_id]
			else:
				predicted_label = "NOT ENOUGH INFO"
			claim["predicted_label"] = predicted_label
			claim["label"] = line["label"]
			json.dump(claim, f)
			f.write("\n")

def evaluate(claims):
	y, preds = [], []
	# move this to preprocessing

	#unify_labels = {"SUPPORTS": "SUPPORTS", "NOT_ENOUGH_INFO": "NOT ENOUGH INFO", "REFUTES": "REFUTES"}
	#unify_labels = {"SUPPORT": "SUPPORTS", "NOT ENOUGH INFO": "NOT ENOUGH INFO", "CONTRADICT": "REFUTES"}
	with open(claims) as f:
		for claim in f:
			claim = json.loads(claim)
			y.append(claim["label"])
			preds.append(claim["predicted_label"])

	print(metrics.classification_report(y, preds))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--sentence_retrieval_predictions', required=True, help="path to sentence retrieval predictions")
	parser.add_argument('--sentence_retrieval_predictions_multihop', required=True, help="path to sentence retrieval predictions")
	parser.add_argument('--athene_document_retrieval_dev', required=True)
	parser.add_argument('--batch_size', type=int, default=2, help='')
	parser.add_argument('--model_name', type=str, default='models/rte-model/')
	parser.add_argument('--outfile', default="claim_predictions.jsonl", help="filename for results")
	args = parser.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"

	model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)
	model.to(device)

	retrieved_evidence = load_predicted_evidence_sentences(args.sentence_retrieval_predictions, args.sentence_retrieval_predictions_multihop)
	devset = SequenceClassificationDataset(retrieved_evidence, tokenizer)	
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

	write_outfile(args.outfile, devset.examples, predictions, athene_document_retrieval_dev)	
	evaluate(args.outfile)


