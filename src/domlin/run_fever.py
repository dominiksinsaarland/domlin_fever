# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import re
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import random
import json
from collections import defaultdict
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string("use_squad_data", None, "")
flags.DEFINE_string("prediction_file", None, "filename")
flags.DEFINE_string("use_hingeloss", None, "whether to use hinge loss")
flags.DEFINE_string("max_pooling", None, "pooling strategy")
flags.DEFINE_string("train_file", None, "filename")

flags.DEFINE_integer("negative_samples", 2, "filename")

flags.DEFINE_string("file_test_results", None, "filename")

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None,"The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for i,line in enumerate(reader):
        #if i > 10500:
        #  return lines
        lines.append(line)
      return lines


class EvidChains(DataProcessor):
  def __init__(self):
    self.language = "en"



  def process_evid(self, sentence):
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



  def get_train_examples(self, data_dir):
    """See base class."""
    #lines = self._read_tsv(os.path.join(data_dir, "sentences_IR_train_v2.tsv"))
    lines = self._read_tsv(os.path.join(data_dir, "IR_training_set_evidence_chains_v5.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_a = self.process_evid(text_a)
      text_b = tokenization.convert_to_unicode(line[2])
      text_b = self.process_evid(text_b)
      label = tokenization.convert_to_unicode(line[3])

      #print (text_a)
      #print (text_b)
      #print (line[3])
      #input("")

      if label == "2":
        if random.randint(1,FLAGS.negative_samples) != 1:
          continue
        label = 0

      else:
        label = 1
      #if len(text_a.split()) + len(text_b.split()) > 100:
      #  continue

      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples




  def get_dev_examples(self, data_dir):
    #lines = self._read_tsv(os.path.join(data_dir, "evidence_chains_dev_set_v2.tsv"))
    #lines = self._read_tsv(os.path.join(data_dir, "NEI_evidence_chains.tsv"))




    if FLAGS.prediction_file is not None:
      lines = self._read_tsv(os.path.join(data_dir, FLAGS.prediction_file))
    else:
      lines = self._read_tsv(os.path.join(data_dir, "evid_chains_dev_athene.tsv"))

    examples = []
    for (i, line) in enumerate(lines):
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_a = self.process_evid(text_a)
      text_b = tokenization.convert_to_unicode(line[1])
      text_b = self.process_evid(text_b)
      #label = "2"
      label = 0
      #if len(text_a.split()) + len(text_b.split()) > 100:
      #  continue

      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


  def get_labels(self):
    """See base class."""
    #return ["0", "1", "2"]
    return [0,1]


class DocumentRetrieval(DataProcessor):
  """ disambiguate wikipedia titles"""

  def __init__(self):
    self.language = "en"


  def process_sent(self, sentence):
    sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
    sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
    sentence = re.sub(" -LRB-", " ( ", sentence)
    sentence = re.sub("-RRB-", " )", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)

    return sentence

  def process_wiki_title(self, title):
    title = re.sub("_", " ", title)
    title = re.sub(" -LRB-", " ( ", title)
    title = re.sub("-RRB-", " )", title)
    title = re.sub("-COLON-", ":", title)
    return title




  def process_evid(self, sentence):
    sentence = re.sub(" -LSB-.*-RSB-", " ", sentence)
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

  def get_train_examples(self, data_dir):
    """See base class."""
    #lines = self._read_tsv(os.path.join(data_dir, "sentences_IR_train_v2.tsv"))
    lines = self._read_tsv(os.path.join(data_dir, "IR_training_set.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_a = self.process_sent(text_a)
      title = tokenization.convert_to_unicode(line[1])
      title = self.process_wiki_title(title)
      text_b = tokenization.convert_to_unicode(line[2])
      text_b = self.process_evid(text_b)
      text_b = title + " : " + text_b
      label = tokenization.convert_to_unicode(line[3])

      #print (text_a)
      #print (text_b)
      #print (line[2])
      #input("")

      if label == "2":
        if random.randint(1,FLAGS.negative_samples) != 1:
          continue
        else:
          label = 0
      else:
        label = 1
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples




  def get_dev_examples(self, data_dir):
    """See base class."""


    if FLAGS.prediction_file is not None:
      lines = self._read_tsv(os.path.join(data_dir, FLAGS.prediction_file))
    else:
      lines = self._read_tsv(os.path.join(data_dir, "all_sentences_dev_set.tsv"))
    #lines = self._read_tsv(os.path.join(data_dir, "all_NEI_sents_train.tsv")) 

    #lines = self._read_tsv(os.path.join(data_dir, "second_pass_sentences_dev_set_v2.tsv"))
    #lines = self._read_tsv(os.path.join(data_dir, "sentences_IR_dev_v2.tsv"))
    #lines = self._read_tsv(os.path.join(data_dir, "sentences_IR_dev.tsv")) 
    #lines = self._read_tsv(os.path.join(data_dir, "second_pass_sentences_NEI.tsv")) 


    examples = []
    for (i, line) in enumerate(lines):
      guid = "dev-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_a = self.process_sent(text_a)

      title = tokenization.convert_to_unicode(line[1])
      title = self.process_wiki_title(title)

      text_b = self.process_evid(tokenization.convert_to_unicode(line[2]))
      text_b = title + " : " + text_b
      label = "2"
      label = 0
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    #return ["0", "1", "2"]
    return [0, 1]

class RTEProcessor(DataProcessor):


  def __init__(self):
    self.language = "en"

  def process_sent(self, sentence):
    sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
    sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
    sentence = re.sub(" -LRB-", " ( ", sentence)
    sentence = re.sub("-RRB-", " )", sentence)
    return sentence

  def process_wiki_title(self, title):
    title = re.sub("_", " ", title)
    title = re.sub(" -LRB-", " ( ", title)
    title = re.sub("-RRB-", " )", title)
    title = re.sub("-COLON-", ":", title)
    return title


  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "RTE_not_enough_evidence.tsv"))
    examples = []

    for (i, line) in enumerate(lines):

      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_a = self.process_sent(text_a)
      text_b = ""
      for j, evidence in enumerate(line[1:-1]):
        if j % 2 == 0 and evidence != ".":
          title = tokenization.convert_to_unicode(evidence)
          title = self.process_wiki_title(title)
          text_b += title
        elif j % 1 == 0:
          evid = tokenization.convert_to_unicode(evidence)
          evid = self.process_sent(evid)
          text_b += evid
      label = line[-1]
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "RTE_first_classifier_dev.tsv"))
    examples = []

    for (i, line) in enumerate(lines):

      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_a = self.process_sent(text_a)
      text_b = ""
      for j, evidence in enumerate(line[1:-1]):
        if j % 2 == 0 and evidence != ".":
          title = tokenization.convert_to_unicode(evidence)
          title = self.process_wiki_title(title)
          text_b += title
        elif j % 1 == 0:
          evid = tokenization.convert_to_unicode(evidence)
          evid = self.process_sent(evid)
          text_b += evid
      label = line[-1]
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples

  def get_labels(self):
    """See base class."""
    return ["VERIFIABLE", "NOT ENOUGH EVIDENCE"]





class FEVERProcessor(DataProcessor):




  """Processor for the FEVER data set."""

  def __init__(self):
    self.language = "en"


  def process_evid(self, sentence):
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


  def get_train_examples(self, data_dir):

    #print ("HERE", "reading from file", os.path.join(data_dir, "RTE_train_set_with_additional_evidence.tsv"))
    #input("")

    tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    paras = set()
    examples = []

    if FLAGS.use_squad_data is not None:
      with tf.gfile.Open("../../squad2.0_data/train-v2.0.json", "r") as reader:
        input_data = json.load(reader)["data"]
      c = 0
      print (len(input_data))
      for i, entry in enumerate(input_data):
        if i % 100 == 0:
          print (i)
        for paragraph in entry["paragraphs"]:
          paragraph_text = paragraph["context"]
          text_b = tokenization.convert_to_unicode(self.process_evid(paragraph_text))
          tokens_b = tokenizer.tokenize(text_b)
          for qa in paragraph["qas"]:
            qas_id = qa["id"]
            question_text = qa["question"]
            is_impossible = qa["is_impossible"]
            text_a = tokenization.convert_to_unicode(question_text)

            if is_impossible:
              label = "NOT ENOUGH INFO"
            else:
              continue
            #label = "SUPPORTS"
            tokens_a = tokenizer.tokenize(text_a)
            paras.add(paragraph_text)
            if len(tokens_a) + len(tokens_b) + 3 < 375:
              c += 1
              if c < 3:
                print (text_a, text_b, label)
              #if label == "SUPPORTS":
              #  if random.randint(1,3) != 1:
              #    continue
              guid = "squad-%d" % (c)
              examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))


    if FLAGS.train_file is None:
      lines = self._read_tsv(os.path.join(data_dir, "RTE_train_set_with_additional_evidence.tsv"))
    else:
      lines = self._read_tsv(os.path.join(data_dir, FLAGS.train_file))


    #examples = []
    for (i, line) in enumerate(lines):

          
      guid = "train-%d" % (i)

      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      #text_a = self.process_sent(text_a)
      text_b = self.process_evid(text_b)
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    random.shuffle(examples)
    return examples
    
  """
  def get_train_examples_old(self, data_dir):
    if FLAGS.train_file is None:
      lines = self._read_tsv(os.path.join(data_dir, "RTE_good_bad_train.tsv"))
    else:
      lines = self._read_tsv(os.path.join(data_dir, FLAGS.train_file))

    #lines = self._read_tsv(os.path.join(data_dir, "RTE_good_bad_combined_evid.txt"))
    examples = []
    for (i, line) in enumerate(lines):

          
      guid = "train-%d" % (i)

      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      #text_a = self.process_sent(text_a)
      text_b = self.process_evid(text_b)
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))


    #lines = self._read_tsv(os.path.join(data_dir, "RTE_not_enough_info_ensemble.tsv"))
    lines = self._read_tsv(os.path.join(data_dir, "NEI_set_new_approach.tsv"))
    counter = len(examples) + 1
    for (i, line) in enumerate(lines):

      guid = "train-%d" % (i + counter)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      #text_a = self.process_sent(text_a)
      text_b = self.process_evid(text_b)
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    random.shuffle(examples)
    return examples
  """
    
  def get_dev_examples(self, data_dir):
    """See base class."""

    if FLAGS.prediction_file is not None:
      lines = self._read_tsv(os.path.join(data_dir, FLAGS.prediction_file))
    else:
      #lines = self._read_tsv("/raid/dost01/fever_2.0/RTE_good_bad_dev.tsv")
      lines = self._read_tsv(os.path.join(data_dir, "RTE_dev_with_retrieved_evidence.tsv"))
      #lines = self._read_tsv(os.path.join(data_dir, "RTE_test_with_retrieved_evidence.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      guid = "dev-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      text_b = self.process_evid(text_b)
      label = tokenization.convert_to_unicode(line[2])
      #label = "SUPPORTS"
      #if label == "NOT ENOUGH EVIDENCE":
      #  continue
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples

  def get_labels(self):
    """See base class."""
    return ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    #return ["SUPPORTS", "REFUTES"]
    #return ["SUPPORTS", "NOT ENOUGH EVIDENCE"]


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)
    all_seq_lengths[len(tokens_a) + len(tokens_b)] += 1
  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  if FLAGS.max_pooling is None:
    output_layer = model.get_pooled_output()
  elif FLAGS.max_pooling == "max_pooling":
    def create_initializer(initializer_range=0.02):
      """Creates a `truncated_normal_initializer` with the given range."""
      return tf.truncated_normal_initializer(stddev=initializer_range)


    input_mask = tf.cast(input_mask, tf.float32)

    minus_mask = lambda x, m: x - tf.expand_dims(1.0 - m, axis=-1) * 1e30
    mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
    masked_reduce_max = lambda x, m: tf.reduce_max(minus_mask(x, m), axis=1)
    masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)


    output_layer = model.get_sequence_output()
    output_layer = masked_reduce_max(output_layer, input_mask)

    output_layer = tf.layers.dense(
            output_layer,
            bert_config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(bert_config.initializer_range))

  elif FLAGS.max_pooling == "all_pooling":
    def create_initializer(initializer_range=0.02):
      """Creates a `truncated_normal_initializer` with the given range."""
      return tf.truncated_normal_initializer(stddev=initializer_range)

    input_mask = tf.cast(input_mask, tf.float32)
    minus_mask = lambda x, m: x - tf.expand_dims(1.0 - m, axis=-1) * 1e30
    mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)

    masked_reduce_max = lambda x, m: tf.reduce_max(minus_mask(x, m), axis=1)
    masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)


    output_layer = model.get_sequence_output()
    output_layer = tf.concat([masked_reduce_max(output_layer, input_mask), masked_reduce_mean(output_layer, input_mask)], axis=-1)
    output_layer = tf.layers.dense(
            output_layer,
            bert_config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(bert_config.initializer_range))


  hidden_size = output_layer.shape[-1].value

  """
  output_layer = model.get_sequence_output()
  output_0 = model.get_pooled_output()
  output_1 = tf.reduce_max(output_layer, axis=1)
  output_2 = tf.reduce_min(output_layer, axis=1)
  output_3 = tf.reduce_mean(output_layer, axis=1)
  hidden_size = output_layer.shape[-1].value * 4
  output_layer = tf.concat([output_0, output_1, output_2, output_3], axis=-1)
  """

  if FLAGS.use_hingeloss is not None:
    output_weights = tf.get_variable(
      "output_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  else:
    output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  #init = tf.constant_initializer([0.61,0.25, 0.11])
  if FLAGS.use_hingeloss is not None:
    output_bias = tf.get_variable(
      "output_bias", [1], initializer=tf.zeros_initializer())

  else:
    output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())
  #    "output_bias", [num_labels], initializer=init)

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    if FLAGS.use_hingeloss is not None:
      print (labels.get_shape)

      labels = tf.expand_dims(tf.cast(labels, tf.float32), axis=-1)
      print (labels.get_shape)     
      print (logits.get_shape())
      loss = tf.losses.hinge_loss(labels, logits)
      probabilities = logits
      per_example_loss = 0
    else:
      probabilities = tf.nn.softmax(logits, axis=-1)
      log_probs = tf.nn.log_softmax(logits, axis=-1)

      one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
      loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    #tf.logging.info("*** Features ***")
    #for name in sorted(features.keys()):
    #  tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    """
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    """
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def eval_confusion_matrix(labels, predictions):
        """

	confusion_matrix 


	The matrix columns represent the prediction labels and the rows represent the real labels.
	[[86075   496]
	[  798 12536]]

	so TP 12536, FP 496, FN 798
        """ 
        with tf.variable_scope("eval_confusion_matrix"):
          con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_labels)

          con_matrix_sum = tf.Variable(tf.zeros(shape=(num_labels,num_labels), dtype=tf.int32),
                                            trainable=False,
                                            name="confusion_matrix_result",
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])
          update_op = tf.assign_add(con_matrix_sum, con_matrix)
          return tf.convert_to_tensor(con_matrix_sum), update_op


      def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(label_ids, predictions)
        loss = tf.metrics.mean(per_example_loss)
        cm = eval_confusion_matrix(label_ids,predictions)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
            "confusion_matrix": cm
        }

      eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=probabilities, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "fever": FEVERProcessor,
      "ir": DocumentRetrieval,
      "evid_chains": EvidChains,
      "rte": RTEProcessor,
  }

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")


  if FLAGS.do_train and FLAGS.init_checkpoint is None:
    raise ValueError(
        "No init checkpoint for training ")

   
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      # Eval will be slightly WRONG on the TPU because it will truncate
      # the last batch.
      eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    #predict_examples = processor.get_test_examples(FLAGS.data_dir)
    # override later as soon as we have test_examples
    predict_examples = processor.get_dev_examples(FLAGS.data_dir)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    if FLAGS.use_tpu:
      # Warning: According to tpu_estimator.py Prediction on TPU is an
      # experimental feature and hence not supported here
      raise ValueError("Prediction in TPU not supported")

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)


    if FLAGS.file_test_results is None:
      output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    else:
      output_predict_file = os.path.join(FLAGS.output_dir, FLAGS.file_test_results)
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      tf.logging.info("***** Predict results *****")
      for prediction in result:
        output_line = "\t".join(
            str(class_probability) for class_probability in prediction) + "\n"
        writer.write(output_line)
  with open("all_seq_lengths.txt", "w") as outfile:
    for i,j in sorted(all_seq_lengths.items(), reverse=True):
      outfile.write("seq length " + str(i) + " occurrences " + str(j) + "\n")


if __name__ == "__main__":
  all_seq_lengths = defaultdict(int)
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")


  tf.app.run()

