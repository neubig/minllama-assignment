from contextlib import nullcontext
import json
import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

# change it with respect to the original model
from classifier import LlamaZeroShotClassifier, LlamaEmbeddingClassifier
from llama import Llama, load_pretrained
from optimizer import AdamW
from tokenizer import Tokenizer
from tqdm import tqdm
from typing import Optional


TQDM_DISABLE=False
# fix the random seed
def seed_everything(seed=11711):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

# create a custom Dataset Class to be used for the dataloader
class LlamaDataset(Dataset):
	def __init__(self, dataset, args, eos=False):
		self.dataset = dataset
		self.p = args
		self.tokenizer = Tokenizer(max_len=args.max_sentence_len)
		self.eos = eos

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		ele = self.dataset[idx]
		return ele

	def pad_data(self, data):
		sents = [x[0] for x in data]
		labels = [x[1] for x in data]
		encoding = [self.tokenizer.encode(s, bos=True, eos=self.eos) for s in sents]
		max_length_in_batch = max([len(sentence) for sentence in encoding])
		encoding_padded = [sentence + [self.tokenizer.pad_id] * (max_length_in_batch - len(sentence)) for sentence in encoding]
		token_ids = torch.LongTensor(encoding_padded)
		labels = torch.LongTensor(labels)

		return token_ids, labels, sents

	def collate_fn(self, all_data):

		token_ids, labels, sents = self.pad_data(all_data)
		batched_data = {
				'token_ids': token_ids,
				'labels': labels,
				'sents': sents,
			}

		return batched_data


# create the data which is a list of (sentence, label, token for the labels)
def create_data(filename, tokenizer: Tokenizer, flag: str ='train', lower: bool = False, eos: bool = True, prompt_suffix: Optional[str]=None):
	# specify the tokenizer
	num_labels = {}
	data = []

	with open(filename, 'r') as fp:
		for line in fp:
			label, org_sent = line.split(' ||| ')
			if lower:
				org_sent = org_sent.lower()
			sent = org_sent.strip()
			if prompt_suffix is not None:
				sent = f"{sent} {prompt_suffix}"
			tokens = tokenizer.encode(sent, bos=True, eos=eos)
			label = int(label.strip())
			if label not in num_labels:
				num_labels[label] = len(num_labels)
			data.append((sent, label, tokens))
	print(f"load {len(data)} data from {filename}")
	if flag == 'train':
		return data, len(num_labels)
	else:
		return data

# perform model evaluation in terms of the accuracy and f1 score.
def model_eval(dataloader, model, device):
	model.eval() # switch to eval model, will turn off randomness like dropout
	y_true = []
	y_pred = []
	sents = []
	for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
		b_ids, b_labels, b_sents = batch['token_ids'], batch['labels'], batch['sents']

		b_ids = b_ids.to(device)

		logits = model(b_ids)
		logits = logits.detach().cpu().numpy()
		preds = np.argmax(logits, axis=1).flatten()

		b_labels = b_labels.flatten()
		y_true.extend(b_labels)
		y_pred.extend(preds)
		sents.extend(b_sents)

	f1 = f1_score(y_true, y_pred, average='macro')
	acc = accuracy_score(y_true, y_pred)

	return acc, f1, y_pred, y_true, sents

def save_model(model, optimizer, args, config, filepath):
	save_info = {
		'model': model.state_dict(),
		'optim': optimizer.state_dict(),
		'args': args,
		'model_config': config,
		'system_rng': random.getstate(),
		'numpy_rng': np.random.get_state(),
		'torch_rng': torch.random.get_rng_state(),
	}

	torch.save(save_info, filepath)
	print(f"save the model to {filepath}")

def train(args):
	device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
	#### Load data
	# create the data and its corresponding datasets and dataloader
	tokenizer = Tokenizer(args.max_sentence_len)
	train_data, num_labels = create_data(args.train, tokenizer, 'train')
	dev_data = create_data(args.dev, tokenizer, 'valid')

	train_dataset = LlamaDataset(train_data, args)
	dev_dataset = LlamaDataset(dev_data, args)

	train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
								  collate_fn=train_dataset.collate_fn)
	dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
								collate_fn=dev_dataset.collate_fn)

	#### Init model
	config = {'hidden_dropout_prob': args.hidden_dropout_prob,
			  'pretrained_model_path': args.pretrained_model_path,
			  'num_labels': num_labels,
			  'data_dir': '.',
			  'option': args.option}

	config = SimpleNamespace(**config)

	# initialize the Senetence Classification Model
	model = LlamaEmbeddingClassifier(config)
	model = model.to(device)

	lr = args.lr
	## specify the optimizer
	optimizer = AdamW(model.parameters(), lr=lr)
	best_dev_acc = 0

	## run for the specified number of epochs
	for epoch in tqdm(range(args.epochs)):
		model.train()
		train_loss = 0
		num_batches = 0
		for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
			b_ids, b_labels, b_sents = batch['token_ids'], batch['labels'], batch['sents']

			b_ids = b_ids.to(device)
			b_labels = b_labels.to(device)

			optimizer.zero_grad()
			logits = model(b_ids)
			loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size

			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			num_batches += 1

		train_loss = train_loss / (num_batches)

		train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
		dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

		if dev_acc > best_dev_acc:
			best_dev_acc = dev_acc
			save_model(model, optimizer, args, config, args.filepath)

		print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

def generate_sentence(args, prefix, outfile, max_new_tokens = 75, temperature = 0.0):
	with torch.no_grad():
		device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
		ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float32) if args.use_gpu else nullcontext()
		llama = load_pretrained(args.pretrained_model_path)
		llama = llama.to(device)
		print(f"load model from {args.pretrained_model_path}")
		enc = Tokenizer(args.max_sentence_len)

		start_ids = enc.encode(prefix, bos=True, eos=False)
		x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

		# run generation
		with torch.no_grad():
			with ctx:
				y = llama.generate(x, max_new_tokens, temperature=temperature)
				sentence = enc.decode(y[0].tolist())
				print(f"Temperature is {temperature}")
				print(sentence)
				print('---------------')
				writer = open(outfile, 'w')
				writer.write(sentence)
				print(f"Wrote generated sentence to {outfile}.")
				writer.close()

def write_predictions_to_file(split: str, outfile: str, acc: float, pred: list[str], sents: list[str]):
	with open(outfile, "w+") as f:
		print(f"{split} acc :: {acc :.3f}")
		for s, p in zip(sents, pred):
			f.write(f"{p} ||| {s}\n")

def test_with_prompting(args):
	assert args.dev_out.endswith("dev-prompting-output.txt"), 'For saving prompting results, please set the dev_out argument as "<dataset>-dev-prompting-output.txt"'
	assert args.test_out.endswith("test-prompting-output.txt"), 'For saving prompting results, please set the test_out argument as "<dataset>-test-prompting-output.txt"'

	with torch.no_grad():

		device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
		#### Load data
		# create the data and its corresponding datasets and dataloader
		tokenizer = Tokenizer(args.max_sentence_len)
		label_names = json.load(open(args.label_names, 'r'))
		_, num_labels = create_data(args.train, tokenizer, 'train')

		#### Init model
		config = {'pretrained_model_path': args.pretrained_model_path,
				'label_names': label_names,
				'num_labels': num_labels,
				'data_dir': '.',
				'option': args.option}

		config = SimpleNamespace(**config)

		if len(label_names) == 2:
			label_name_str = " or ".join(label_names)
		else:
			label_name_str = ", ".join(label_names[:-1]) + ", or " + label_names[-1]
		prompt_suffix=f"Is this movie {label_name_str}? This movie is "
		model = LlamaZeroShotClassifier(config, tokenizer, label_names)
		model = model.to(device)

		dev_data = create_data(args.dev, tokenizer, 'valid', eos=False, prompt_suffix=prompt_suffix)
		dev_dataset = LlamaDataset(dev_data, args, eos=False)
		dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

		test_data = create_data(args.test, tokenizer, 'test', eos=False, prompt_suffix=prompt_suffix)
		test_dataset = LlamaDataset(test_data, args, eos=False)
		test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

		dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
		test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

		write_predictions_to_file("dev", args.dev_out, dev_acc, dev_pred, dev_sents)
		write_predictions_to_file("test", args.test_out, test_acc, test_pred, test_sents)

def test(args):
	assert args.dev_out.endswith("dev-finetuning-output.txt"), 'For saving finetuning results, please set the dev_out argument as "<dataset>-dev-finetuning-output.txt"'
	assert args.test_out.endswith("test-finetuning-output.txt"), 'For saving finetuning results, please set the test_out argument as "<dataset>-test-finetuning-output.txt"'
	with torch.no_grad():
		device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
		saved = torch.load(args.filepath)
		config = saved['model_config']
		model = LlamaEmbeddingClassifier(config)
		model.load_state_dict(saved['model'])
		model = model.to(device)
		print(f"load model from {args.filepath}")
		tokenizer = Tokenizer(args.max_sentence_len)
		dev_data = create_data(args.dev, tokenizer, 'valid')
		dev_dataset = LlamaDataset(dev_data, args)
		dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

		test_data = create_data(args.test, tokenizer, 'test')
		test_dataset = LlamaDataset(test_data, args)
		test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

		dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
		test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)
	
		write_predictions_to_file("dev", args.dev_out, dev_acc, dev_pred, dev_sents)
		write_predictions_to_file("test", args.test_out, test_acc, test_pred, test_sents)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", type=str, default="data/cfimdb-train.txt")
	parser.add_argument("--dev", type=str, default="data/cfimdb-dev.txt")
	parser.add_argument("--test", type=str, default="data/cfimdb-test.txt")
	parser.add_argument("--label-names", type=str, default="data/cfimdb-label-mapping.json")
	parser.add_argument("--pretrained-model-path", type=str, default="stories42M.pt")
	parser.add_argument("--max_sentence_len", type=int, default=None)
	parser.add_argument("--seed", type=int, default=1337)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--option", type=str,
						help='prompt: the Llama parameters are frozen; finetune: Llama parameters are updated',
						choices=('generate', 'prompt', 'finetune'), default="generate")
	parser.add_argument("--use_gpu", action='store_true')
	parser.add_argument("--generated_sentence_low_temp_out", type=str, default="generated-sentence-temp-0.txt")
	parser.add_argument("--generated_sentence_high_temp_out", type=str, default="generated-sentence-temp-1.txt")
	parser.add_argument("--dev_out", type=str, default="cfimdb-dev-prompting-output.txt")
	parser.add_argument("--test_out", type=str, default="cfimdb-test-prompting-output.txt")

	# hyper parameters
	parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
	parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
	parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
						default=2e-5)

	args = parser.parse_args()
	print(f"args: {vars(args)}")
	return args

if __name__ == "__main__":
	args = get_args()
	args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt' # save path
	seed_everything(args.seed)  # fix the seed for reproducibility

	if args.option == "generate":
		# Step 1
		# Complete this sentence to test your implementation!
		prefix = "I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"
		generate_sentence(args, prefix, args.generated_sentence_low_temp_out, max_new_tokens=75, temperature=0.0)
		generate_sentence(args, prefix, args.generated_sentence_high_temp_out, max_new_tokens=75, temperature=1.0)
	elif args.option == "prompt":
		# Step 2
		# Solve this task with prompted language modeling
		test_with_prompting(args)
	elif args.option == "finetune":
		# Step 3
		# Finetune a classification model
		train(args)

		# Step 4
		# Evaluate your model on the dev and test sets
		test(args)
	else:
		raise ValueError(f"Invalid option: {args.option}")