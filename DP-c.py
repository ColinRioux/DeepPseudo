import os

import tqdm
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from torchtext.legacy.data import Dataset, Example, Field
from torchtext.legacy.data.iterator import BucketIterator
from torchtext.data.metrics import bleu_score
from torch.nn import Parameter
from transformers import RobertaTokenizer

from torchtext.legacy import data
from beam_utils import Node, find_best_path, find_path

import argparse
import pickle as pkl
from src.PGD import PGD
from src.ScaleUp import ScaleUp
from src.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from src.PositionWiseFeedForwardLayer import PositionWiseFeedForwardLayer
from src.PositionalEncodingLayer import PositionalEncodingLayer
from src.EncoderLayer import EncoderLayer
from src.DecoderLayer import DecoderLayer
from src.EncoderBlockLayer import EncoderBlockLayer
from src.DecoderBlockLayer import DecoderBlockLayer
from src.Trainer import Trainer
from src.Transformer import Transformer
from src.AverageMeter import AverageMeter

arg_parser = argparse.ArgumentParser(description='Execute DeepPseudo')
arg_parser.add_argument('--train', action='store_true', help='Train or not train')
arg_parser.add_argument('--data_path', help='The location of the test set', default='data/django')
arg_parser.add_argument('--file_type', help='File type of the datasets', default='csv')
arg_parser.add_argument('--batch_size', default=64, type=int)
arg_parser.add_argument('--model_path', help='The location of the trained model', default='model/django.pth')
args = arg_parser.parse_args()

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

SEED = 546
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

"""
Hyperparams
"""
sep = ','
if args.file_type == 'tsv':
    sep = '\t'

train = args.train
data_dir = args.data_path
test_path = "test." + args.file_type
save_path = args.model_path
D_MODEL = 256
N_LAYERS = 3
N_HEADS = 8
HIDDEN_SIZE = 512
CODE_MAX_LEN = 50
NL_MAX_LEN = 50
KERNEL_SIZE = 3
DROPOUT = 0.25
SCALE = np.sqrt(0.5)
BATCH_SIZE = args.batch_size
LR = 1e-3
N_EPOCHS = 40
GRAD_CLIP = 1.0
print(f'Training: {train}')
print(f'Data Directory: {data_dir}')
print(f'Model Save: {save_path}')
print(f'Batch Size: {BATCH_SIZE}')

def tokenize_code(text):
    return text.split()

def tokenize_nl(text):
    text = text.replace("_"," ")
    text = text.replace("."," ")
    return text.split()

def read_vocab(pth):
    with open(pth, 'rb') as f:
        return pkl.load(f)

CODE = Field(tokenize = tokenize_code,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

NL = Field(tokenize = tokenize_code,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

test_data = data.TabularDataset(path=os.path.join(data_dir,test_path), format=args.file_type, skip_header=True, fields=[('sc',CODE)]) 

MIN_COUNT = 1
#CODE.build_vocab(train_data, min_freq=MIN_COUNT)
CODE.vocab = read_vocab("./data/staqc/vocab.pickle")
NL.vocab = read_vocab("./data/nl_vocab.pkl")
print(f'Length of CODE vocabulary: {len(CODE.vocab):,}')

test_iterator = BucketIterator(test_data, batch_size=BATCH_SIZE, sort_within_batch=True, sort_key=(lambda x: len(x.sc)), device=DEVICE)

threshold_wordcount = np.percentile([len(i) for i in test_data.sc], 97.5)

def scaling_factor(sequence_threshold):
    return np.log2((sequence_threshold ** 2) - sequence_threshold)

def accuracy(outputs, target_sequences, k=5):
    """ Calculate Top-k accuracy
    :param Tensor[batch_size, dest_seq_len, vocab_size] outputs
    :param Tensor[batch_size, dest_seq_len] target_sequences
    :return float Top-k accuracy
    """
    # print([*map(lambda token: EN.vocab.itos[token], outputs.argmax(dim=-1)[0].tolist())])
    # print([*map(lambda token: EN.vocab.itos[token], target_sequences[0].tolist())])
    # print("="*100)
    batch_size = target_sequences.size(0)
    _, indices = outputs.topk(k, dim=2, largest=True, sorted=True)  # [batch_size, dest_seq_len, 5]
    correct = indices.eq(target_sequences.unsqueeze(-1).expand_as(indices))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / indices.numel())

transformer = Transformer(
    encoder=EncoderLayer(
        vocab_size=len(CODE.vocab),
        max_len=CODE_MAX_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        hidden_size=HIDDEN_SIZE,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        n_layers=N_LAYERS,
        scale = SCALE
    ),
    decoder=DecoderLayer(
        vocab_size=len(NL.vocab),
        max_len=NL_MAX_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        n_layers=N_LAYERS
    ),
    src_pad_index=CODE.vocab.stoi[CODE.pad_token],
    dest_pad_index=NL.vocab.stoi[NL.pad_token]
).to(DEVICE)

# print(transformer)
optimizer = optim.AdamW(params=transformer.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=NL.vocab.stoi[NL.pad_token])
print(f'Number of parameters of the model: {sum(p.numel() for p in transformer.parameters() if p.requires_grad):,}')
pgd = PGD(transformer)
K = 3

if(train):
    if(os.path.exists(save_path)):
        transformer.load_state_dict(torch.load(save_path))
        transformer.to(DEVICE)
    trainer = Trainer(model=transformer, optimizer=optimizer, criterion=criterion)
    history = trainer.train(train_loader=train_iterator, valid_loader=valid_iterator, n_epochs=N_EPOCHS, grad_clip=GRAD_CLIP)

transformer.load_state_dict(torch.load(save_path))
transformer.to(DEVICE)


def translate(sentences, model, beam_size, src_field, dest_field, max_len, device):
    if isinstance(sentences, list):
        sentences = [*map(src_field.preprocess, sentences)]
        targets = None
    if isinstance(sentences, Dataset):
        targets = [*map(lambda example: ' '.join(example.trg), sentences.examples)]
        sentences = [*map(lambda example: example.src, sentences.examples)]
    data = [*map(lambda word_list: src_field.process([word_list]), sentences)]
    translated_sentences, attention_weights, pred_logps = [], [], []
    model.eval()
    with torch.no_grad():
        for i, src_sequence in tqdm.tqdm(enumerate(data), total=len(data)):
            src_sequence = src_sequence.to(device)
            src_mask = model.make_src_mask(src_sequence)
            src_encoded = model.encoder(src_sequences=src_sequence, src_mask=src_mask)
            tree = [
                [Node(token=torch.LongTensor([dest_field.vocab.stoi[dest_field.init_token]]).to(device), states=())]]
            for _ in range(max_len):
                next_nodes = []
                for node in tree[-1]:
                    if node.eos:  # Skip eos token
                        continue
                    # Get tokens that're already translated
                    already_translated = torch.LongTensor(
                        [*map(lambda node: node.token, find_path(tree))][::-1]).unsqueeze(0).to(device)
                    dest_mask = model.make_dest_mask(already_translated)
                    logit, attn_weights = model.decoder(dest_sequences=already_translated, src_encoded=src_encoded,
                                                        dest_mask=dest_mask,
                                                        src_mask=src_mask)  # [1, dest_seq_len, vocab_size]
                    logp = F.log_softmax(logit[:, -1, :], dim=1).squeeze(
                        dim=0)  # [vocab_size] Get scores
                    topk_logps, topk_tokens = torch.topk(logp,
                                                         beam_size)  # Get top k tokens & logps
                    for k in range(beam_size):
                        next_nodes.append(Node(token=topk_tokens[k, None], states=(attn_weights,),
                                               logp=topk_logps[k, None].cpu().item(), parent=node,
                                               eos=topk_tokens[k].cpu().item() == dest_field.vocab[
                                                   dest_field.eos_token]))
                if len(next_nodes) == 0:
                    break
                next_nodes = sorted(next_nodes, key=lambda node: node.logps, reverse=True)
                tree.append(next_nodes[:beam_size])
            best_path = find_best_path(tree)[::-1]
            # Get the translation
            pred_translated = [*map(lambda node: dest_field.vocab.itos[node.token], best_path)]
            pred_translated = [*filter(lambda word: word not in [
                dest_field.init_token, dest_field.eos_token
            ], pred_translated)]
            translated_sentences.append(' '.join(pred_translated))
            # Get probabilities
            pred_logps.append(sum([*map(lambda node: node.logps, best_path)]))
            # Get attention weights
            attention_weights.append(best_path[-1].states[0].cpu().numpy())
        sentences = [*map(lambda sentence: ' '.join(sentence), sentences)]
    return sentences, translated_sentences, targets, attention_weights, pred_logps

sentences, translated_sentences, dest_sentences, attention_weights, pred_logps = translate(sentences=test_data, model=transformer, beam_size=3, src_field=CODE,
                                                                                           dest_field=NL, max_len=NL_MAX_LEN, device=DEVICE)

import pandas as pd
column_name = ['comment']
nl_df = pd.DataFrame(dest_sentences, columns=column_name)
nl_df.to_csv('result/true_pseudo.' + args.file_type, sep=sep, index=None, header=False)
pred_df = pd.DataFrame(translated_sentences, columns=column_name)
pred_df.to_csv('result/pred_pseudo.' + args.file_type, sep=sep, index=None, header=False)

from nlgeval import compute_metrics
metrics_dict = compute_metrics(hypothesis='result/pred_pseudo.' + args.file_type,
        references=['result/true_pseudo.' + args.file_type])
