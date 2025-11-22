import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
nltk.download('punkt',quiet=True)
nltk.download('wordnet',quiet=True)

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###
    text = example["text"]
    words = word_tokenize(text)
    new_words = []
    
    for w in words:
        prob = random.random()

        # 1. 25% 機率換同義字
        if prob < 0.25:
            syns = wordnet.synsets(w)
            if syns:
                lemmas = [l.name().replace("_", " ") for l in syns[0].lemmas()]
                lemmas = [s for s in lemmas if s.lower() != w.lower()]
                if lemmas:
                    synonym = random.choice(lemmas)
                    new_words.append(synonym)
                    continue

        # 2. 另外 10% 機率加 typo（把最後一個字母 duplicate 或換成附近鍵）
        elif prob < 0.35 and w.isalpha() and len(w) > 3:
            new_w = list(w)
            new_w[-1] = random.choice("abcdefghijklmnopqrstuvwxyz")
            new_words.append("".join(new_w))
            continue

        # 無變化
        new_words.append(w)

    detok = TreebankWordDetokenizer()
    example["text"] = detok.detokenize(new_words)
    return example
    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.


