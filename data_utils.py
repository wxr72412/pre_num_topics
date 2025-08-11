# data_utils.py
import os
import random
import jieba.posseg
from gensim.models import Word2Vec
from config import WORD2VEC_SIZE, WINDOW, MIN_COUNT

def load_stopword(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def seg_depart(sentence, stopwords):
    words, nouns = [], []
    for x in jieba.posseg.cut(sentence.strip()):
        if x.word.strip() not in stopwords and len(x.word.strip()) >= 2:
            words.append(x.word.strip())
            if 'n' in x.flag:
                nouns.append(x.word)
    return words, nouns

def load_thuc(path, stopwords, sample_size=100, save_path=None):
    corpus_full, corpus_nouns = [], []
    for _, dirs, _ in os.walk(path):
        for d in dirs:
            files = os.listdir(os.path.join(path, d))
            for i in random.sample(files, min(sample_size, len(files))):
                with open(os.path.join(path, d, i), 'r', encoding='utf-8') as f:
                    text = ''.join(line.strip() for line in f)
                out, nouns = seg_depart(text, stopwords)
                corpus_full.append(out)
                corpus_nouns.append(nouns)

    if save_path:
        model = Word2Vec(corpus_full, vector_size=WORD2VEC_SIZE, window=WINDOW, min_count=MIN_COUNT)
        os.makedirs(save_path, exist_ok=True)
        model.save(os.path.join(save_path, 'w2v'))
        model.wv.save_word2vec_format(os.path.join(save_path, 'vec.txt'), binary=False)
    return corpus_nouns

def load_sohu(path, stopwords):
    corpus = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            words = [w for w in line.split(' ') if w not in stopwords and not w.encode('utf-8').isalnum()]
            corpus.append(words)
    return corpus

def load_mini(path):
    corpus = []
    for root, _, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file), encoding='utf-8') as f:
                words = [w for line in f for w in line.strip().split(' ')]
            corpus.append(words)
    return corpus
