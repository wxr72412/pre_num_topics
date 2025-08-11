# main.py
import os
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LsiModel, Nmf
from config import *
from data_utils import load_stopword, load_thuc, load_sohu, load_mini
from metrics import (
    compute_coherence, compute_perplexity, compute_AICDR,
    compute_sBIC, compute_TS, compute_IA
)
import pandas as pd

def get_topic_model(corpus, num_topics, model_type):
    dictionary = Dictionary(corpus)
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]

    if model_type == 'lda':
        model = LdaModel(corpus=bow_corpus, id2word=dictionary, alpha='auto', eta='auto',
                         iterations=ITERATIONS, num_topics=num_topics, passes=PASSES, eval_every=EVAL_EVERY)
    elif model_type == 'lsi':
        model = LsiModel(corpus=bow_corpus, id2word=dictionary, num_topics=num_topics)
    elif model_type == 'nmf':
        model = Nmf(corpus=bow_corpus, id2word=dictionary, num_topics=num_topics)
    else:
        raise ValueError("Unsupported model type.")

    return model, dictionary, bow_corpus

def run_experiment():
    stopwords = load_stopword(STOPWORDS_PATH)
    corpus = load_thuc(THUCNEWS_PATH, stopwords, SAMPLE_SIZE, save_path=SAVE_PATH)

    results = []
    for k in range(MIN_K, MAX_K + 1):
        model, dictionary, bow_corpus = get_topic_model(corpus, k, 'lda')

        metrics = compute_coherence(model, corpus, dictionary)
        metrics['perplexity'] = compute_perplexity(model, bow_corpus)
        metrics['AICDR'] = compute_AICDR(model, corpus, k)
        metrics['sBIC'] = compute_sBIC(model, corpus, k)

        topics_words = [[w.split('*')[1].strip().replace('"', '') for w in topic_str.split('+')]
                        for _, topic_str in model.show_topics(num_topics=k, num_words=TOPIC_WORDS_NUM)]
        topic_docs = [[] for _ in range(k)]
        for i, doc in enumerate(model.get_document_topics(bow_corpus)):
            best_topic = max(doc, key=lambda x: x[1])[0]
            topic_docs[best_topic].append(corpus[i])

        metrics['TS'] = compute_TS(model, corpus, k)
        metrics['IA'] = compute_IA(topics_words, topic_docs, os.path.join(SAVE_PATH, 'w2v'))

        results.append({'k': k, **metrics})

    df = pd.DataFrame(results)
    os.makedirs(SAVE_PATH, exist_ok=True)
    df.to_excel(os.path.join(SAVE_PATH, 'metrics.xlsx'), index=False)
    print("实验完成，结果已保存到 metrics.xlsx")

if __name__ == "__main__":
    run_experiment()
