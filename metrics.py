# metrics.py
import math
import numpy as np
from gensim.models import CoherenceModel, Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from config import TOPIC_WORDS_NUM

# =============== 主题一致性 & 困惑度 ===============
def compute_coherence(model, corpus, dictionary):
    return {
        'c_v': CoherenceModel(model=model, texts=corpus, dictionary=dictionary, coherence='c_v').get_coherence(),
        'c_uci': CoherenceModel(model=model, texts=corpus, dictionary=dictionary, coherence='c_uci').get_coherence(),
        'c_npmi': CoherenceModel(model=model, texts=corpus, dictionary=dictionary, coherence='c_npmi').get_coherence()
    }

def compute_perplexity(model, bow_corpus):
    return model.log_perplexity(bow_corpus)

# =============== IA 指标 ===============
def count_importance(content_docs, topic_words):
    if not content_docs:
        return 0.0
    docs_joined = [' '.join(doc) for doc in content_docs]
    tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    sparse_matrix = tfidf_model.fit_transform(docs_joined)
    score = 0.0
    for w in topic_words:
        idx = tfidf_model.vocabulary_.get(w.lower())
        if idx is not None:
            score += sparse_matrix[:, idx].sum()
    return score

def count_independence(topic_words, all_topics, w2v_model):
    distances = []
    for other_topic in all_topics:
        if other_topic == topic_words:
            continue
        word_dists = []
        for w1 in topic_words:
            if w1 not in w2v_model.wv:
                continue
            other_dists = [w2v_model.wv.distance(w1, w2) for w2 in other_topic if w2 in w2v_model.wv]
            if other_dists:
                word_dists.append(np.mean(other_dists))
        if word_dists:
            distances.append(np.mean(word_dists))
    return np.mean(distances) if distances else 0.0

def compute_IA(topics, topic_docs, w2v_model_path):
    w2v_model = Word2Vec.load(w2v_model_path)
    imps, inds = [], []
    for i, topic_words in enumerate(topics):
        imps.append(count_importance(topic_docs[i], topic_words))
        inds.append(count_independence(topic_words, topics, w2v_model))
    imp_arr, ind_arr = np.array(imps), np.array(inds)
    imp_norm = (imp_arr - imp_arr.min()) / (imp_arr.max() - imp_arr.min() + 1e-9)
    ind_norm = (ind_arr - ind_arr.min()) / (ind_arr.max() - ind_arr.min() + 1e-9)
    IA_scores = imp_norm * ind_norm
    return float(np.mean(IA_scores))



# =============== 其他指标 ===============
# ====== 辅助函数，用于compute_AICDR ======
def compute_diameter(vectors):
    centroid = np.mean(vectors, axis=0)
    distances = np.sqrt(np.sum((vectors - centroid) ** 2, axis=1))
    return np.sum(distances)

def compute_inter_class_distance(vectors_p, vectors_q):
    D_p = compute_diameter(vectors_p)
    D_q = compute_diameter(vectors_q)
    vectors_merged = np.vstack((vectors_p, vectors_q))
    D_pq = compute_diameter(vectors_merged)
    return np.sqrt(max(D_pq - D_p - D_q, 0))

def compute_AICDR(model, bow_corpus, num_topics):
    corpus = bow_corpus

    tfidf = TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
    docs_joined = [" ".join(doc) for doc in corpus]
    tfidf_matrix = tfidf.fit_transform(docs_joined).toarray()

    Z = ward(pairwise_distances(tfidf_matrix, metric='euclidean'))
    aicdr_values = []
    for k in range(num_topics - 1, num_topics + 2):
        labels = fcluster(Z, k, criterion='maxclust')
        unique_labels = np.unique(labels)
        ave_dis_list = []
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                idx_i = np.where(labels == unique_labels[i])[0]
                idx_j = np.where(labels == unique_labels[j])[0]
                vectors_i = tfidf_matrix[idx_i]
                vectors_j = tfidf_matrix[idx_j]
                dist = compute_inter_class_distance(vectors_i, vectors_j)
                ave_dis_list.append(dist)
        ave_dis = np.mean(ave_dis_list) if ave_dis_list else 0
        aicdr_values.append(ave_dis)
    aicdr_changes = [abs(aicdr_values[i + 1] - aicdr_values[i]) for i in range(len(aicdr_values) - 1)]
    return aicdr_changes[0] if aicdr_changes else 0


def compute_sBIC(model, bow_corpus, num_topics):
    corpus = bow_corpus

    tfidf = TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
    docs_joined = [" ".join(doc) for doc in corpus]
    X = tfidf.fit_transform(docs_joined)

    from sklearn.decomposition import LatentDirichletAllocation
    lda = LatentDirichletAllocation(n_components=num_topics, learning_method='online', random_state=0)
    lda.fit(X)
    log_likelihood = lda.score(X)
    n_docs = X.shape[0]
    bic = log_likelihood * n_docs - 0.5 * num_topics * np.log(n_docs)
    return bic


def compute_TS(model, bow_corpus, num_topics):
    corpus = bow_corpus
    # 获取词频统计
    from collections import Counter
    word_freq = Counter()
    for doc in corpus:
        word_freq.update(doc)
    vocab = list(word_freq.keys())
    N = len(vocab)
    sorted_items = sorted(word_freq.items(), key=lambda x: x[1])
    freq_to_min_rank = {}
    current_rank = 1
    for word, freq in sorted_items:
        if freq not in freq_to_min_rank:
            freq_to_min_rank[freq] = current_rank
        current_rank += 1
    rho_dict = {word: freq_to_min_rank[freq] / N for word, freq in word_freq.items()}

    # 从model获取主题词
    raw_topics = model.show_topics(num_topics=num_topics, num_words=TOPIC_WORDS_NUM, formatted=True)
    topic_words = []
    for t in raw_topics:
        words = [w.split('*')[1].strip().strip('"') for w in t[1].split('+')]
        topic_words.append(words)

    total_rho = 0
    count = 0
    top_M = TOPIC_WORDS_NUM
    for tw in topic_words:
        for w in tw[:top_M]:
            if w in rho_dict:
                total_rho += rho_dict[w]
                count += 1
    return total_rho / count if count > 0 else 0