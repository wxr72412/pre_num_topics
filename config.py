# config.py
# 全局配置参数

# 数据路径
STOPWORDS_PATH = 'stopwords.txt'
THUCNEWS_PATH = 'data/THUCNews'
SOHU_PATH = 'data/sohu/cnews.train_jieba.txt'
MINI_PATH = 'data/mini'

# 保存路径
SAVE_PATH = 'results/'

# 词向量训练参数
WORD2VEC_SIZE = 100
WINDOW = 5
MIN_COUNT = 4

# 主题建模参数
PASSES = 20
ITERATIONS = 100
EVAL_EVERY = 1

# 实验参数
MIN_K = 4
MAX_K = 50
ORDER = 1     # 重复实验次数
SAMPLE_SIZE = 100  # load_thuc 采样文件数量

# 主题词数 L（从结论得出，设置为15能更快得到主题数上界）
TOPIC_WORDS_NUM = 15