# Code for **An efficient method for predicting the number of topics** 

# 代码：**一种高效的主题数目预测方法** 

## 1. Code structure 代码结构 

├── data/ # 存放数据集
│ ├── mini/ # 20Newsgroups数据（英文）
│ ├── THUCNews/ # 清华新闻数据集（中文）
│ └── sohu/ # 搜狐新闻数据集（中文）
├── requirements.txt # 环境依赖配置文件
├── config.py # 全局配置文件，包含路径和超参数
├── data_utils.py # 数据加载与预处理工具
├── main.py # 主题建模主程序，训练与评估流程
└── metrics.py # 多种主题模型指标计算函数

## 2. Datasets 数据集

(1) 20Newsgroup新闻数据集(http://qwone.com/~jason/20Newsgroups/)

(2) 清华新闻数据集(http://thuctc.thunlp.org/message)
(3) 搜狐新闻数据集(https://gitcode.com/Resource-Bundle-Collection/db22d8)

## 3. parameter configuration 参数配置

config.py 

## 4. Run 运行程序

```python
python main.py
```

