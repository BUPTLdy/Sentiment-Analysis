# Chinese Sentiment Analysis

中文商品评论情感分类示例。仓库保留了 2016 年的 Word2Vec + SVM/LSTM 实现和预训练文件，并新增了一个面向现代 Python 3 环境的、可重新训练的 TF-IDF + Linear SVM 基线。

## 项目状态

原始代码依赖 Python 2.7、Keras 1、TensorFlow 1 和早期版本的 gensim/scikit-learn。由于这些框架及序列化格式已经发生重大变化，旧模型仅作为研究和复现实验的历史资料保留，不应直接加载不可信的 `.pkl`、`.h5` 或 YAML 文件。

新的 [`sentiment_cli.py`](sentiment_cli.py) 具备以下特点：

- 支持 Python 3.9 及以上版本；
- 使用字符级 TF-IDF，不依赖中文分词词典；
- 使用 Linear SVM 训练和预测；
- 从仓库现有正负评论数据重新训练模型；
- 保存训练时间、数据量、评估准确率和运行环境等元数据；
- 默认不提交生成的模型文件，避免把本地序列化产物误当成可信发布物。

## 快速开始

创建独立环境并安装现代依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

训练模型：

```bash
python sentiment_cli.py train
```

默认模型会写入 `artifacts/sentiment_svm.joblib`。训练命令会在固定随机种子下划分测试集，并打印 JSON 格式的评估摘要。

预测一条或多条评论：

```bash
python sentiment_cli.py predict "这个手机质量很好，我很喜欢"
python sentiment_cli.py predict "酒店环境不错" "电脑散热太差了" --json
```

指定模型路径：

```bash
python sentiment_cli.py train --model /path/to/model.joblib
python sentiment_cli.py predict --model /path/to/model.joblib "物流很快"
```

## 数据

- [`data/pos.xls`](data/pos.xls)：正向评论。
- [`data/neg.xls`](data/neg.xls)：负向评论。

这些数据来自早期实验，仅适合教学和基线比较。实际应用前应重新确认来源、许可、领域偏差和隐私要求，并使用目标领域数据重新评估。

## 旧版实现

- [`code/Sentiment_svm.py`](code/Sentiment_svm.py)：Word2Vec + RBF SVM。
- [`code/Sentiment_lstm.py`](code/Sentiment_lstm.py)：Word2Vec + Keras LSTM。
- [`predict.py`](predict.py)：旧版 Python 2 预测入口。
- [`requirements-legacy.txt`](requirements-legacy.txt)：原始依赖版本记录，仅用于历史研究。

旧版模型与现代库通常不兼容。推荐通过新的 Python 3 CLI 重新训练，而不是尝试在生产环境中恢复旧运行栈。

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall sentiment_cli.py tests
```

## 参与维护

欢迎改进评估方法、数据说明、命令行体验和现代模型基线。提交前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。处理模型文件或数据时请同时阅读 [SECURITY.md](SECURITY.md)。

## 背景文章

[购物评论情感分析的实现](https://buptldy.github.io/2016/07/20/2016-07-20-sentiment%20analysis/)
