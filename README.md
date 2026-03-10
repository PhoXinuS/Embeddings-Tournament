# Embeddings-Tournament

This is a pure NumPy implementation of both CBOW and Skip-Gram from the original paper:

[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781)


[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546)

### 25 MB | 1 epoch:

| Architecture | Words/Sec | WordSim (ρ) | Analogy Acc |
|--------------|-----------|-------------|-------------|
| cbow         |   4,036   |    0.272    |    1.4%     |
| skip-gram    |     976   |    0.160    |    0.2%     |

![](benchmark_25_1.png)

As those results may seem surprising, especially in regards to the analogy accuracy, this is still a very small success rate. And from my understanding the skip-gram starts to perform with more epochs and training data, at least by looking at random graphs from much more sophisticated runs. Additionally, keeping in mind that it only trained on 1 epoch with 25MB of data due to time and computational constraints, this result is understandable. More minor experiments proved that both models are improving with more data, so the system is still training and tuning the embeddings.

It seems that further tests on larger data samples and with more epochs proved that conclusion :)

### 10 MB | 3 epochs:
| Architecture | Words/Sec | WordSim (ρ) | Analogy Acc |
|--------------|-----------|-------------|-------------|
| cbow         | 4,531     | 0.302       | 1.3%        |
| skip-gram    | 1,641     | 0.478       | 3.9%        |

![](benchmark_10_3.png)

### 20 MB | 10 epochs:
| Architecture | Words/Sec | WordSim (ρ) | Analogy Acc |
|--------------|-----------|-------------|-------------|
| cbow         | 8,558     | 0.464       | 6.8%        |
| skip-gram    | 1,819     | 0.585       | 11.2%       |

![](benchmark_20_10.png)

### Datasets:
This model has been trained on the first n-MB od the [text8](https://www.kaggle.com/datasets/gupta24789/text8-word-embedding) (Wikipedia) lowercased words. For testing purposes it utilized the [wordsim353](https://gabrilovich.com/resources/data/wordsim353/wordsim353.html) and [questions-words](http://download.tensorflow.org/data/questions-words.txt) 