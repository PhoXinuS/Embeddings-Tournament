# Embeddings-Tournament

This is a pure NumPy implementation of both CBOW and Skip-Gram from the original paper:

[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781)


[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546)

| Architecture | Words/Sec | WordSim (ρ) | Analogy Acc |
|--------------|-----------|-------------|-------------|
| cbow         |   4,036   |    0.272    |    1.4%     |
| skip-gram    |     976   |    0.160    |    0.2%     |

![](benchmark.png)