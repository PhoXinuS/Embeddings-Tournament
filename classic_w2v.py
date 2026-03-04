'''
=============
Author notes:
=============

pure numpy implementation of classic word2vec

https://arxiv.org/pdf/1301.3781
https://arxiv.org/pdf/1310.4546


Algorithm Steps:

1. Load the dataset and preprocess it
    - Change every word to a unique integer representation
    - Create Huffmann tree based on the frequency of words in the dataset

2. Create the training data - method to be decided
    Paper described "Subsampling of frequent words"

3. Define the model:
It is important to state that there are two main architectures:
    - Continuous Bag of Words (CBOW): predicts the target word based on multiple context words
        word1 word2 -> target word <- word3 word4
        Faster, less accurate, better for smaller datasets (?)
        
    - Skip-gram: predicts the context words based on one target word
        context words <- target word -> context words
        Slower, more accurate with syntatic tasks

    CBOW:
    - Input layer: multiple one-hot encoded vectors of the context words 
        (one hot size equal to the vocabulary size)
    - Projection layer: linear layer that averages the input vectors to create a single vector representation
    - Output later: a log-linear classifier predicting the middle word, using Hierarchical Softmax (according to the paper)

    Skip-gram:
    - Input layer: one-hot encoded vector of the target word
    - Projection layer: linear layer that creates a single vector representation of the target word
    - Output layer: lof-linear classifier that predicts words within a range before and after the target word
        Once again it uses Hierarchical Softmax to evaluate the output layer
    The context window in this example is choesen dynamically and selected randomly up to the maximum window size each way (ex. 5 in paper)

    
Also, the most important layer is actually a matrix of [embedding_dim X vocab_size]
and then, in CBOW when we input those 4 words as an index to the matrix, then we average them (projection)
this singualr word combination is used to predict the middle word by using softmax

So yes, there are no non-linear functions in this model, it is just a linear dot product and then a softmax at the end


4. Define:
    - Backpropagation
    - Forward pass
    - Optimizer (sgadam? or sgd, original paper used sgd)
    - Loss function (cross-entropy loss?)
    - Linearly decaying learning rate (as described in the paper)

5. Train the model for a specified number of epochs, updating the weights using the optimizer and backpropagation

'''

import numpy as np

class HuffmanNode:
    def __init__(self, frequency, word_index=None, left=None, right=None):
        self.frequency = frequency
        self.word_index = word_index
        self.left = left
        self.right = right

class Word2VecDataset:
    def __init__(self, raw_text_corpus):
        self.raw_text_corpus = raw_text_corpus
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_frequencies = {}
        self.huffman_root = None
        self.processed_corpus = []

    def build_vocabulary(self):
        pass

    def build_huffman_tree(self):
        pass

    def subsample_frequent_words(self, subsampling_threshold):
        pass

    def generate_training_samples(self, max_window_size, architecture):
        pass

class Word2VecModel:
    def __init__(self, vocabulary_size, embedding_dimension, architecture="cbow"):
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.architecture = architecture
        
        self.input_embeddings = np.random.uniform(
            -0.5 / embedding_dimension, 
            0.5 / embedding_dimension, 
            (vocabulary_size, embedding_dimension)
        )
        self.output_weights = np.zeros((vocabulary_size, embedding_dimension))
        
        self.initial_learning_rate = 0.025 if architecture == "skip-gram" else 0.05
        self.current_learning_rate = self.initial_learning_rate

    def forward_cbow(self, context_word_indices):
        pass

    def forward_skip_gram(self, target_word_index):
        pass

    def hierarchical_softmax(self, hidden_layer_representation, target_word_index):
        pass

    def backward_pass(self, loss_gradient, hidden_layer_representation, target_index):
        pass

    def update_weights(self):
        pass

    def update_learning_rate(self, current_word_count, total_word_count):
        pass

    def compute_loss(self, predicted_probabilities, true_labels):
        pass

def train_word2vec(raw_corpus, epochs, embedding_dimension, max_window_size, architecture="cbow"):
    dataset = Word2VecDataset(raw_corpus)
    dataset.build_vocabulary()
    dataset.build_huffman_tree()
    dataset.subsample_frequent_words(subsampling_threshold=1e-3)
    
    model = Word2VecModel(len(dataset.word_to_index), embedding_dimension, architecture)
    
    for epoch in range(epochs):
        training_samples = dataset.generate_training_samples(max_window_size, architecture)
        
        for sample in training_samples:
            if architecture == "cbow":
                context_indices, target_index = sample
                hidden_representation = model.forward_cbow(context_indices)
            else:
                target_index, context_indices = sample
                hidden_representation = model.forward_skip_gram(target_index)
            
            model.backward_pass(
                loss_gradient=None, 
                hidden_layer_representation=hidden_representation,
                context_indices=context_indices,
                target_index=target_index
            )
            model.update_weights()
            
        model.update_learning_rate(current_word_count=0, total_word_count=1)
        
    return model