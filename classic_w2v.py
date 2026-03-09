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
from collections import Counter
import heapq
import math
import random
import time

class HuffmanNode:
    def __init__(self, frequency, word_index=None, left=None, right=None):
        self.frequency = frequency
        self.word_index = word_index
        self.left = left
        self.right = right
        
    # Required for heapq to compare nodes based on frequency
    def __lt__(self, other):
        return self.frequency < other.frequency

class Word2VecDataset:
    def __init__(self, raw_text_corpus):
        self.raw_text_corpus = raw_text_corpus
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_frequencies = {}
        self.huffman_root = None
        self.processed_corpus = []
        self.word_paths = {}
        self.word_codes = {}
        self.total_words = 0


    # 1. Load the dataset and preprocess it
    def build_vocabulary(self):
        # Change every word to a unique integer representation
        words = self.raw_text_corpus.split()
        self.total_words = len(words)
        word_counts = Counter(words)
        
        for index, (word, count) in enumerate(word_counts.items()):
            self.word_to_index[word] = index
            self.index_to_word[index] = word
            self.word_frequencies[index] = count
            
        self.processed_corpus = [self.word_to_index[w] for w in words]

    def build_huffman_tree(self):
        # Create Huffman tree based on word frequencies.
        # We use heapq (priority queue) for optimal O(N log N) tree construction.
        priority_queue = [
            HuffmanNode(freq, word_index=idx) 
            for idx, freq in self.word_frequencies.items()
        ]
        heapq.heapify(priority_queue)
        
        node_index = len(self.word_frequencies)
        
        while len(priority_queue) > 1:
            left_node = heapq.heappop(priority_queue)
            right_node = heapq.heappop(priority_queue)
            
            merged_node = HuffmanNode(
                left_node.frequency + right_node.frequency, 
                word_index=node_index, 
                left=left_node, 
                right=right_node
            )
            node_index += 1
            heapq.heappush(priority_queue, merged_node)
            
        self.huffman_root = priority_queue[0]
        self._generate_huffman_paths(self.huffman_root, [], [])

    def _generate_huffman_paths(self, node, current_path, current_code):
        # Traverse the tree to generate binary codes and node paths for Hierarchical Softmax
        if node is None:
            return
        if node.left is None and node.right is None:
            self.word_paths[node.word_index] = current_path
            self.word_codes[node.word_index] = current_code
            return
        
        self._generate_huffman_paths(node.left, current_path + [node.word_index], current_code + [1])
        self._generate_huffman_paths(node.right, current_path + [node.word_index], current_code + [0])

    # 2: Create the training data
    def subsample_frequent_words(self, subsampling_threshold=1e-3):
        # "Subsampling of frequent words" as described in the paper.
        # This formula aggressively drops highly frequent words (like "the", "a") 
        # to speed up training and improve representations of rare words.
        subsampled_corpus = []
        for word_idx in self.processed_corpus:
            frequency_fraction = self.word_frequencies[word_idx] / self.total_words
            retention_probability = min(1.0, math.sqrt(subsampling_threshold / frequency_fraction))
            
            if random.random() < retention_probability:
                subsampled_corpus.append(word_idx)
                
        self.processed_corpus = subsampled_corpus

    def generate_training_samples(self, max_window_size, architecture):
        # Context window is chosen dynamically and randomly up to max_window_size
        samples = []
        for i, target_word in enumerate(self.processed_corpus):
            dynamic_window = random.randint(1, max_window_size)
            start = max(0, i - dynamic_window)
            end = min(len(self.processed_corpus), i + dynamic_window + 1)
            
            context_words = self.processed_corpus[start:i] + self.processed_corpus[i+1:end]
            if not context_words:
                continue
                
            if architecture == "cbow":
                # CBOW: multiple context words -> 1 target word
                samples.append((context_words, target_word))
            elif architecture == "skip-gram":
                # Skip-gram: 1 target word -> 1 context word (creates multiple pairs)
                for context_word in context_words:
                    samples.append((target_word, context_word))
        return samples

# 3, 4. Define the model, Forward, Backprop, Optimizer
class Word2VecModel:
    def __init__(self, vocabulary_size, embedding_dimension, dataset, architecture="cbow"):
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.architecture = architecture
        self.dataset = dataset
        
        # Input layer: Matrix of [vocab_size X embedding_dim]
        # Initialized with small random values as per standard practices
        self.input_embeddings = np.random.uniform(
            -0.5 / embedding_dimension, 
            0.5 / embedding_dimension, 
            (vocabulary_size, embedding_dimension)
        )
        
        # Output layer for Hierarchical Softmax. 
        # The number of internal nodes is strictly (vocab_size - 1), but we allocate vocab_size * 2 for safety based on tree indices.
        internal_nodes_count = vocabulary_size * 2
        self.output_weights = np.zeros((internal_nodes_count, embedding_dimension))
        
        # Initial learning rate based on original C implementation
        self.initial_learning_rate = 0.025 if architecture == "skip-gram" else 0.05
        self.current_learning_rate = self.initial_learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def forward_cbow(self, context_word_indices):
        # Projection layer: averages the input vectors of context words
        hidden_layer_representation = np.mean(self.input_embeddings[context_word_indices], axis=0)
        return hidden_layer_representation

    def forward_skip_gram(self, target_word_index):
        # Projection layer: directly passes the target word vector
        hidden_layer_representation = self.input_embeddings[target_word_index]
        return hidden_layer_representation

    def backward_pass_and_update(self, hidden_layer_representation, target_index, context_indices=None):
        # To avoid storing large intermediate gradients, we update weights immediately (SGD).
        # This closely matches Mikolov's original C implementation.
        path = self.dataset.word_paths[target_index]
        codes = self.dataset.word_codes[target_index]
        
        hidden_gradient = np.zeros(self.embedding_dimension)
        
        # Iterate through the path in the Huffman tree
        for node_index, code in zip(path, codes):
            node_vector = self.output_weights[node_index]
            dot_product = np.dot(hidden_layer_representation, node_vector)
            prediction = self.sigmoid(dot_product)
            
            # Derivative of log-loss with sigmoid simplifies to (label - prediction)
            error = code - prediction
            gradient_step = self.current_learning_rate * error
            
            # Accumulate gradient for the hidden layer (input embeddings)
            hidden_gradient += gradient_step * node_vector
            
            # Update output weights (internal nodes of the tree)
            self.output_weights[node_index] += gradient_step * hidden_layer_representation
            
        # Update input embeddings based on the architecture
        if self.architecture == "cbow":
            for context_idx in context_indices:
                self.input_embeddings[context_idx] += hidden_gradient / len(context_indices)
        else:
            self.input_embeddings[context_indices] += hidden_gradient

    def update_learning_rate(self, current_word_count, total_word_count):
        # Linearly decaying learning rate
        progress = current_word_count / total_word_count
        self.current_learning_rate = self.initial_learning_rate * (1 - progress)
        # Prevent LR from becoming zero or negative
        if self.current_learning_rate < self.initial_learning_rate * 0.0001:
            self.current_learning_rate = self.initial_learning_rate * 0.0001

# 5.Training Loop
def train_word2vec(raw_corpus, epochs, embedding_dimension, max_window_size, architecture="cbow"):
    dataset = Word2VecDataset(raw_corpus)
    dataset.build_vocabulary()
    dataset.build_huffman_tree()
    dataset.subsample_frequent_words()

    model = Word2VecModel(len(dataset.word_to_index), embedding_dimension, dataset, architecture)
    total_samples = len(dataset.processed_corpus) * epochs
    processed_words = 0

    start_time = time.time()
    percent_step = 5
    next_percent = percent_step

    for epoch in range(epochs):
        for i, center_word in enumerate(dataset.processed_corpus):
            processed_words += 1
            if processed_words % 1000 == 0:
                model.update_learning_rate(processed_words, total_samples)

            dynamic_window = random.randint(1, max_window_size)
            start = max(0, i - dynamic_window)
            end = min(len(dataset.processed_corpus), i + dynamic_window + 1)
            context_words = dataset.processed_corpus[start:i] + dataset.processed_corpus[i+1:end]

            if not context_words:
                continue

            if architecture == "cbow":
                hidden = model.forward_cbow(context_words)
                model.backward_pass_and_update(hidden, center_word, context_indices=context_words)
            else:
                hidden = model.forward_skip_gram(center_word)
                for context_word in context_words:
                    model.backward_pass_and_update(hidden, context_word, context_indices=center_word)

            if processed_words >= (next_percent * total_samples // 100):
                elapsed = time.time() - start_time
                print(f"Trained {next_percent}% ({processed_words}/{total_samples}) - {elapsed:.1f}s elapsed")
                next_percent += percent_step

    return model, dataset


if __name__ == "__main__":
    print("Loading corpus...")
    with open("dataset/text8", "r", encoding="utf-8") as f:
        raw_corpus = f.read(1024 * 1024)  # 1MB

    print("Loaded corpus length:", len(raw_corpus))

    # Train with small parameters for a quick test
    print("Building vocabulary...")
    dataset = Word2VecDataset(raw_corpus)
    dataset.build_vocabulary()
    print("Vocabulary size:", len(dataset.word_to_index))
    print("Sample words:", list(dataset.word_to_index.keys())[:10])

    print("Building Huffman tree...")
    dataset.build_huffman_tree()
    print("Huffman tree built. Root frequency:", dataset.huffman_root.frequency)

    print("Subsampling frequent words...")
    dataset.subsample_frequent_words()
    print("Corpus size after subsampling:", len(dataset.processed_corpus))

    print("Initializing model...")
    model = Word2VecModel(len(dataset.word_to_index), 50, dataset, architecture="cbow")

    total_samples = len(dataset.processed_corpus)
    print(f"Training for 1 epoch, {total_samples} samples...")
    training_samples = dataset.generate_training_samples(2, "cbow")

    processed_words = 0
    for sample in training_samples:
        context_indices, target_index = sample
        hidden_representation = model.forward_cbow(context_indices)
        model.backward_pass_and_update(hidden_representation, target_index, context_indices=context_indices)
        processed_words += 1
        if processed_words % 1000 == 0:
            model.update_learning_rate(processed_words, total_samples)
            print(f"Processed {processed_words}/{total_samples} samples. Current learning rate: {model.current_learning_rate:.5f}")

    print("\nSample word embeddings:")
    for idx in range(5):
        word = dataset.index_to_word[idx]
        embedding = model.input_embeddings[idx]
        print(f"{word}: {embedding[:5]}...")