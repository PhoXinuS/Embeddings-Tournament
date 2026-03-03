'''
pure numpy implementation

https://arxiv.org/pdf/1301.3781
https://arxiv.org/pdf/1310.4546


Algorithm Steps:

1. Load the dataset and preprocess it
    - Change every word to a unique intiger representation
    - Create Huffmann tree based on the frequency of words in the dataset

2. Create the training data - method to be decided
    Paper described "Subsampling of frequent words"

3. Define the model:
It is important to state that there are two main architectures:
    - Continuous Bag of Words (CBOW): predicts the target word based on multiple context words
        word1 word2 -> target word <- word3 word4
        Faster, less accurate, better for smaller datasets (?)
        
    - Skip-gram: predicts the context words based on one target word
        Slower, more accurate with syntatic tasks

    CBOW:
    - Input layer: multiple one-hot encoded vectors of the context words 
        (one hot size equal to the vocabulary size)
    - Projection layer: linear layer that averages the input vectors to create a single vector representation
    - Output later: a log-linear classifier predicting the middle word, using Hierarchical Softmax (according to the paper)

Also, the most important layer is actually a matrix of [embedding_dim X vocab_size]
and then, when we input those 4 words as an index to the matrix, then we average them (projection)
this singualr word combination is used to predict the middle word by using softmax


4. Define:
    - Backpropagation
    - Forward pass
    - Optimizer (sgadam? or sgd, original paper used sgd)
    - Loss function (cross-entropy loss?)
    - Linearly decaying learning rate (as described in the paper)

5. Train the model for a specified number of epochs, updating the weights using the optimizer and backpropagation

'''