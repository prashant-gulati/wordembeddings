import re
import numpy as np
import matplotlib.pyplot as plt

text = '''Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks.'''

np.random.seed(42)

# returns a list of tokens where each token is a word
# regex: * zero or more; + one or more; | or
# [A-Za-z] letters; \w = word characters i.e. letters/digits/underscore); ^ \'  carets or (escape) apostrophes
# .lower() for lowercase
def tokenize(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

tokens = tokenize(text)
# 84 tokens
print(len(tokens))
print(tokens)

# returns dicts
# set() deduplicates; unordered, no duplicates vs lists which are ordered and allow duplicates
def mapping(tokens):
    word_to_id = {}
    id_to_word = {}

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word

word_to_id, id_to_word = mapping(tokens)
# 60 unique tokens
print(len(word_to_id))

# chain multiple iterables together; yield each element one by one, flattening them in order
# generator: yields one value at a time instead of generating all at once; more memory efficient; but also not re-usable; 
# yield instead of return
def concat(*iterables):
    for iterable in iterables:
        yield from iterable

# [1,2,3,4]
print(list(concat([1,2],[3,4])))

# 1 in the position of id, 0s for all else
def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res

# training data label is a word; input data is the #window words before and #window words after that word
# return nparrays
def generate_training_data(tokens, word_to_id, window):
    X = []
    y = []
    n_tokens = len(tokens)

    for i in range(n_tokens):

        # construct indexes for the second for loop to move through
        idx = concat(
            range(max(0, i - window), i),
            range(i, min(n_tokens, i + window + 1))
        )
        for j in idx:
            if i == j:
                continue
            X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
            y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))

    return np.asarray(X), np.asarray(y)

X, y = generate_training_data(tokens, word_to_id, 2)
# columns = 60; len of the one hot encoding is the len of the vocabulary
# rows = 330; 84*4 - 1 - 2 - 1 - 2 for the instances where there aren't any words in the window sized vicinity of a word
print(X.shape)
print(y.shape)

# Naive softmax applied to rows of data with 60 columns
def softmax(X):
    res = []
    for x in X:
        # for one row, make each element e^original_value
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res

# categorical cross entropy; z is the prediction, y is the label
def cross_entropy(z, y):
    return - np.sum(np.log(z) * y)

# Might wonder how training a NN that predicts some nearby context word given an input token can be used to embed words into vectors
# b/c Output of the network is going to be some probability vector that passed through a softmax layer, not an embedding vector.

# Input is a sparse vector containing one-hot encoded vectors, the weight matrix effectively acts as a lookup table that moves one-hot encoded 
# vectors to dense vectors in a different dimension. This is what we want to achieve with embedding: representing words as dense vectors, a step-up from simple one-hot encoding
# The rows of the first weight matrix is the embedding i.e W1

# Each row in the weight matrix of that layer represents a single item (like a word or a user ID).
# The values in that row are the vector coordinates (the embedding) for that item.
# When you pass an index (e.g., word #402) into the network, the layer simply "grabs" the 402nd row of its weight matrix. 

# eg. (m, 60) matrix in the input layer => (60, 10) W1 matrix => Z1 (m, 10) matrix
# W1 is a (60,10) lookup table that converts (m, 60) into (m, 10)
# 60 nodes in the input layer each represent one word
# 60 rows in W1, each corresponding to a word i.e. eg. first row of W1 = first word of vocabulary

# dimensions: 60 dimensional one-hot-encoded vectors -> n_embedding eg. 10 dimensional embedding vectors -> 60 dimensions output
# returns a dictionary
def init_network(vocab_size, n_embedding):
    model = {
        "w1": np.random.randn(vocab_size, n_embedding),
        "w2": np.random.randn(n_embedding, vocab_size)
    }
    return model

# cache is a dictionary
# bias is omitted on purpose; otherwise all embeddings would be shifted by the same constant
# No RELU or other activation function - function approximation stays linear
def forward(model, X, return_cache=True):
    cache = {}

    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])

    if not return_cache:
        return cache["z"]
    return cache

model = init_network(len(word_to_id), 10)
print((X @ model["w1"]).shape)
print((X @ model["w1"] @ model["w2"]).shape)

# training: forward pass, backward pass: perform gradient descent and update parameters
def backward(model, X, y, alpha):
    cache  = forward(model, X)
    da2 = cache["z"] - y
    dw2 = cache["a1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1
    assert(dw2.shape == model["w2"].shape)
    assert(dw1.shape == model["w1"].shape)
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], y)

print("*"*50)

#inference before training
learning = one_hot_encode(word_to_id["learning"], len(word_to_id))
# [] converts python list of length (60) to to a list of lists. During numpy's matrix multiplication, its converted to a matrix of (1, 60)
# forward's output is the output of softmax which is a python list of 1 numpy array. [0] extracts the (60,) array
result = forward(model, [learning], return_cache=False)[0]

# np.argsort(result)[::-1] returns index corresponding to the highest probability word
# loop prints all words in descending order of probability
for word in (id_to_word[id] for id in np.argsort(result)[::-1]):
    print(word)

# training
n_iter = 50
learning_rate = 0.05

# python list of 50 loss values
history = [backward(model, X, y, learning_rate) for _ in range(n_iter)]

plt.style.use("seaborn-v0_8")
plt.plot(range(len(history)), history, color="skyblue")
plt.show()

print("*"*50)

#inference after training
learning = one_hot_encode(word_to_id["intelligence"], len(word_to_id))
result = forward(model, [learning], return_cache=False)[0]

for word in (id_to_word[id] for id in np.argsort(result)[::-1]):
    print(word)

# returns idx'th row of w1
def get_embedding(model, word):
    try:
        idx = word_to_id[word]
    except KeyError:
        print("`word` not in corpus")
    one_hot = one_hot_encode(idx, len(word_to_id))
    return forward(model, one_hot)["a1"]

print(get_embedding(model, "machine"))