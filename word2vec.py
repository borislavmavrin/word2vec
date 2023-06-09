"""word2vec implementation without autodiff, using NumPy only."""
import numpy as np
import random

from utils.utils import softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """
    s = 1.0 / (1.0 + np.exp(-x))

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """Naive Softmax loss & gradient function for word2vec models.

    Implement the naive softmax loss and gradients between a center word's
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length)
                    for all words in vocab (tranpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length)
                    (dJ / dU)
    """
    dot = outsideVectors.dot(centerWordVec)
    exp = np.exp(dot)
    sum = np.sum(exp)
    loss = -np.log(softmax(dot)[outsideWordIdx])
    gradCenterVec = np.sum(np.multiply(exp.reshape(-1, 1), outsideVectors), 0)
    gradCenterVec /= sum
    gradCenterVec -= outsideVectors[outsideWordIdx]
    gradOutsideVecs = np.multiply(exp.reshape(-1, 1), centerWordVec) / sum
    gradOutsideVecs[outsideWordIdx] -= centerWordVec

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """Sample K indexes which are not the outsideWordIdx."""
    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """Negative sampling loss function for word2vec models.

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """
    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    sigma = sigmoid(outsideVectors[indices[0]].dot(centerWordVec))
    sigma_s = sigmoid(-outsideVectors[indices[1:]].dot(centerWordVec))
    loss = -np.log(sigma) - np.sum(np.log(sigma_s))
    gradCenterVec = (sigma - 1.0) * outsideVectors[indices[0]]
    gradCenterVec += np.sum(
        np.multiply((1.0 - sigma_s).reshape(-1, 1), outsideVectors[indices[1:]]), 0
    )
    gradOutsideVecs = np.zeros_like(outsideVectors)
    gradOutsideVecs[outsideWordIdx] = (sigma - 1.0) * centerWordVec
    d = np.multiply((1 - sigma_s).reshape(-1, 1), centerWordVec.reshape(1, -1))
    for i in range(len(d)):
        gradOutsideVecs[negSampleWordIndices[i]] += d[i]

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """Skip-gram model in word2vec.

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape
                        (num words in vocab, word vector length)
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape
                        (num words in vocab, word vector length)
                        for all words in vocab (transpose of U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vector
                     in shape (num words in vocab, word vector length)
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length)
                    (dJ / dU)
    """
    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    centerWordVec = centerWordVectors[word2Ind[currentCenterWord]]
    for outsideWord in outsideWords:
        outsideWordIdx = word2Ind[outsideWord]
        (
            loss_word,
            gradCenterVecs_word,
            gradOutsideVectors_word,
        ) = word2vecLossAndGradient(
            centerWordVec, outsideWordIdx, outsideVectors, dataset
        )
        loss += loss_word
        gradCenterVecs[word2Ind[currentCenterWord]] += gradCenterVecs_word
        gradOutsideVectors += gradOutsideVectors_word

    return loss, gradCenterVecs, gradOutsideVectors


def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """SGD for word2vec."""
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N / 2), :]
    outsideVectors = wordVectors[int(N / 2):, :]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N / 2), :] += gin / batchsize
        grad[int(N / 2):, :] += gout / batchsize

    return loss, grad
