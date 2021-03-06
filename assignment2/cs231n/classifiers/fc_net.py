from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        W1 = np.random.normal(size=(input_dim, hidden_dim), scale=weight_scale)
        b1 = np.zeros(hidden_dim)
        W2 = np.random.normal(size=(hidden_dim, num_classes), scale=weight_scale)
        b2 = np.zeros(num_classes)

        self.params["W1"] = W1
        self.params["W2"] = W2
        self.params["b1"] = b1
        self.params["b2"] = b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        a1, cache1 = affine_relu_forward(X, W1, b1)
        a2, cache2 = affine_forward(a1, W2, b2)
        scores = a2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        reg = self.reg
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += (0.5 * reg * np.sum(W2*W2)) + (0.5 * reg * np.sum(W1*W1))
        da2, dW2, db2 = affine_backward(dscores, cache2)
        da1, dW1, db1 = affine_relu_backward(da2, cache1)
        dW1 += reg * W1
        dW2 += reg * W2
        grads["W1"] = dW1
        grads["W2"] = dW2
        grads["b1"] = db1
        grads["b2"] = db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.cache = {}
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        hidden_dims.append(num_classes) # add last layer to the list
        previous_layer_size = input_dim
        last_layer = len(hidden_dims) - 1
        for index, hidden_layer_size in enumerate(hidden_dims):
            curr_layer_size = hidden_layer_size
            # initialize all parameters
            curr_layer_weight_str = "W" + str(index + 1)
            curr_layer_bias_str = "b" + str(index + 1)
            curr_layer_gamma_str = "gamma" + str(index + 1)
            curr_layer_beta_str = "beta" + str(index + 1)

            curr_layer_bias = np.zeros(curr_layer_size)
            curr_layer_weights = np.random.normal(scale=weight_scale,
                                                  size=(previous_layer_size,
                                                        curr_layer_size))
            curr_layer_gamma = np.ones(curr_layer_size)
            curr_layer_beta = np.zeros(curr_layer_size)

            # save all parameters in dict
            self.params[curr_layer_weight_str] = curr_layer_weights
            self.params[curr_layer_bias_str] = curr_layer_bias
            if self.use_batchnorm and index < last_layer:
                self.params[curr_layer_gamma_str] = curr_layer_gamma
                self.params[curr_layer_beta_str] = curr_layer_beta

            # move ahead one index
            previous_layer_size = curr_layer_size
        # print(self.params.keys())
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def get_layer_params(self, index):
        W = self.params["W" + str(index)]
        b = self.params["b" + str(index)]
        gamma_str = "gamma{}".format(index)
        beta_str = "beta{}".format(index)
        gamma = self.params[gamma_str] if gamma_str in self.params else None
        beta = self.params[beta_str] if beta_str in self.params else None
        return W, b, gamma, beta


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        previous_a = X
        last_layer = self.num_layers
        for layers in range(1, last_layer + 1):
            # print(layers)
            W, b, gamma, beta = self.get_layer_params(layers)
            layer_cache = {}
            if layers == last_layer:
                result, cache = affine_forward(previous_a, W, b)
            else:
                result, cache = affine_relu_forward(previous_a, W, b)
                if self.use_batchnorm:
                    result, batchnorm_cache = batchnorm_forward(result, gamma,
                                                beta, self.bn_params[layers - 1])
                    layer_cache["batchnorm"] = batchnorm_cache
                if self.use_dropout:
                    result, cache_dropout = dropout_forward(result, self.dropout_param)
                    layer_cache["dropout"] = cache_dropout
            layer_cache["fc"] = cache
            self.cache[layers] = layer_cache
            previous_a = result
        scores = result
        # print("Scores shape: {}".format(scores.shape))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads, reg = 0.0, {}, self.reg
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        # print(loss)
        num_train = X.shape[0]
        upper_layer_gradient = dscores
        for layers in reversed(range(1, last_layer + 1)):
            layer_cache = self.cache[layers]
            if layers == last_layer:
                da, dw, db = affine_backward(upper_layer_gradient,
                                             layer_cache["fc"])
            else:
                if self.use_dropout:
                    upper_layer_gradient = dropout_backward(upper_layer_gradient,
                                                        layer_cache["dropout"])
                if self.use_batchnorm:
                    upper_layer_gradient, dgamma, dbeta = \
                                        batchnorm_backward(upper_layer_gradient,
                                                       layer_cache["batchnorm"])
                    grads["gamma" + str(layers)] = dgamma
                    grads["beta" + str(layers)] = dbeta
                da, dw, db = affine_relu_backward(upper_layer_gradient,
                                                  layer_cache["fc"])
            dw += self.reg * self.params["W" + str(layers)]
            grads["W" + str(layers)] = dw
            grads["b" + str(layers)] = db
            upper_layer_gradient = da
            loss += 0.5 * self.reg * np.sum(self.params["W" + str(layers)]**2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
