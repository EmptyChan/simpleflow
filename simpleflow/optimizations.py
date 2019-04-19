#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Optimizer classes for parameters optimization.
'''
import numpy as np
from simpleflow.base import compute_gradients, DEFAULT_GRAPH


class Optimizer(object):
    def __init__(self):
        self.output_value = None

    def minimize(self, loss):
        raise NotImplementedError

    def compute_output(self):
        ''' Compute and return the output value of the operation.
        '''
        raise NotImplementedError


class GradientDescentOptimizer(Optimizer):
    ''' Optimizer that implements the gradient descent algorithm.
    '''
    def __init__(self, learning_rate):
        ''' Construct a new gradient descent optimizer

        :param learning_rate: learning rate of optimizier.
        :type learning_rate: float
        '''
        super(self.__class__, self).__init__()
        self.learning_rate = learning_rate
        self.loss = None

    def compute_output(self):
        ''' Compute and return the output value of the operation.
        '''
        # Get gradient table.
        grad_table = compute_gradients(self.loss)

        # Iterate all trainable variables in graph.
        for var in DEFAULT_GRAPH.trainable_variables:
            if var in grad_table:
                grad = grad_table[var]
                # Update its output value.
                var.output_value -= self.learning_rate * grad
        return None

    def minimize(self, loss):
        ''' Generate an gradient descent optimization operation for loss.

        :param loss: The loss operation to be optimized.
        :type loss: Object of `Operation`
        '''
        self.loss = loss

        def call():
            yield self.loss
            yield self
        return call


class StochasticGradientDescentOptimizer(Optimizer):
    ''' Optimizer that implements the gradient descent algorithm.
    '''
    def __init__(self, learning_rate):
        ''' Construct a new gradient descent optimizer

        :param learning_rate: learning rate of optimizier.
        :type learning_rate: float
        '''
        super(self.__class__, self).__init__()
        self.learning_rate = learning_rate
        self.loss = None

    def compute_output(self):
        ''' Compute and return the output value of the operation.
        '''
        # 随机梯度下降，TODO
        # Get gradient table.
        raise NotImplementedError

    def minimize(self, loss):
        ''' Generate an gradient descent optimization operation for loss.

        :param loss: The loss operation to be optimized.
        :type loss: Object of `Operation`
        '''
        self.loss = loss

        def call():
            yield self.loss
            yield self
        return call


class MomentumOptimizer(Optimizer):
    ''' Optimizer that implements the gradient descent algorithm.
    '''
    def __init__(self, learning_rate, momentum=0., decay=0.):
        ''' Construct a new gradient descent optimizer

        :param learning_rate: learning rate of optimizier.
        :type learning_rate: float
        '''
        super(self.__class__, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.loss = None

    def compute_output(self):
        ''' Compute and return the output value of the operation.
        '''
        # 动量梯度下降
        # Get gradient table.
        grad_table = compute_gradients(self.loss)

        # Iterate all trainable variables in graph.
        for var in DEFAULT_GRAPH.trainable_variables:
            if var in grad_table:
                grad = grad_table[var]
                # Update its output value.
                self.decay = self.momentum * self.decay - self.learning_rate * grad
                var.output_value += self.decay
        return None

    def minimize(self, loss):
        ''' Generate an gradient descent optimization operation for loss.

        :param loss: The loss operation to be optimized.
        :type loss: Object of `Operation`
        '''
        self.loss = loss

        def call():
            yield self.loss
            yield self
        return call


class AdaGradOptimizer(Optimizer):
    ''' Optimizer that implements the gradient descent algorithm.
    '''
    def __init__(self, learning_rate, decay=0.):
        ''' Construct a new gradient descent optimizer

        :param learning_rate: learning rate of optimizier.
        :type learning_rate: float
        '''
        super(self.__class__, self).__init__()
        self.learning_rate = learning_rate
        self.decay = decay
        self.loss = None

    def compute_output(self):
        ''' Compute and return the output value of the operation.
        '''
        # AdaGrad梯度下降
        # Get gradient table.
        grad_table = compute_gradients(self.loss)

        # Iterate all trainable variables in graph.
        for var in DEFAULT_GRAPH.trainable_variables:
            if var in grad_table:
                grad = grad_table[var]
                # Update its output value.
                self.decay += np.power(grad, 2)
                var.output_value -= self.learning_rate * grad / (np.power(self.decay, -2) + 1e-8)
        return None

    def minimize(self, loss):
        ''' Generate an gradient descent optimization operation for loss.

        :param loss: The loss operation to be optimized.
        :type loss: Object of `Operation`
        '''
        self.loss = loss

        def call():
            yield self.loss
            yield self
        return call


class RMSPropOptimizer(Optimizer):
    ''' Optimizer that implements the gradient descent algorithm.
    '''
    def __init__(self, learning_rate, beta=0.999, decay=0.):
        ''' Construct a new gradient descent optimizer

        :param learning_rate: learning rate of optimizier.
        :type learning_rate: float
        '''
        super(self.__class__, self).__init__()
        self.learning_rate = learning_rate
        self.decay = decay
        self.beta = beta
        self.loss = None

    def compute_output(self):
        ''' Compute and return the output value of the operation.
        '''
        # RMSProp梯度下降
        # Get gradient table.
        grad_table = compute_gradients(self.loss)

        # Iterate all trainable variables in graph.
        for var in DEFAULT_GRAPH.trainable_variables:
            if var in grad_table:
                grad = grad_table[var]
                # Update its output value.
                self.decay = self.beta * self.decay + (1 - self.beta) * np.power(grad, 2)
                var.output_value -= self.learning_rate * grad / (np.power(self.decay, -2) + 1e-8)
        return None

    def minimize(self, loss):
        ''' Generate an gradient descent optimization operation for loss.

        :param loss: The loss operation to be optimized.
        :type loss: Object of `Operation`
        '''
        self.loss = loss

        def call():
            yield self.loss
            yield self
        return call


class AdamOptimizer(Optimizer):
    ''' Optimizer that implements the gradient descent algorithm.
    '''
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, momentum=0., decay=0.):
        ''' Construct a new gradient descent optimizer

        :param learning_rate: learning rate of optimizier.
        :type learning_rate: float
        '''
        super(self.__class__, self).__init__()
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.loss = None

    def compute_output(self):
        ''' Compute and return the output value of the operation.
        '''
        # RMSProp梯度下降
        # Get gradient table.
        grad_table = compute_gradients(self.loss)

        # Iterate all trainable variables in graph.
        for var in DEFAULT_GRAPH.trainable_variables:
            if var in grad_table:
                grad = grad_table[var]
                # Update its output value.
                self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * grad
                self.decay = self.beta2 * self.decay + (1 - self.beta2) * np.power(grad, 2)
                var.output_value -= self.learning_rate * self.momentum / (np.power(self.decay, -2) + 1e-8)
        return None

    def minimize(self, loss):
        ''' Generate an gradient descent optimization operation for loss.

        :param loss: The loss operation to be optimized.
        :type loss: Object of `Operation`
        '''
        self.loss = loss

        def call():
            yield self.loss
            yield self
        return call