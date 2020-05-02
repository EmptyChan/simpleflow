#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Optimizer classes for parameters optimization.
'''
import numpy as np
from simpleflow.base import compute_gradients, DEFAULT_GRAPH
from functools import partial


class Optimizer(object):
    """
    优化器
    """
    def __init__(self):
        self.output_value = None
        self.loss = None
        self.grad_table = None

    def minimize(self, loss):
        """
        最小化损失度
        :param loss: 损失度
        :return:
        """
        self.loss = loss
        return [self.loss, self]

    def compute_output(self):
        """
        计算优化器的输出
        :return:
        """
        self.grad_table = compute_gradients(self.loss)
        for var in DEFAULT_GRAPH.trainable_variables:
            self.call(var)
        return None

    def call(self, var):
        raise NotImplementedError


class GradientDescentOptimizer(Optimizer):
    """
    梯度下降优化器
    """
    def __init__(self, learning_rate):
        ''' Construct a new gradient descent optimizer

        :param learning_rate: learning rate of optimizier.
        :type learning_rate: float
        '''
        super(self.__class__, self).__init__()
        self.learning_rate = learning_rate

    def call(self, var):
        ''' Compute and return the output value of the operation.
        '''
        # Get gradient table.
        if var in self.grad_table:
            # Update its output value.
            var.output_value -= self.learning_rate * self.grad_table[var]
            del self.grad_table[var]


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

    def call(self, var):
        ''' Compute and return the output value of the operation.
        '''
        # 随机梯度下降，TODO
        # Get gradient table.
        raise NotImplementedError


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

    def call(self, var):
        ''' Compute and return the output value of the operation.
        '''
        # 动量梯度下降
        # Iterate all trainable variables in graph.
        if var in self.grad_table:
            # Update its output value.
            self.decay = self.momentum * self.decay - self.learning_rate * self.grad_table[var]
            var.output_value += self.decay
            del self.grad_table[var]


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

    def call(self, var):
        ''' Compute and return the output value of the operation.
        '''
        # AdaGrad梯度下降
        if var in self.grad_table:
            # Update its output value.
            self.decay += np.power(self.grad_table[var], 2)
            var.output_value -= self.learning_rate * self.grad_table[var] / (np.power(self.decay, -2) + 1e-8)
            del self.grad_table[var]


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

    def call(self, var):
        ''' Compute and return the output value of the operation.
        '''
        # RMSProp梯度下降
        if var in self.grad_table:
            # Update its output value.
            self.decay = self.beta * self.decay + (1 - self.beta) * np.power(self.grad_table[var], 2)
            var.output_value -= self.learning_rate * self.grad_table[var] / (np.power(self.decay, -2) + 1e-8)
            del self.grad_table[var]


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

    def call(self, var):
        ''' Compute and return the output value of the operation.
        '''
        # Adam梯度下降
        if var in self.grad_table:
            # Update its output value.
            self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * self.grad_table[var]
            self.decay = self.beta2 * self.decay + (1 - self.beta2) * np.power(self.grad_table[var], 2)
            var.output_value -= self.learning_rate * self.momentum / (np.power(self.decay, -2) + 1e-8)
            del self.grad_table[var]
