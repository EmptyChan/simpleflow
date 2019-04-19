# -*- coding: utf-8 -*- 
"""
 Created with IntelliJ IDEA.
 Description:
 User: jinhuichen
 Date: 4/11/2019 6:18 PM 
 Description: 
"""
import numpy as np
from simpleflow.operations import Operation


# ------------------------------------------------------------------------------
# Sigmoid operation
# ------------------------------------------------------------------------------


class Sigmoid(Operation):
    ''' Sigmoid operation.
    '''

    def __init__(self, x, name=None):
        ''' Sigmoid operation constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        ''' Compute and return the value of sigmoid function.
        '''
        x, = self.input_nodes
        self.output_value = 1 / (1 + np.exp(-x.output_value))
        return self.output_value

    def compute_gradient(self, grad=None, *compute_nodes):
        ''' Compute the gradient for sigmoid operation wrt input value.

        :param grad: The gradient of other operation wrt the sigmoid output.
        :type grad: ndarray.
        '''
        if grad is None:
            grad = np.ones_like(self.output_value)
        return grad * self.output_value * (1 - self.output_value)


def sigmoid(x, name=None):
    ''' Computes sigmoid of `x` element-wise.
    '''
    return Sigmoid(x, name=name)


# ------------------------------------------------------------------------------
# Tanh operation
# ------------------------------------------------------------------------------


class Tanh(Operation):
    ''' Sigmoid operation.
    '''

    def __init__(self, x, name=None):
        ''' Sigmoid operation constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        ''' Compute and return the value of sigmoid function.
        '''
        x, = self.input_nodes
        self.output_value = np.tanh(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None, *compute_nodes):
        ''' Compute the gradient for sigmoid operation wrt input value.

        :param grad: The gradient of other operation wrt the sigmoid output.
        :type grad: ndarray.
        '''
        if grad is None:
            grad = np.ones_like(self.output_value)
        return grad * (1 - np.power(self.output_value, 2))


def tanh(x, name=None):
    ''' Computes sigmoid of `x` element-wise.
    '''
    return Tanh(x, name=name)


# ------------------------------------------------------------------------------
# ReLU operation
# ------------------------------------------------------------------------------


class ReLU(Operation):
    ''' Sigmoid operation.
    '''

    def __init__(self, x, name=None):
        ''' Sigmoid operation constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        ''' Compute and return the value of sigmoid function.
        '''
        x, = self.input_nodes
        self.output_value = np.clip(x.output_value, a_min=0, a_max=None)
        return self.output_value

    def compute_gradient(self, grad=None, *compute_nodes):
        ''' Compute the gradient for sigmoid operation wrt input value.

        :param grad: The gradient of other operation wrt the sigmoid output.
        :type grad: ndarray.
        '''
        if grad is None:
            grad = np.ones_like(self.output_value)
        x, = self.input_nodes
        return 0 if x.output_value <= 0 else grad


def relU(x, name=None):
    ''' Computes sigmoid of `x` element-wise.
    '''
    return ReLU(x, name=name)


# ------------------------------------------------------------------------------
# Softmax operation
# ------------------------------------------------------------------------------


class Softmax(Operation):
    ''' Sigmoid operation.
    '''

    def __init__(self, x, name=None):
        ''' Sigmoid operation constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        ''' Compute and return the value of sigmoid function.
        '''
        x, = self.input_nodes
        e_x = np.exp(x.output_value)
        self.output_value = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return self.output_value

    def compute_gradient(self, grad=None, *compute_nodes):
        ''' Compute the gradient for sigmoid operation wrt input value.

        :param grad: The gradient of other operation wrt the sigmoid output.
        :type grad: ndarray.
        '''
        if grad is None:
            grad = np.ones_like(self.output_value)
        return grad * self.output_value * (1 - self.output_value)


def softmax(x, name=None):
    ''' Computes sigmoid of `x` element-wise.
    '''
    return Softmax(x, name=name)


class Dropout(Operation):
    def __init__(self, x, prob, name=None):
        self.prob = prob
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        if self.prob < 0. or self.prob >= 1:  # level是概率值，必须在0~1之间
            raise Exception('Dropout level must be in interval [0, 1[.')
        retain_prob = 1. - self.prob
        # 我们通过binomial函数，生成与x一样的维数向量。binomial函数就像抛硬币一样，我们可以把每个神经元当做抛硬币一样
        # 硬币 正面的概率为p，n表示每个神经元试验的次数
        # 因为我们每个神经元只需要抛一次就可以了所以n=1，size参数是我们有多少个硬币。
        # 即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
        x, = self.input_nodes
        sample = np.random.binomial(n=1, p=retain_prob, size=x.shape)
        x *= sample  # 0、1与x相乘，我们就可以屏蔽某些神经元，让它们的值变为0

        # 平均网络
        self.output_value = x / retain_prob
        return self.output_value


def dropout(x, name=None):
    ''' Computes sigmoid of `x` element-wise.
    '''
    return Dropout(x, prob=0.2, name=name)