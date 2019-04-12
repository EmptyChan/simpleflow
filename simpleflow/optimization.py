# -*- coding: utf-8 -*- 
"""
 Created with IntelliJ IDEA.
 Description:
 User: jinhuichen
 Date: 4/11/2019 6:18 PM 
 Description: 
"""
import numpy as np
from simpleflow import Operation
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

    def compute_gradient(self, grad=None):
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
# Logarithm operation
# ------------------------------------------------------------------------------


class Log(Operation):
    ''' Natural logarithm operation.
    '''
    def __init__(self, x, name=None):
        ''' Logarithm constructor.

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
        self.output_value = np.log(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        ''' Compute the gradient for natural logarithm operation wrt input value.

        :param grad: The gradient of other operation wrt the logarithm output.
        :type grad: ndarray.
        '''
        x = self.input_nodes[0].output_value
        if grad is None:
            grad = np.ones_like(self.output_value)
        return grad * 1 / x


def log(x, name=None):
    ''' Computes the natural logarithm of x element-wise.
    '''
    return Log(x, name=name)

# ------------------------------------------------------------------------------
# Negative operation
# ------------------------------------------------------------------------------


class Negative(Operation):
    ''' Negative operation.
    '''
    def __init__(self, x, name=None):
        ''' Operation constructor.

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
        self.output_value = -x.output_value
        return self.output_value

    def compute_gradient(self, grad=None):
        ''' Compute the gradient for negative operation wrt input value.

        :param grad: The gradient of other operation wrt the negative output.
        :type grad: ndarray.
        '''
        if grad is None:
            grad = np.ones_like(self.output_value)
        return -grad

# ------------------------------------------------------------------------------
# Reduce sum operation
# ------------------------------------------------------------------------------


class ReduceSum(Operation):
    ''' Reduce sum operation.
    '''
    def __init__(self, x, axis=None):
        ''' Operation constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param axis: The dimensions to reduce. If `None`, reduces all dimensions.
        :type axis: int.
        '''
        super(self.__class__, self).__init__(x)
        self.axis = axis

    def compute_output(self):
        ''' Compute and return the value of sigmoid function.
        '''
        x, = self.input_nodes
        self.output_value = np.sum(x.output_value, self.axis)
        return self.output_value

    def compute_gradient(self, grad=None):
        ''' Compute the gradient for negative operation wrt input value.

        :param grad: The gradient of other operation wrt the negative output.
        :type grad: ndarray.
        '''
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value)

        output_shape = np.array(np.shape(input_value))
        output_shape[self.axis] = 1.0
        tile_scaling = np.shape(input_value) // output_shape
        grad = np.reshape(grad, output_shape)
        return np.tile(grad, tile_scaling)


def reduce_sum(x, axis=None):
    ''' Computes the sum of elements across dimensions of a tensor.
    '''
    return ReduceSum(x, axis=axis)

# ------------------------------------------------------------------------------
# Square operation
# ------------------------------------------------------------------------------


class Square(Operation):
    ''' Square operation.
    '''
    def __init__(self, x, name=None):
        ''' Operation constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The name of the operation.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        ''' Compute and return the value of square function.
        '''
        x, = self.input_nodes
        self.output_value = np.square(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        ''' Compute the gradient for square operation wrt input value.

        :param grad: The gradient of other operation wrt the square output.
        :type grad: ndarray.
        '''
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value)

        return grad*np.multiply(2.0, input_value)


def square(x, name=None):
    ''' Computes square of x element-wise.
    '''
    return Square(x, name=name)
