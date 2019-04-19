# -*- coding: utf-8 -*- 
"""
 Created with IntelliJ IDEA.
 Description:
 User: jinhuichen
 Date: 4/17/2019 1:24 PM 
 Description: 
"""
import numpy as np
from simpleflow.operations import Operation, Square, Log, Add, Negative
from simpleflow.variables import Constant

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

    def compute_gradient(self, *compute_nodes, grad=None):
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
# Reduce mean operation
# ------------------------------------------------------------------------------


class ReduceMean(Operation):
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
        self.output_value = np.mean(x.output_value, self.axis)
        return self.output_value

    def compute_gradient(self, *compute_nodes, grad=None):
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


def reduce_mean(x, axis=None):
    ''' Computes the sum of elements across dimensions of a tensor.
    '''
    return ReduceMean(x, axis=axis)


class Loss(object):
    def __init__(self, truth, prediction):
        self.truth = truth
        self.prediction = prediction

    def __call__(self, *args, **kwargs):
        ''' Compute and return the output value of the operation.
        '''
        raise NotImplementedError


# ------------------------------------------------------------------------------
# MSE operation
# ------------------------------------------------------------------------------


class MSELoss(Loss):
    def __call__(self, *args, **kwargs):
        ''' Compute and return the value of sigmoid function.
        '''
        return ReduceMean(ReduceSum(Square(self.prediction - self.truth), axis=1))


# ------------------------------------------------------------------------------
# CrossEntropy Loss operation
# ------------------------------------------------------------------------------


class CrossEntropyLoss(Loss):

    def __call__(self, *args, **kwargs):
        ''' Compute and return the value of sigmoid function.
        '''
        entroy1 = Negative(self.truth * Log(self.prediction))

        rest_truth = Constant(1) - self.truth
        rest_prediction = Constant(1) - self.prediction
        entroy2 = Negative(rest_truth * Log(rest_prediction))

        return ReduceMean(ReduceSum(Add(entroy1, entroy2)), axis=1)
