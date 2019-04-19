#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Operation classes in computational graph.
'''
from simpleflow.base import DEFAULT_GRAPH
import numpy as np


class Operation(object):
    ''' Base class for all operations in simpleflow.

    An operation is a node in computational graph receiving zero or more nodes
    as input and produce zero or more nodes as output. Vertices could be an
    operation, variable or placeholder.
    '''
    def __init__(self, *input_nodes, name=None):
        ''' Operation constructor.

        :param input_nodes: Input nodes for the operation node.
        :type input_nodes: Objects of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        # Nodes received by this operation.
        self.input_nodes = input_nodes

        # Nodes that receive this operation node as input.
        self.output_nodes = []

        # Output value of this operation in session execution.
        self.output_value = None

        # Operation name.
        self.name = name

        # Graph the operation belongs to.
        self.graph = DEFAULT_GRAPH

        # Add this operation node to destination lists in its input nodes.
        for node in input_nodes:
            node.output_nodes.append(self)

        # Add this operation to default graph.
        self.graph.operations.append(self)

    def compute_output(self):
        ''' Compute and return the output value of the operation.
        '''
        raise NotImplementedError

    def compute_gradient(self, *compute_nodes, grad=None):
        ''' Compute and return the gradient of the operation wrt inputs.
        '''
        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)


# ------------------------------------------------------------------------------
# Addition operation
# ------------------------------------------------------------------------------


class Add(Operation):
    ''' An addition operation.
    '''
    def __init__(self, x, y, name=None):
        ''' Addition constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        ''' Compute and return the value of addition operation.
        '''
        x, y = self.input_nodes
        self.output_value = np.add(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, *compute_nodes, grad=None):
        ''' Compute the gradients for this operation wrt input values.

        :param grad: The gradient of other operation wrt the addition output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        node_values = [node.output_value for node in self.input_nodes if node in compute_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)

        for value in node_values:
            grad_wrt = grad
            while np.ndim(grad_wrt) > len(np.shape(value)):
                grad_wrt = np.sum(grad_wrt, axis=0)

            # 把上一步1维相加导致的矩阵的秩变化，再还原回来相同的秩序
            for axis, size in enumerate(np.shape(value)):
                if size == 1:
                    grad_wrt = np.sum(grad_wrt, axis=axis, keepdims=True)
            return grad_wrt
        return None


def add(x, y, name=None):
    ''' Returns x + y element-wise.
    '''
    return Add(x, y, name)

# ------------------------------------------------------------------------------
# Multiplication operation
# ------------------------------------------------------------------------------


class Multiply(Operation):
    ''' Multiplication operation.
    '''
    def __init__(self, x, y, name=None):
        ''' Multiplication constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        ''' Compute and return the multiplication operation result.
        '''
        x, y = self.input_nodes
        self.output_value = np.multiply(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, *compute_nodes, grad=None):
        ''' Compute and return gradients for this operation wrt input values.

        :param grad: The gradient of other operation wrt the mutiply output.
        :type grad: number or a ndarray.
        '''
        x, y = [node.output_value for node in self.input_nodes]
        a, b = self.input_nodes
        if grad is None:
            grad = np.ones_like(self.output_value)

        if a in compute_nodes:
            grad_wrt_x = grad * y  # 标量与矩阵相乘， 或者矩阵相乘，则导数为另一个数字
            while np.ndim(grad_wrt_x) > len(np.shape(x)):
                grad_wrt_x = np.sum(grad_wrt_x, axis=0)
            for axis, size in enumerate(np.shape(x)):
                if size == 1:
                    grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)
            return grad_wrt_x

        if b in compute_nodes:
            grad_wrt_y = grad * x
            while np.ndim(grad_wrt_y) > len(np.shape(y)):
                grad_wrt_y = np.sum(grad_wrt_y, axis=0)
            for axis, size in enumerate(np.shape(y)):
                if size == 1:
                    grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

            return grad_wrt_y
        return None


def multiply(x, y, name=None):
    ''' Returns x * y element-wise.
    '''
    return Multiply(x, y, name)

# ------------------------------------------------------------------------------
# Matrix multiplication operation
# ------------------------------------------------------------------------------


class MatMul(Operation):
    ''' Matrix multiplication operation.
    '''
    def __init__(self, x, y, name=None):
        ''' MatMul constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        ''' Compute and return the multiplication operation result.
        '''
        x, y = self.input_nodes
        self.output_value = np.dot(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, *compute_nodes, grad=None):
        ''' Compute and return the gradient for matrix multiplication.

        :param grad: The gradient of other operation wrt the matmul output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        # Get input values.
        x, y = [node.output_value for node in self.input_nodes]
        a, b = self.input_nodes
        # Default gradient wrt the matmul output.
        if grad is None:
            grad = np.ones_like(self.output_value)
        if a in compute_nodes:
            # Gradients wrt inputs.
            dfdx = np.dot(grad, np.transpose(y))
            return dfdx
        if b in compute_nodes:
            dfdy = np.dot(np.transpose(x), grad)
            return dfdy
        return None


def matmul(x, y, name=None):
    ''' Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
    '''
    return MatMul(x, y, name)


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

    def compute_gradient(self, *compute_nodes, grad=None):
        ''' Compute the gradient for negative operation wrt input value.

        :param grad: The gradient of other operation wrt the negative output.
        :type grad: ndarray.
        '''
        if grad is None:
            grad = np.ones_like(self.output_value)
        return -grad


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

    def compute_gradient(self, *compute_nodes, grad=None):
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

    def compute_gradient(self, *compute_nodes, grad=None):
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