# -*- coding: utf-8 -*- 
"""
 Created with IntelliJ IDEA.
 Description:
 User: jinhuichen
 Date: 4/15/2019 10:30 AM 
 Description: 
"""
import numpy as np
from queue import Queue
from simpleflow.graph import Graph

# Create a default graph.
DEFAULT_GRAPH = Graph()


# ------------------------------------------------------------------------------
# Function for gradients computation. 广度优先遍历
# ------------------------------------------------------------------------------


def compute_gradients(target_op):
    ''' Backpropagation implementation computing gradient of target operation wrt
        all the other connected nodes.

    :param target_op: The target operation whose gradient wrt other nodes would
                      be computed.
    :type target_op: Any operation type.

    :return grad_table: A table containing node objects and gradients.
    :type grad_table: dict.
    '''
    # A dict containing a mapping between node and gradient value of target_op wrt the node's output.
    # NOTE: It is the gradient wrt the node's OUTPUT NOT input.
    grad_table = {
        target_op: np.ones_like(target_op.output_value)
    }

    # The gradient wrt target_op itself is 1.

    # Perform a breadth-first search staring from the target_op in graph.
    # Queue for node traverasl.
    queue = Queue()
    queue.put(target_op)

    # Set for visited nodes.
    visited = set()
    visited.add(target_op)

    while not queue.empty():
        node = queue.get()

        # Compute gradient wrt the node's output.
        if node != target_op:
            grads_wrt_node_output = filter(lambda item: item is not None, map(
                lambda output_node: output_node.compute_gradient(node, grad=grad_table[output_node]),
                node.output_nodes
            ))
            # 相加所有的偏导
            grad_table[node] = sum(grads_wrt_node_output)

        # 放入邻接节点到队列
        if hasattr(node, 'input_nodes'):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table
