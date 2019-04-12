# -*- coding: utf-8 -*- 
"""
 Created with IntelliJ IDEA.
 Description:
 User: jinhuichen
 Date: 2/13/2019 2:44 PM 
 Description: 
"""
import numpy as np
import simpleflow as sf

# # Create a graph
# with sf.Graph().as_default():
#     a = sf.constant(1.0, name='a')
#     b = sf.constant(2.0, name='b')
#     result = sf.add(a, b, name='a+b')
#     c = sf.constant(3.0, name="c")
#     result = sf.multiply(result, c, name="c * (a + b)")
#     # Create a session to run the graph
#     with sf.Session() as sess:
#         print(sess.run(result))


input_x = np.linspace(-1, 1, 100, dtype=np.float32)[:, np.newaxis]
input_y = input_x * 3 + np.random.randn(input_x.shape[0])*0.5
# Placeholders for training data
x = sf.Placeholder()
y_ = sf.Placeholder()

# Weigths
w = sf.Variable([[1.0]], name='weight')

# Threshold
b = sf.Variable(0.0, name='threshold')

# Predicted class by model
y = x*w + b

loss = sf.reduce_sum(sf.square(y - y_))

train_op = sf.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)

# feed_dict = {x: np.reshape(input_x, (-1, 1)), y_: np.reshape(input_y, (-1, 1))}
feed_dict = {x: input_x, y_: input_y}
with sf.Session() as sess:
    for step in range(20):
        loss_value = sess.run(loss, feed_dict=feed_dict)
        mse = loss_value / len(input_x)

        if step % 1 == 0:
            print('step: {}, loss: {}, mse: {}'.format(step, loss_value, mse))
        sess.run(train_op, feed_dict)
    w_value = sess.run(w, feed_dict=feed_dict)
    b_value = sess.run(b, feed_dict=feed_dict)
    print('w: {}, b: {}'.format(w_value, b_value))