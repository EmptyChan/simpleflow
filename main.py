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
import matplotlib.pyplot as plt

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
noise = np.random.normal(0, 0.25, input_x.shape).astype(np.float32)
input_y = input_x * 3 + 0.5 + noise
# Placeholders for training data
x = sf.Placeholder(name="X值")
y_ = sf.Placeholder(name="Y值")

# Weigths
w = sf.Variable([[1.0]], name='weight')

# Threshold
b = sf.Variable(0.1, name='threshold')

# Predicted class by model
# y = x*w + b
y = sf.matmul(x, w, name="x.dot(w)") + b

loss = sf.MSELoss(y_, y)()

train_op = sf.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)

# plot the real data
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(np.reshape(input_x, (-1, 1)), np.reshape(input_y, (-1, 1)))
# plt.ion()
# plt.show()

feed_dict = {x: input_x, y_: input_y}
with sf.Session() as sess:
    for step in range(200):
        # 迭代训练
        sess.run(train_op, feed_dict)

        # 画图
        # if step % 10 == 0:
        #     try:
        #         ax.lines.remove(lines[0])
        #     except Exception:
        #         pass
        #     prediction_value = sess.run(y, feed_dict=feed_dict)
        #     # plot the prediction
        #     lines = ax.plot(input_x, prediction_value, 'r-', lw=5)
        #     plt.pause(1)
    w_value = sess.run(w, feed_dict=feed_dict)
    b_value = sess.run(b, feed_dict=feed_dict)
    print('w: {}, b: {}'.format(w_value, b_value))