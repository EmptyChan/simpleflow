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

loss = sf.reduce_mean(
    sf.reduce_sum(
        sf.square(y - y_, name="平方"), axis=1
    )
)

train_op = sf.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)

feed_dict = {x: input_x, y_: input_y}
with sf.Session() as sess:
    for step in range(200):
        # 迭代训练
        sess.run(train_op, feed_dict)

    w_value = sess.run(w, feed_dict=feed_dict)
    b_value = sess.run(b, feed_dict=feed_dict)
    print('w: {}, b: {}'.format(w_value, b_value))