import tensorflow as tf


x = tf.Variable([5,10])
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/')

    writer.add_graph(sess.graph)