'''
Autoencoder Script
'''

import tensorflow as tf


#rescaling inputs to 64*64
n_inputs = 64*64
n_hidden_1 = n_inputs/2
n_hidden_2 = n_hidden_1/2


#parameters
learning_rate = 0.01



X = tf.placeholder(tf.float32, shape=[None, n_inputs])

initialize = tf.variance_scaling_initializer()

weights={
    'h1_encoder':tf.Variable(initialize([n_inputs, n_hidden_1]), dtype=tf.float32),
    'h2_encoder': tf.Variable(initialize([n_hidden_1, n_hidden_2]), dtype=tf.float32),
    'h1_decoder': tf.Variable(initialize([n_hidden_2, n_hidden_1]), dtype=tf.float32),
    'h2_decoder': tf.Variable(initialize([n_hidden_1, n_inputs]), dtype=tf.float32)
}

biases = {
    'h1_bias': tf.Variable(tf.zeros(n_hidden_1), dtype=tf.float32),
    'h2_bias': tf.Variable(tf.zeros(n_hidden_2), dtype=tf.float32),
    'h1_bias': tf.Variable(tf.zeros(n_hidden_2), dtype=tf.float32),
    'h2_bias': tf.Variable(tf.zeros(n_hidden_1), dtype=tf.float32)
}



def encoder(x):

    h1_layer = tf.sigmoid(tf.add(tf.matmul(weights['h1_encoder'], X), biases['h1_bias']))
    h2_layer = tf.sigmoid(tf.add(tf.matmul(weights['h2_encoder'], h1_layer), biases['h2_bias']))

    return h2_layer


def decoder(x):

    h1_layer = tf.sigmoid(tf.add(tf.matmul(weights['h1_decoder'], x), biases['h1_bias']))
    h2_layer = tf.sigmoid(tf.add(tf.matmul(weights['h2_decoder'], h1_layer), biases['h2_bias']))

    return h2_layer

encoder_module = encoder(X)
decoder_module = decoder(encoder_module)
y_pred = decoder_module
y_true = X

loss = tf.reduce_mean(tf.square(y_true - y_pred))
optimizer = tf.train.AdamOptimizer().minimize(loss)
init = tf.global_variables_initializer()


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)


