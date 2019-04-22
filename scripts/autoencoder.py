'''
Autoencoder Script
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("[+]Sizes: ", x_train.shape)

#rescaling inputs to 64*64
n_inputs = 784
n_hidden_1 = n_inputs//2
n_hidden_2 = n_hidden_1//2


#parameters
learning_rate = 0.001
training_steps = 100000
batch_size = 1


display_step = 1000

# print("[+]Initializing with hidden shapes:{} and {}".format(n_hidden_1, n_hidden_2))
X = tf.placeholder(tf.float32, shape=[None, n_inputs])

initialize = tf.variance_scaling_initializer()

weights={
    'h1_en': tf.Variable(tf.random_normal([n_inputs, n_hidden_1])),
    'h2_en': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h1_de': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'h2_de': tf.Variable(tf.random_normal([n_hidden_1, n_inputs]))
}

biases = {
    'h1_bias_en': tf.Variable(tf.zeros(n_hidden_1), dtype=tf.float32),
    'h2_bias_en': tf.Variable(tf.zeros(n_hidden_2), dtype=tf.float32),
    'h1_bias_de': tf.Variable(tf.zeros(n_hidden_1), dtype=tf.float32),
    'h2_bias_de': tf.Variable(tf.zeros(n_inputs), dtype=tf.float32)
}


def encoder(x):

    h1_layer = tf.sigmoid(tf.add(tf.matmul(x, weights['h1_en']), biases['h1_bias_en']))
    h2_layer = tf.sigmoid(tf.add(tf.matmul(h1_layer, weights['h2_en']), biases['h2_bias_en']))

    return h2_layer


def decoder(x):

    h1_layer = tf.sigmoid(tf.add(tf.matmul(x, weights['h1_de']), biases['h1_bias_de']))
    h2_layer = tf.sigmoid(tf.add(tf.matmul(h1_layer, weights['h2_de']), biases['h2_bias_de']))

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
        writer = tf.summary.FileWriter('logs/')
        writer.add_graph(sess.graph)
        # Training
        for i in range(0, len(x_train)//batch_size):
            # Prepare Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_x = x_train[i * batch_size:i * batch_size + batch_size]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x.reshape(-1, 784)})
            tf.summary.scalar('Loss', l)
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l))



        n = 4
        canvas_orig = np.empty((28 * n, 28 * n))
        canvas_recon = np.empty((28 * n, 28 * n))
        for i in range(n):
            # MNIST test set
            batch_x = x_test[i * batch_size:i * batch_size + batch_size]
            # Encode and decode the digit image
            g = sess.run(decoder_module, feed_dict={X: batch_x.reshape(-1, 784)})

            # Display original images
            for j in range(n):
                # Draw the original digits
                canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    batch_x[j].reshape([28, 28])
            # Display reconstructed images
            for j in range(n):
                # Draw the reconstructed digits
                canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    g[j].reshape([28, 28])

        print("Original Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        plt.show()

        print("Reconstructed Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        plt.show()



