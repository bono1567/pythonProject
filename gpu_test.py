import tensorflow as tf

tf.debugging.set_log_device_placement(True)

# This runs on GPu
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)


# This runs on CPU
with tf.device('/CPU:0'):
    c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    d = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Run on the GPU
    d = tf.matmul(a, b)
    print(d)
