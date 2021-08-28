import tensorflow as tf


model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(64, 64, 3), name='Block0-Input'),
    tf.keras.layers.Reshape((64*64*3,), name='Block0-Reshape'),
    tf.keras.layers.Dense(500, activation='relu', name='Block1-Dense'),
    tf.keras.layers.Dense(500, activation='relu', name='Block2-Dense'),
    tf.keras.layers.Dense(10, activation='softmax', name='Block3-Output'),
])
