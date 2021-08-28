import tensorflow as tf


# model = tf.keras.models.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(64, 64, 3), name='Block0-Input'),
#     tf.keras.layers.Reshape((64*64*3,), name='Block0-Reshape'),
#     tf.keras.layers.Dense(500, activation='relu', name='Block1-Dense'),
#     tf.keras.layers.Dense(500, activation='relu', name='Block2-Dense'),
#     tf.keras.layers.Dense(500, activation='relu', name='Block3-Dense'),
#     tf.keras.layers.Dense(500, activation='relu', name='Block4-Dense'),
#     tf.keras.layers.Dense(500, activation='relu', name='Block5-Dense'),
#     tf.keras.layers.Dense(1, activation='sigmoid', name='Block6-Output'),
# ])

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(64, 64, 3), name='Block0-Input'),
    # Pad, Convolve, Batch Normalize, ReLU
    tf.keras.layers.ZeroPadding2D((3, 3), name='Block1-Padding'),
    tf.keras.layers.Conv2D(32, (3, 3), strides = (1, 1), name='Block1-Convolution'),
    tf.keras.layers.BatchNormalization(axis = 3, name = 'Block1-Normalization'),
    tf.keras.layers.Activation('relu', name = 'Block1-Activation'),
    # Convolve, Batch Normalize, ReLU, Pool
    tf.keras.layers.Conv2D(32, (3, 3), strides = (1, 1), name = 'Block2-Convolution'),
    tf.keras.layers.BatchNormalization(axis = 3, name = 'Block2-Normalization'),
    tf.keras.layers.Activation('relu', name = 'Block2-Activation'),
    tf.keras.layers.MaxPooling2D((2, 2), name= 'Block2-MaxPool'),
    # Flatten, Fully Connected, Sigmoid
    tf.keras.layers.Flatten(name = 'Block3-Flatten'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='Block3-Dense'),
])
