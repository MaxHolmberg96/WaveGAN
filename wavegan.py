import tensorflow as tf


def add_Conv1DTranpose(model, filters, kernel_size, strides):
    """
    Conv1D tranpose adapted from
        https://stackoverflow.com/a/45788699/13185722
        and
        https://github.com/chrisdonahue/wavegan/blob/v1/wavegan.py#L13
    """
    model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2)))
    model.add(tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
    ))
    model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, axis=2)))

def wavegan_generator(d, c):
    strides = 4
    k_size = 25
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(100))
    model.add(tf.keras.layers.Dense(units=256 * d))
    model.add(tf.keras.layers.Reshape((16, 16 * d)))
    model.add(tf.keras.layers.ReLU())
    add_Conv1DTranpose(model, filters=8 * d, kernel_size=(k_size, 1), strides=(strides, 1))
    model.add(tf.keras.layers.ReLU())
    add_Conv1DTranpose(model, filters=4 * d, kernel_size=(k_size, 1), strides=(strides, 1))
    model.add(tf.keras.layers.ReLU())
    add_Conv1DTranpose(model, filters=2 * d, kernel_size=(k_size, 1), strides=(strides, 1))
    model.add(tf.keras.layers.ReLU())
    add_Conv1DTranpose(model, filters=d, kernel_size=(k_size, 1), strides=(strides, 1))
    model.add(tf.keras.layers.ReLU())
    add_Conv1DTranpose(model, filters=c, kernel_size=(k_size, 1), strides=(strides, 1))
    model.add(tf.keras.layers.Activation(tf.nn.tanh, name="tanh"))
    model.build()
    model.summary()
    return model


def add_phase_shuffle(model, n):
    model.add(tf.keras.layers.Lambda(lambda x: x))

def wavegan_discriminator(d, c):
    n = 2
    alpha = 0.2
    stride = 4
    k_size = 25

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((16384, c))) # Fattar inte varför man inte behöver None här...
    model.add(tf.keras.layers.Conv1D(
        filters=d,
        kernel_size=k_size,
        strides=stride,
        padding="same"
    ))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
    add_phase_shuffle(model, n)
    model.add(tf.keras.layers.Conv1D(
        filters=2 * d,
        kernel_size=k_size,
        strides=stride,
        padding="same"
    ))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
    add_phase_shuffle(model, n)
    model.add(tf.keras.layers.Conv1D(
        filters=4 * d,
        kernel_size=k_size,
        strides=stride,
        padding="same"
    ))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
    add_phase_shuffle(model, n)
    model.add(tf.keras.layers.Conv1D(
        filters=8 * d,
        kernel_size=k_size,
        strides=stride,
        padding="same"
    ))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
    add_phase_shuffle(model, n)
    model.add(tf.keras.layers.Conv1D(
        filters=16 * d,
        kernel_size=k_size,
        strides=stride,
        padding="same"
    ))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1))

    model.build()
    model.summary()
    return model