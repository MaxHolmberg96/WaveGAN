import tensorflow as tf


def add_Conv1DTranpose(model, filters, kernel_size, strides, kernel_initializer):
    """
    Conv1D tranpose adapted from
        https://stackoverflow.com/a/45788699/13185722
        and
        https://github.com/chrisdonahue/wavegan/blob/v1/wavegan.py#L13
    """
    #model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2)))
    model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1)))
    model.add(tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer=kernel_initializer
    ))
    model.add(tf.keras.layers.Lambda(lambda x: x[:, 0]))
    #model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, axis=2)))

def wavegan_generator(d, c):
    strides = (1, 4)
    k_size = (1, 25)
    initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.02)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(100))
    model.add(tf.keras.layers.Dense(units=256 * d, kernel_initializer=initializer))
    model.add(tf.keras.layers.Reshape((16, 16 * d)))
    model.add(tf.keras.layers.ReLU())
    add_Conv1DTranpose(model, filters=8 * d, kernel_size=k_size, strides=strides, kernel_initializer=initializer)
    model.add(tf.keras.layers.ReLU())
    add_Conv1DTranpose(model, filters=4 * d, kernel_size=k_size, strides=strides, kernel_initializer=initializer)
    model.add(tf.keras.layers.ReLU())
    add_Conv1DTranpose(model, filters=2 * d, kernel_size=k_size, strides=strides, kernel_initializer=initializer)
    model.add(tf.keras.layers.ReLU())
    add_Conv1DTranpose(model, filters=d, kernel_size=k_size, strides=strides, kernel_initializer=initializer)
    model.add(tf.keras.layers.ReLU())
    add_Conv1DTranpose(model, filters=c, kernel_size=k_size, strides=strides, kernel_initializer=initializer)
    model.add(tf.keras.layers.Activation(tf.nn.tanh, name="tanh"))
    model.build()
    return model



def _phaseshuffle(x, rad=2, pad_type='reflect'):
    b, x_len, nch = x.get_shape().as_list()
    phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)
    x = x[:, phase_start:phase_start + x_len]
    x.set_shape([b, x_len, nch])
    return x


def wavegan_discriminator(d, c):
    n = 2
    alpha = 0.2
    stride = 4
    k_size = 25
    initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.02)


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((16384, c))) # Fattar inte varför man inte behöver None här...
    model.add(tf.keras.layers.Conv1D(
        filters=d,
        kernel_size=k_size,
        strides=stride,
        padding="same",
        kernel_initializer=initializer
    ))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
    model.add(tf.keras.layers.Lambda(lambda x: _phaseshuffle(x)))
    model.add(tf.keras.layers.Conv1D(
        filters=2 * d,
        kernel_size=k_size,
        strides=stride,
        padding="same",
        kernel_initializer=initializer
    ))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
    model.add(tf.keras.layers.Lambda(lambda x: _phaseshuffle(x)))
    model.add(tf.keras.layers.Conv1D(
        filters=4 * d,
        kernel_size=k_size,
        strides=stride,
        padding="same",
        kernel_initializer=initializer
    ))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
    model.add(tf.keras.layers.Lambda(lambda x: _phaseshuffle(x)))
    model.add(tf.keras.layers.Conv1D(
        filters=8 * d,
        kernel_size=k_size,
        strides=stride,
        padding="same",
        kernel_initializer=initializer
    ))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
    model.add(tf.keras.layers.Lambda(lambda x: _phaseshuffle(x)))
    model.add(tf.keras.layers.Conv1D(
        filters=16 * d,
        kernel_size=k_size,
        strides=stride,
        padding="same",
        kernel_initializer=initializer
    ))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1, kernel_initializer=initializer))

    model.build()
    return model