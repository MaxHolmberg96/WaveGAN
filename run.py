from wavegan import *

hyperparams = {
    'num_channels': 1,
    'batch_size': 64,
    'model_dim': 64,
    'phase_shuffle': 2,
    'wgan_gp_lambda': 10,
    'd_per_g_update': 5,
    'adam_alpha': 1e-4,
    'adam_beta1': 0.5,
    'adam_beta2': 0.9,
}

generator = wavegan_generator(64, 1)
discriminator = wavegan_discriminator(64, 1)

def wgan_loss(generator, discriminator, z, data, batch_size=64):
    G_z = generator.predict(z)
    D_G_z = discriminator.predict(G_z)
    D_x = discriminator.predict(data)
    G_loss = -tf.math.reduce_mean(D_G_z)
    D_loss = tf.math.reduce_mean(D_G_z) - tf.math.reduce_mean(D_x)

    alpha = tf.math.random_uniform(shape=[batch_size, 1, 1], minval=0., maxval=1.)
    differences = G_z - data
    interpolates = data + (alpha * differences)
    # with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
    #   D_interp = WaveGANDiscriminator(interpolates, **args.wavegan_d_kwargs)

    LAMBDA = 10
    gradients = tf.gradients(D_interp, [interpolates])[0]
    slopes = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.math.reduce_mean((slopes - 1.) ** 2.)
    D_loss += LAMBDA * gradient_penalty

