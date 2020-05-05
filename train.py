from wavegan import *

hyperparams = {
    'num_channels': 1,
    'batch_size': 64,
    'model_dim': 64,
    'latent_dim': 100,
    'phase_shuffle': 2,
    'wgan_gp_lambda': 10,
    'd_per_g_update': 5,
    'adam_alpha': 1e-4,
    'adam_beta1': 0.5,
    'adam_beta2': 0.9,
    'LAMBDA': 10,
}

generator = wavegan_generator(64, 1)
discriminator = wavegan_discriminator(64, 1)

"""
Define optimizers
"""
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=hyperparams['adam_alpha'],
    beta_1=hyperparams['adam_beta1'],
    beta_2=hyperparams['adam_beta2']
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=hyperparams['adam_alpha'],
    beta_1=hyperparams['adam_beta1'],
    beta_2=hyperparams['adam_beta2']
)

@tf.function
def train_step(x):
    z = tf.random.uniform(
        shape=[hyperparams['batch_size'], hyperparams['latent_dim']],
        minval=-1.,
        maxval=1.,
        dtype=tf.float32
    )

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        G_z = generator.predict(z)
        D_x = discriminator.predict(x)
        D_G_z = discriminator.predict(G_z)
        gen_loss = -tf.reduce_mean(D_G_z)
        disc_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)
        alpha = tf.random.uniform(
            shape=[hyperparams['batch_size'], 1, 1],
            minval=0.,
            maxval=1.
        )
        differences = G_z - x
        interpolates = x + (alpha * differences)
        with tf.GradientTape() as interp_tape:
            D_interp = discriminator.predict(interpolates)
        interp_grads = interp_tape.gradient(D_interp, discriminator.trainable_variables)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(interp_grads), reduction_indices=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
        disc_loss += hyperparams['LAMBDA'] * gradient_penalty

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    import time
    from tqdm import tqdm
    for epoch in range(epochs):
        start = time.time()
        for batch in tqdm(dataset):
            train_step(batch)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
