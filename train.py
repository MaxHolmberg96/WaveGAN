from wavegan import *
import numpy as np
import datetime
import os
#tf.random.set_seed(0)
#tf.config.experimental.set_visible_devices([], 'GPU')
@tf.function
def train_step_disc(x):
    z = tf.random.uniform(
        shape=[hyperparams['batch_size'], hyperparams['latent_dim']],
        minval=-1.,
        maxval=1.,
        dtype=tf.float32
    )

    with tf.GradientTape() as disc_tape:
        _, disc_loss = wavegan_loss(generator, discriminator, x, z)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


@tf.function
def train_step_gen(x):
    z = tf.random.uniform(
        shape=[hyperparams['batch_size'], hyperparams['latent_dim']],
        minval=-1.,
        maxval=1.,
        dtype=tf.float32
    )

    with tf.GradientTape() as gen_tape:
        gen_loss, _ = wavegan_loss(generator, discriminator, x, z)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

@tf.function
def wavegan_loss(gen, disc, x, z):
    G_z = gen(z)
    D_x = disc(x)
    D_G_z = disc(G_z)
    #gen_loss = -tf.reduce_mean(D_G_z) #Expected value
    gen_loss = tf.reduce_mean(D_x) - tf.reduce_mean(D_G_z)
    disc_loss = -gen_loss

    # Gradient penalty
    epsilon = tf.random.uniform([hyperparams['batch_size'], 1, 1], minval=0., maxval=1.)
    x_hat = epsilon * x + (1 - epsilon) * G_z #batch x 16384 x 1
    with tf.GradientTape() as tape:
        tape.watch(x_hat)
        y = disc(x_hat) # 64 x 1
    dy_d_x_hat = tape.gradient(y, x_hat) #batch x 16384 x 1
    slopes = tf.sqrt(tf.compat.v1.reduce_sum(tf.square(dy_d_x_hat), reduction_indices=[1, 2]))
    gp = hyperparams['wgan_gp_lambda'] * tf.reduce_mean((slopes - 1.) ** 2)
    return gen_loss, disc_loss + gp


def train(dataset, epochs, shuffle=False, initial_log_step=0):
    import time
    from tqdm import tqdm
    update_step = 0
    update_log_step = initial_log_step
    for epoch in range(epochs):
        start = time.time()
        offset = 0
        iterator = tqdm(range(dataset.shape[0] // hyperparams['batch_size']))
        for i in iterator:
            batch = dataset[offset:offset + hyperparams['batch_size']]
            if i % hyperparams['d_per_g_update'] == 0:
                train_step_gen(batch)
            train_step_disc(batch)
            if update_step % hyperparams['update_losses'] == 0:
                z = tf.random.uniform(
                    shape=[hyperparams['batch_size'], hyperparams['latent_dim']],
                    minval=-1.,
                    maxval=1.,
                    dtype=tf.float32
                )
                gen_loss, disc_loss = wavegan_loss(generator, discriminator, batch, z)
                iterator.set_description("\nGen loss: {}, Disc loss: {}".format(gen_loss, disc_loss))

                # Write to tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('gen_loss', gen_loss, step=update_log_step)
                    tf.summary.scalar('disc_loss', disc_loss, step=update_log_step)
                update_log_step += 1

            offset += hyperparams['batch_size']
            update_step += 1

        if shuffle:
            dataset = tf.random.shuffle(dataset)
        generator.save_weights(hyperparams['weights_folder'] + "generator/")
        discriminator.save_weights(hyperparams['weights_folder'] + "discriminator/")

        z = tf.random.uniform(
            shape=[hyperparams['batch_size'], hyperparams['latent_dim']],
            minval=-1.,
            maxval=1.,
            dtype=tf.float32
        )
        generate_sample(generator(z), hyperparams['generated_audio_output_dir'], epoch)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


def generate_sample(generated_audio, output_dir, epoch):
    sample_dir = os.path.join(output_dir, str(epoch))
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    for i in range(generated_audio.shape[0]):
        output_path = os.path.join(sample_dir, "{}_{}.wav".format(epoch, i + 1))
        string = tf.audio.encode_wav(generated_audio[i], hyperparams['sample_rate'])
        tf.io.write_file(output_path, string)



hyperparams = {
    'dataset': 'sc09.npy',
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
    'update_losses': 10,
    'weights_folder': 'weights_folder/',
    'sample_rate': 16000,
    'generated_audio_output_dir': "generated_audio"
}

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


z = tf.random.uniform(
    shape=[hyperparams['batch_size'], hyperparams['latent_dim']],
    minval=-1.,
    maxval=1.,
    dtype=tf.float32
)
generator = wavegan_generator(hyperparams['model_dim'], hyperparams['num_channels'])
discriminator = wavegan_discriminator(hyperparams['model_dim'], hyperparams['num_channels'])
generator.summary()
discriminator.summary()
#generator.load_weights(hyperparams['weights_folder'] + "generator/")
#discriminator.load_weights(hyperparams['weights_folder'] + "discriminator/")
#generator.save_weights(hyperparams['weights_folder'] + "generator/")
#discriminator.save_weights(hyperparams['weights_folder'] + "discriminator/")

generate_sample(generator(z), "generated_audio", 0)


x = np.load(hyperparams['dataset'])

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

train(x, 100, initial_log_step=0)


