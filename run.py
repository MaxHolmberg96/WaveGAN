from wavegan import *
import numpy as np
import datetime
import os
import argparse

#tf.config.experimental.set_visible_devices([], 'GPU')
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
    'update_losses': 10,
    'weights_folder': 'weights_folder_piano/',
    'sample_rate': 16000,
    'generated_audio_output_dir': "generated_audio"
}


@tf.function
def train_step_disc(x):
    z = tf.random.uniform(
        shape=[hyperparams['batch_size'], hyperparams['latent_dim']],
        minval=-1.,
        maxval=1.,
        dtype=tf.float32
    )

    with tf.GradientTape() as disc_tape:
        _, disc_loss, _ = wavegan_loss(generator, discriminator, x, z)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gradients_of_discriminator


@tf.function
def train_step_gen(x):
    z = tf.random.uniform(
        shape=[hyperparams['batch_size'], hyperparams['latent_dim']],
        minval=-1.,
        maxval=1.,
        dtype=tf.float32
    )

    with tf.GradientTape() as gen_tape:
        gen_loss, _, _ = wavegan_loss(generator, discriminator, x, z)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return gradients_of_generator

@tf.function
def wavegan_loss(gen, disc, x, z):
    G_z = gen(z)
    D_x = disc(x)
    D_G_z = disc(G_z)
    gen_loss_one = -tf.reduce_mean(D_G_z) #Expected value
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
    return gen_loss, disc_loss + gp, gen_loss_one


def train(dataset, epochs, shuffle=True, initial_log_step=0):
    import time
    from tqdm import tqdm
    update_step = 0
    update_log_step = initial_log_step
    for epoch in range(epochs):
        if shuffle:
            np.random.shuffle(dataset)
        start = time.time()
        offset = 0
        iterator = tqdm(range(dataset.shape[0] // hyperparams['batch_size']))
        for i in iterator:
            batch = dataset[offset:offset + hyperparams['batch_size']]
            if i % hyperparams['d_per_g_update'] == 0:
                grad_gen = train_step_gen(batch)
            grad_disc = train_step_disc(batch)
            if update_step % hyperparams['update_losses'] == 0:
                z = tf.random.uniform(
                    shape=[hyperparams['batch_size'], hyperparams['latent_dim']],
                    minval=-1.,
                    maxval=1.,
                    dtype=tf.float32
                )
                gen_loss, disc_loss, gen_loss_one = wavegan_loss(generator, discriminator, batch, z)
                iterator.set_description("\nGen loss: {}, Disc loss: {}".format(gen_loss, disc_loss))

                # Write to tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('gen_loss', gen_loss, step=update_log_step)
                    tf.summary.scalar('disc_loss', disc_loss, step=update_log_step)
                    tf.summary.scalar('gen_loss_one_term', gen_loss_one, step=update_log_step)
                update_log_step += 1

            offset += hyperparams['batch_size']
            update_step += 1

        save_model(generator, discriminator, generator_optimizer, discriminator_optimizer, hyperparams)

        write_summaries(grad_gen, grad_disc, hyperparams['generated_audio_output_dir'], epoch)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


def write_summaries(grad_gen, grad_disc, output_dir, epoch):
    z = tf.random.uniform(
        shape=[hyperparams['batch_size'], hyperparams['latent_dim']],
        minval=-1.,
        maxval=1.,
        dtype=tf.float32
    )
    generated_audio = generator(z)
    sample_dir = os.path.join(output_dir, str(epoch))
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    audi = []

    # Generate audio
    for i in range(generated_audio.shape[0]):
        audi.append(tf.expand_dims(generated_audio[i], 0))
        output_path = os.path.join(sample_dir, "{}_{}.wav".format(epoch, i + 1))
        string = tf.audio.encode_wav(generated_audio[i], hyperparams['sample_rate'])
        tf.io.write_file(output_path, string)
    audi = tf.concat(audi, axis=0)

    with train_summary_writer.as_default():
        tf.summary.audio("audio_samples", audi, hyperparams['sample_rate'], step=epoch, encoding="wav")
        for i in range(len(grad_gen)):
            tf.summary.histogram("gen_grads_layer_" + str(i), grad_gen[i], step=epoch)
            tf.summary.scalar("gen_grad_norm_layer_" + str(i), tf.norm(grad_gen[i], ord=2), step=epoch)
        for i in range(len(grad_disc)):
            tf.summary.histogram("disc_grads_layer_" + str(i), grad_disc[i], step=epoch)
            tf.summary.scalar("disc_grad_norm_layer_" + str(i), tf.norm(grad_disc[i], ord=2), step=epoch)
        generator_weights = generator.get_weights()
        for i in range(len(generator_weights)):
            tf.summary.histogram("gen_weights_layer_" + str(i), generator_weights[i], step=epoch)
        discriminator_weights = discriminator.get_weights()
        for i in range(len(discriminator_weights)):
            tf.summary.histogram("disc_weights_layer_" + str(i), discriminator_weights[i], step=epoch)


def generate_n_samples(n=50000):
    assert n % 10 == 0
    c = 0
    from tqdm import tqdm
    for i in tqdm(range(n // 10)):
        z = tf.random.uniform(shape=[10, hyperparams['latent_dim']], minval=-1., maxval=1., dtype=tf.float32)
        generated_audio = generator(z)
        for j in range(10):
            string = tf.audio.encode_wav(generated_audio[j], hyperparams['sample_rate'])
            tf.io.write_file(os.path.join(hyperparams['generated_audio_output_dir'], "samples", "{}.wav".format(c)), string)
            c += 1


ap = argparse.ArgumentParser()
ap.add_argument("-generate", "--generate", required=False, help="If we want to generate n audio samples", action='store_true')
ap.add_argument("-train", "--train", required=False, help="If we want to train", action='store_true')
ap.add_argument("-continue", "--continue", required=False, help="If we want to load the old weights", action='store_true')
ap.add_argument("-epochs", "--epochs", required=True, type=int, help="The number of epochs to train for")
ap.add_argument("-dataset", "--dataset", required=True, help="The path to the dataset file (.npy)")
ap.add_argument("-initial_log_step", "--initial_log_step", required=False, type=int, help="The step at where we should start logging")
args = vars(ap.parse_args())

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

generator = wavegan_generator(hyperparams['model_dim'], hyperparams['num_channels'])
discriminator = wavegan_discriminator(hyperparams['model_dim'], hyperparams['num_channels'])

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


if args['train']:
    initial_log_step = 0
    if args['continue']:
        load_model(generator, discriminator, generator_optimizer, discriminator_optimizer, hyperparams)
    if args['initial_log_step'] is not None:
        initial_log_step = args['initial_log_step']
    x = np.load(args['dataset'])
    train(x, args['epochs'], initial_log_step=initial_log_step)

elif args['generate']:
    load_model(generator, discriminator, generator_optimizer, discriminator_optimizer, hyperparams)
    generate_n_samples(n=1000)
