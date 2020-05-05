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


