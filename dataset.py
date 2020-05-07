import pathlib
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def load_data(dataset_folder, window_length=16384):
    data_dir = pathlib.Path(dataset_folder)
    _, ext = os.path.splitext(os.listdir(data_dir)[3])
    paths = list(data_dir.glob("*" + ext))
    x = [] # 18000, 16384, 1
    for path in tqdm(paths):
        raw_audio = tf.io.read_file(str(path))
        audio_sample = tf.audio.decode_wav(raw_audio, desired_samples=window_length)
        wav = audio_sample[0]# [1] contains sample freq
        print(np.max(wav), np.min(wav))
        x.append(tf.expand_dims(wav, 0))
    return tf.concat(x, axis=0)


x = load_data("dataset/sc09-spoken-numbers/sc09/train")

#np.save("sc09.npy", x)
