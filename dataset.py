import pathlib
import os
import tensorflow as tf
import librosa
from tqdm import tqdm
import argparse
import numpy as np


def load_data(dataset_folder, window_length=16384):
    data_dir = pathlib.Path(dataset_folder)
    _, ext = os.path.splitext(os.listdir(data_dir)[3])
    paths = list(data_dir.glob("*" + ext))
    x = [] # 18000, 16384, 1
    for path in tqdm(paths):
        raw_audio = tf.io.read_file(str(path))
        audio_sample = tf.audio.decode_wav(raw_audio, desired_samples=window_length)
        wav = audio_sample[0]# [1] contains sample freq
        x.append(tf.expand_dims(wav, 0))
    return tf.concat(x, axis=0)



def concat_piano(dataset_folder, fs=16000, window_length=16384):
    data_dir = pathlib.Path(dataset_folder)
    _, ext = os.path.splitext(os.listdir(data_dir)[3])
    paths = list(data_dir.glob("*" + ext))
    x = []  # 18000, 16384, 1
    for path in tqdm(paths):
        raw_audio = tf.io.read_file(str(path))
        audio_sample = librosa.load(str(path), sr=fs)
        audio_sample = audio_sample[0]
        if tf.keras.backend.max(tf.abs(audio_sample)) > 1:
            audio_sample /= tf.keras.backend.max(tf.abs(audio_sample))
        x.append(audio_sample)
    return tf.concat(x, axis=0)


def piano_generator(wav_file, batch_size, window_length=16384):
    raw_audio = tf.io.read_file(wav_file)
    audio_sample = tf.audio.decode_wav(raw_audio)[0]
    audio_length = audio_sample.shape[0]
    while True:
        starts = tf.random.uniform([batch_size], minval=0, maxval=audio_length - window_length + 1, dtype=tf.int32)
        x = []
        for i in starts:
            x.append(tf.expand_dims(audio_sample[i:i + window_length], 0))
        yield tf.concat(x, axis=0)


def create_piano_npy(wav_file, stride=16384//2, window_length=16384):
    raw_audio = tf.io.read_file(wav_file)
    audio_sample = tf.audio.decode_wav(raw_audio)[0]
    audio_length = audio_sample.shape[0]
    start = 0
    x = []
    for start in tqdm(range(0, audio_length - window_length, stride)):
        x.append(tf.expand_dims(audio_sample[start:start + window_length], axis=0))
    return tf.concat(x, axis=0)



ap = argparse.ArgumentParser()
ap.add_argument("-create_piano_wav", "--create_piano_wav", required=False, action="store_true", help="Generates a wav file with the concatenation of all piano files in train")
ap.add_argument("-create_piano_npy", "--create_piano_npy", required=False, action="store_true", help="Generates a npy with windows of piano sounds of window length 16384")
ap.add_argument("-create_sc09_npy", "--create_sc09_npy", required=False, action="store_true", help="Generates a npy file with all the 1 second spoken utterances from sc09")
ap.add_argument("-path", "--path", required=True, help="Path to the folder")
ap.add_argument("-output_path", "--output_path", required=True, help="Where to store the output.")
args = vars(ap.parse_args())

if args['create_piano_wav']:
    piano = concat_piano(args['path'])
    string = tf.audio.encode_wav(tf.expand_dims(piano, 1), 16000)
    tf.io.write_file(args['output_path'], string)
elif args['create_piano_npy']:
    np.save(args['output_path'], create_piano_npy(args["path"], 16384 // 16))
elif args['create_sc09_npy']:
    np.save(args['output_path'], load_data(args['output_path']))
else:
    print("Please give an argument: -create_piano_wav, -create_piano_npy or -create_sc09_npy")
