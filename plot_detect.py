# !pip install tensorflow-io

import os
import time
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import numpy as np
import csv
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"

# Grab yamnet model from tensorflow hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

class_map_path = yamnet_model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

wav_file = os.listdir(r'data/wav') # Change

png_file = os.listdir(r'data/detect')
png_file = [file.replace('png', 'wav') for file in png_file]
wav_file = set(wav_file) ^ set(png_file)
wav_file = sorted(list(wav_file))

for file in wav_file:

    start_time = time.time()

    output_wav = Path(r'data/wav') / Path(file) # Change

    if os.path.exists(output_wav):
        pass

    file_contents = tf.io.read_file(str(output_wav))
    wav_data, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=2)
    wav_data = wav_data[:, 0]
    wav_data = tfio.audio.resample(wav_data, rate_in=44100, rate_out=16000)

    scores, embeddings, spectrogram = yamnet_model(wav_data)
    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]

    # plot image
    plt.figure(figsize=(30, 7))

    plt.subplot(2, 1, 1)
    plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')

    mean_scores = np.mean(scores, axis=0)
    top_n = 10
    top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
    plt.subplot(2, 1, 2)
    plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')

    patch_padding = (0.025/ 2) / 0.01
    plt.xlim([-patch_padding - 0.5, scores.shape[0] + patch_padding - 0.5])
    yticks = range(0, top_n, 1)
    plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
    _ = plt.ylim(-0.5 + np.array([top_n, 0]))

    plt.savefig(f'data/detect/{file[0:10]}.png')
    plt.close()

    print(f'Transformed:{file[0:10]}\tSeconds:{time.time() - start_time}')
