import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# This file calculates the spectrogram and mfcc of a file (defined below)
# How to run:
# python wav_to_spectrogram.py


# Set to "original" to display image as used in the networks
# Set to "enhanced" to display a brightened enhanced version
mode = "original"


# input_wav = '/tmp/speech_dataset/happy/ab00c4b2_nohash_0.wav' # happy (loud, example from tensorflow tutorial)
# input_wav = '/tmp/speech_dataset/happy/0b09edd3_nohash_0.wav' # happy (soft)
# input_wav = '/tmp/speech_dataset/sheila/0b40aa8e_nohash_0.wav' # sheila
input_wav = '/tmp/speech_dataset/bird/0c2ca723_nohash_0.wav' # bird (loud)
# input_wav = '/tmp/speech_dataset/wow/2bdbe5f7_nohash_0.wav' # wow



# desired_samples = samples per second = y axis
# desired_samples / window_stride_samples = samples per second
# 

if mode == "original":
    model_settings = {'dct_coefficient_count': 40, 'window_size_samples': 480, 'label_count': 12, 'desired_samples': 16000, 'window_stride_samples': 160, 'spectrogram_length': 98, 'sample_rate': 16000, 'fingerprint_size': 3920}
else:
    # settings from wav_to_spectrogram script from tensorflow tutorial
    model_settings = {'dct_coefficient_count': 40, 'window_size_samples': 256, 'label_count': 12, 'desired_samples': 16000, 'window_stride_samples': 128, 'spectrogram_length': 98, 'sample_rate': 16000, 'fingerprint_size': 3920}


with tf.Session(graph=tf.Graph()) as sess:

    # load file
    desired_samples = model_settings['desired_samples']
    wav_filename_placeholder_ = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder_)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)


    # required placeholders for things we don't really use, here no values are set yet,
    # they are just placeholders
    foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
    time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
    time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
    background_data_placeholder_ = tf.placeholder(tf.float32, [desired_samples, 1])
    background_volume_placeholder_ = tf.placeholder(tf.float32, [])
    time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
    time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
    scaled_foreground = tf.multiply(wav_decoder.audio, foreground_volume_placeholder_)
    padded_foreground = tf.pad(scaled_foreground, time_shift_padding_placeholder_, mode='CONSTANT')
    sliced_foreground = tf.slice(padded_foreground, time_shift_offset_placeholder_, [desired_samples, -1])
    background_mul = tf.multiply(background_data_placeholder_, background_volume_placeholder_)
    background_add = tf.add(background_mul, sliced_foreground)
    background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)


    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    spectrogram = contrib_audio.audio_spectrogram(
        background_clamp,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True)
    mfcc = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        dct_coefficient_count=model_settings['dct_coefficient_count'])

    # set some paramters / settings for the spectrogram / mfcc
    input_dict = {
        wav_filename_placeholder_: input_wav, # path to file we want to analyze
        time_shift_padding_placeholder_: [[0, 0], [0, 0]],
        time_shift_offset_placeholder_: [0, 0],
        background_data_placeholder_ : np.zeros([desired_samples, 1]), # no background noise
        background_volume_placeholder_ : 0.0, # no background noise
        foreground_volume_placeholder_ : 1.0 # don't silence the wav file
    }


    # run spectrogram and mfcc analysis, output is a numpy array
    spectrogram_data = sess.run( spectrogram, feed_dict= input_dict)
    mfcc_data = sess.run( mfcc, feed_dict= input_dict)

    spectrogram_data_plot = spectrogram_data[0]
    mfcc_data_plot = mfcc_data[0]

    # Do some extra preprocessing to make the spectrogram more easy to read_file
    # if the enhanced mode was chosen
    if mode == "enhanced":
        # normalize the array to the 0-255 range
        spectrogram_data_plot *= 255.0 / spectrogram_data_plot.max()

        # brighten it a bit
        brightness = 3 # brighten by 300%
        spectrogram_data_plot = spectrogram_data_plot * brightness

        # clip back to [0, 255] range
        spectrogram_data_plot = np.clip(spectrogram_data_plot, 0.0, 255.0)



    # init plots
    fig=plt.figure()
    gs1 = gridspec.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs1[0])
    ax2 = fig.add_subplot(gs1[1])

    print ("\nPlots:\nTensorflow website: Because of TensorFlow's memory order, time in this image is increasing from top to bottom, with frequencies going from left to right, unlike the usual convention for spectrograms where time is left to right")

    print ("\nSpectrogram")
    print ("Shape:", np.shape(spectrogram_data))
    print ("Shape spectrogram_data[0]:", np.shape(spectrogram_data[0]))
    ax1.set(xlabel='Frequency', ylabel='Time (with sliding windows of x ms)', title="Spectrogram")
    ax1.imshow(spectrogram_data_plot, cmap='magma')

    print ("\nMFCC")
    print ("Shape:", np.shape(mfcc_data))
    print ("Shape spectrogram_data[0]:", np.shape(mfcc_data[0]))
    print ("MFCC has 40 coefficients")
    ax2.set(xlabel='Mel Frequency Cepstrum Coefficients', ylabel='Time (with sliding windows of x ms)', title="MFCC")
    ax2.imshow(mfcc_data_plot, cmap='gray')

    gs1.tight_layout(fig)
    ax = plt.gca()
    plt.show()
