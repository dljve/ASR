import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
import warnings
from os import listdir
from os.path import isfile, join
from graphviz import Digraph
from imageio import imwrite
import scipy.spatial as sp
# make sure you have scipy, imageio and graphviz


# Before running, make sure you have access to tensorflow (e.g. virtual env)



warnings.simplefilter(action='ignore', category=FutureWarning)
sess = False


# Settings

# dataset path
dataset_path = "/tmp/speech_dataset/"

# model to load
modelType = "spectrogram"  # "mfcc", "spectrogram"
modelFilters = 64   # 64, 32, 16, 8

# If you want to find a wav file to visualize stuff of, set this to True
# It will predict labels for wav files in the train set -> get accuracy
predictLabels = False # set to true to predict labels
predictOn = "go" # set to wav file folder you want to test wav files of

# Visualize filters of model
visualizeFirstConvLayer = False
visualizeSecondConvLayers = False

# Visualize activations of a wav file
wavFile = dataset_path + "/no/7846fd85_nohash_1.wav"
visualizeConvLayerActivations = False
highlightConvFilterI = 0 # bigger single plot of activiation of the i-th filter

visualizeReLuActivations = True
highlightReLuFilterI = 17 # bigger single plot of activiation of the i-th filter

visualizeMaxpoolActivations = False
highlightMaxpoolFilterI = 0 # bigger single plot of activiation of the i-th filter

# Visualize hierarchy
visualizeHierarchy = False
similarityMeasure = "cosine" # "cosine" or "SSE"



def load_model(checkpoint,first_layer_filters=64):
    global sess
    tf.reset_default_graph()
    sess = tf.Session()

    v = tf.get_variable("Variable", shape=[20,8,1,first_layer_filters])
    v1 = tf.get_variable("Variable_1", shape=[first_layer_filters])
    v2 = tf.get_variable("Variable_2", shape=[10,4,first_layer_filters,64])
    v3 = tf.get_variable("Variable_3", shape=[64])
    v4 = tf.get_variable("Variable_4", shape=[62720,12]) if modelType == "mfcc" else tf.get_variable("Variable_4", shape=[404544,12])
    v5 = tf.get_variable("Variable_5", shape=[12])
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess.run(init)

    saver.restore(sess, checkpoint)
    first_weights = np.asarray(v.eval(session=sess))
    first_bias = np.asarray(v1.eval(session=sess))
    second_weights = np.asarray(v2.eval(session=sess))
    second_bias = np.asarray(v3.eval(session=sess))
    final_fc_weights = np.asarray(v4.eval(session=sess))
    final_fc_bias = np.asarray(v5.eval(session=sess))

    weights = {'first_weights':first_weights, 'first_bias':first_bias,
                 'second_weights':second_weights, 'second_bias':second_bias,
                 'final_fc_weights':final_fc_weights, 'final_fc_bias':final_fc_bias}

    return {'session':sess, 'weights':weights}




def wav_to_spectogram(wav, mode="original", plot=False):
    """
    Convert a wav file to a spectrogram and mfcc picture

    param wav (string):     path with name to wav file
    param mode (string):    "original" to display as used by tf code, or "enhanced"
                            for a visualization more human readable
    param plot (bool):      to plot or not
    """

    if mode == "original":
        model_settings = {'dct_coefficient_count': 40,
                          'window_size_samples': 480,
                          'label_count': 12,
                          'desired_samples': 16000,
                          'window_stride_samples': 160,
                          'spectrogram_length': 98,
                          'sample_rate': 16000,
                          'fingerprint_size': 3920}
    else:
        # settings from wav_to_spectrogram script from tensorflow tutorial
        model_settings = {'dct_coefficient_count': 40,
                          'window_size_samples': 256,
                          'label_count': 12,
                          'desired_samples': 16000,
                          'window_stride_samples': 128,
                          'spectrogram_length': 98,
                          'sample_rate': 16000,
                          'fingerprint_size': 3920}

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
        wav_filename_placeholder_: wav, # path to file we want to analyze
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
        brightness = 3 # brighten by 500%
        spectrogram_data_plot = spectrogram_data_plot * brightness

        # clip back to [0, 255] range
        spectrogram_data_plot = np.clip(spectrogram_data_plot, 0.0, 255.0)

    if plot:
        # init plots
        print ("\nSpectrogram data spectrogram: %s" % str(np.shape(spectrogram_data[0])))
        print ("MFCC data shape: %s" % str(np.shape(mfcc_data[0])))
        print ("MFCC has 40 coefficients")

        input_time_size = spectrogram_data.shape[1]
        input_frequency_size = spectrogram_data.shape[2]
        fig2=plt.figure(figsize=(8, 20))
        fig2.suptitle("Spectrogram of wav file")
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.imshow(np.rot90(spectrogram_data_plot), cmap='binary')
        plt.xticks([i*input_time_size/10 for i in range(10)], range(0,1000,100))
        plt.yticks([i*input_frequency_size/39.8 for i in range(40)], range(8000,0,-200))


        # fig3=plt.figure(figsize=(8, 20))
        # fig3.suptitle("MFCC of wav file")
        # plt.xlabel('Time')
        # plt.ylabel('Mel Frequency Cepstrum Coefficients')
        # plt.imshow(np.rot90(mfcc_data_plot), cmap='gray')

    return spectrogram_data, mfcc_data

# create 4D fingerprint from MFCC or spectrogram
def create_fingerprint(data):
    input_frequency_size = data.shape[2] # model_settings['dct_coefficient_count']
    input_time_size = data.shape[1] #model_settings['spectrogram_length']
    fingerprint_input = data
    fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])

    return fingerprint_4d

def predict(fingerprint_4d, weights):
    # initialize weights
    first_weights = weights["first_weights"]
    first_bias = weights["first_bias"]
    second_weights = weights["second_weights"]
    second_bias = weights["second_bias"]
    final_fc_weights = weights["final_fc_weights"]
    final_fc_bias = weights["final_fc_bias"]

    # first pass
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1], 'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)
    max_pool = tf.nn.max_pool(first_relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    # second pass
    second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1], 'SAME') + second_bias
    second_relu = tf.nn.relu(second_conv)
    second_conv_shape = second_relu.get_shape()
    second_conv_output_width = second_conv_shape[2]
    second_conv_output_height = second_conv_shape[1]
    second_conv_element_count = int(second_conv_output_width * second_conv_output_height * 64)
    flattened_second_conv = tf.reshape(second_relu, [-1, second_conv_element_count])
    final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias

    # prediction
    pred = tf.nn.softmax(final_fc).eval(session=sess)
    labels = ["_silence_", "_unknown_", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    pred_label = labels[np.argmax(pred)]
    accuracy = np.max(pred)

    return [pred_label, accuracy, first_conv, first_relu, max_pool]



def predict_labels(weights):

    ## Predict labels
    predictions = []
    input_dir = dataset_path + predictOn + '/'
    wavs = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    print('Predicting labels for %d wav files in %s' % (len(wavs), input_dir))

    # only check specific files
    # wavs = ["813b82a6_nohash_0.wav"]

    # Predict each wav in a dictionary
    for wav in wavs:
        input_wav = input_dir + wav

        [spectrogram, mfcc] = wav_to_spectogram(input_wav)
        data  = spectrogram if modelType == "spectrogram" else mfcc
        fingerprint = create_fingerprint(data)
        label, accuracy, _, _, _ = predict(fingerprint, weights)

        predictions.append((wav, label, accuracy))

        if accuracy > 0.9:
            print("%s, %s, %0.3f <-- High accuracy" % (wav, label, accuracy))
        else:
            print("%s, %s, %0.3f" % (wav, label, accuracy))


def restore_model():
    global sess

    ## Restore model
    steps = 36000 if modelType == "spectrogram" else 18000

    modelPath = "../model_data/" + modelType+ "_" + str(modelFilters)+ "f_" + str(steps) + "s/speech_commands_train/conv.ckpt-" + str(steps)
    print ('Loading model from %s' % modelPath)
    restore = load_model(modelPath, first_layer_filters=modelFilters)
    sess = restore["session"]
    weights = restore["weights"]
    return weights


def visFirstConvLayers(weights):
    print('Visualizing first conv layer filters')
    first_weights = weights["first_weights"]

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("First conv layer filters")

    for i in range(first_weights.shape[3]):
        fig.add_subplot(8, 8, i+1)
        plt.imshow(first_weights[:,:,0,i], cmap='gray')


def visSecondConvLayers(weights):
    print('Visualizing second conv layer filters')
    first_weights = weights["first_weights"]
    second_weights = weights["second_weights"]

    fig=plt.figure(figsize=(10, 10))
    fig.suptitle("Second conv layer filters")
    channel = 0

    for i in range(first_weights.shape[3]):
        fig.add_subplot(8, 8, i+1)
        plt.imshow(second_weights[:,:,channel,i], cmap='gray')


def visConvLayerActivations(data, first_conv):
    print ("Visualizing activations in first_conv layer by wav file ")
    act_conv = np.asarray(first_conv.eval(session=sess))
    input_time_size = data.shape[1]
    input_frequency_size = data.shape[2]

    # All activations
    fig=plt.figure(figsize=(10, 10))
    fig.suptitle("Activation in all filters of first conv layer")
    for i in range(modelFilters):
        fig.add_subplot(8, 8, i+1)
        im = act_conv[0,:,:,i].reshape((input_time_size, input_frequency_size))
        plt.imshow(im, cmap='jet')

    # Single activation
    global highlightConvFilterI
    if highlightConvFilterI > modelFilters:
        print ("HighlightConvFilterI can't be bigger than the number of filters in the model. Displaying first filter.")
        highlightConvFilterI = 0
    fig = plt.figure(figsize=(8, 20))
    fig.suptitle("Highlight: Activation in %d-th filter of first_conv layer" % (highlightConvFilterI + 1))
    im = act_conv[0,:,:,highlightConvFilterI].reshape((input_time_size, input_frequency_size))
    plt.imshow(np.rot90(im), cmap='jet')
    plt.colorbar(fraction=0.02)


def visReLuActivations(data, first_relu):
    act_relu = np.asarray(first_relu.eval(session=sess))

    input_time_size = data.shape[1]
    input_frequency_size = data.shape[2]

    # plot every filter if < 16 filters
    if modelFilters < 16:
        for i in range(modelFilters):
            # large plot
            fig=plt.figure(figsize=(8, 20))
            fig.suptitle("Highlight: Activation in %d-th filter of first relu layer" % (i + 1))
            im = act_relu[0,:,:,i].reshape((input_time_size,input_frequency_size))
            plt.imshow(np.rot90(im), cmap='jet')
            plt.xticks([i*input_time_size/10 for i in range(10)], range(0,1000,100))
            plt.yticks([i*input_frequency_size/39.8 for i in range(40)], range(8000,0,-200))

            plt.colorbar(fraction=0.02)
    else:
        # show overview
        fig = plt.figure(figsize=(20, 20))
        fig.suptitle("Activation in all relu filters of first layer")
        for i in range(modelFilters):

            # plot overview of all filters small in one plot if too many filters
            fig.add_subplot(6, 11, i+1)
            im = act_relu[0,:,:,i].reshape((input_time_size,input_frequency_size))
            plt.imshow(np.rot90(im), cmap='jet')


    # plot the defined filter at the top of this file big if there are many filters
    if modelFilters >= 16:
        # Single activation
        global highlightReLuFilterI
        if highlightReLuFilterI > modelFilters:
            print ("HighlightReluFilterI can't be bigger than the number of filters in the model. Displaying first filter.")
            highlightReLuFilterI = 0
        fig=plt.figure(figsize=(8, 20))
        fig.suptitle("Highlight: Activation in %d-th filter of first relu layer" % (highlightReLuFilterI + 1))
        im = act_relu[0,:,:,highlightReLuFilterI].reshape((input_time_size,input_frequency_size))
        plt.imshow(np.rot90(im), cmap='jet')

        # Show a nice axis
        plt.xticks([i*input_time_size/10 for i in range(10)], range(0,1000,100))
        plt.yticks([i*input_frequency_size/39.8 for i in range(40)], range(8000,0,-200))

        plt.colorbar(fraction=0.02)


def visMaxpoolActivations(data, max_pool):
    act_maxpool = np.asarray(max_pool.eval(session=sess))

    input_time_size = data.shape[1]
    input_frequency_size = data.shape[2]

    # All activations
    fig=plt.figure(figsize=(10, 10))
    fig.suptitle("Activation in all max_pool filters of first layer")
    for i in range(modelFilters):
        fig.add_subplot(8, 8, i+1)
        im = act_maxpool[0,:,:,i].reshape((int(round(input_time_size/2.0)),int(round(input_frequency_size/2.0))))
        plt.imshow(im, cmap='gray')


    # All activations (big)
    global highlightMaxpoolFilterI
    if highlightMaxpoolFilterI > modelFilters:
        print ("HighlightMaxpoolFilterI can't be bigger than the number of filters in the model. Displaying first filter.")
        highlightMaxpoolFilterI = 0
    fig=plt.figure(figsize=(8, 20))
    fig.suptitle("Highlight: Activation in %d-th filter of first maxpool layer" % (highlightMaxpoolFilterI + 1))
    im = act_maxpool[0,:,:,highlightMaxpoolFilterI].reshape((int(round(input_time_size/2.0)),int(round(input_frequency_size/2.0))))
    plt.imshow(np.rot90(im), cmap='jet')
    plt.colorbar(fraction=0.02)


# Sum of squared differences
def SSE(A, B):
    return np.square(np.subtract(A, B)).sum()

# Cosine similarity
def cosine(A, B):
    return (1 - sp.distance.cdist(A, B, 'cosine'))[0,1]

# write the filters to png files
def saveFilters():
    steps = 36000 if modelType == "spectrogram" else 18000
    filters = {}

    for model in [8,16,32,64]:
        # Load model and restore first layer weights
        restore = load_model(("../model_data/" + modelType + "_{}f_" + str(steps) + "s/speech_commands_train/conv.ckpt-" + str(steps)).format(model),
                            first_layer_filters=model)
        first_weights = restore["weights"]["first_weights"]

        for i in range(first_weights.shape[3]):
            f = first_weights[:,:,0,i]

            # Save the filter as matrix to a dictionary
            filters['{}_{}'.format(model,i+1)] = f

            # Save weight as image for visualization
            im = (255.*(f-f.min())/(f.max()-f.min())).astype(np.uint8)
            imwrite('../savedFilters/{}_{}.png'.format(model,i+1), np.rot90(im))

    return filters


def visHierarchy(filters):
    from graphviz import Digraph
    dot = Digraph(comment='The Round Table')

    # Similarity measure to use
    measure = cosine if similarityMeasure == "cosine" else SSE

    # Load all images as graph nodes
    for size in [64,32,16,8]:
        for i in range(size):
            dot.node('{}_{}'.format(size,i), image='../savedFilters/{}_{}.png'.format(size,i+1), label='', style="setlinewidth(0)")

    # For each parent-child layer
    for pair in [(8,16), (16,32), (32,64)]:
        parent_n, child_n = pair
        # For each child in the child layer
        for child in range(1,child_n):
            # Calculate similarity with all parents
            similarities = [measure(filters['{}_{}'.format(child_n, child)],
                                    filters['{}_{}'.format(parent_n, i)]) for i in range(1,parent_n)]
            # Choose the best parent
            parent = 1 + np.argmin(similarities)
            # Add graph edge from parent to child
            dot.edge('{}_{}'.format(parent_n, parent),'{}_{}'.format(child_n, child))

    dot.render('{}_hierarchy.png'.format(measure.__name__), view=True)


def main():

    weights = restore_model()

    # predict labels of wav files, to see the accuracy on each wav file
    if predictLabels:
        predict_labels(weights)

    # Visualize filters
    if visualizeFirstConvLayer:
        visFirstConvLayers(weights)
    if visualizeSecondConvLayers:
        visSecondConvLayers(weights)

    # check if we need to visualize any activations
    if any([visualizeConvLayerActivations, visualizeReLuActivations, visualizeMaxpoolActivations]):
        # get the convolution, relu and maxpool variables from the model for a wav file
        print ("Using %s for activations" % wavFile)
        spectrogram, mfcc = wav_to_spectogram(wavFile, plot=True)
        data = spectrogram if modelType == "spectrogram" else mfcc
        fingerprint = create_fingerprint(data)
        _, _, first_conv, first_relu, max_pool = predict(fingerprint, weights)
        print ("")

        # visualize activations
        if visualizeConvLayerActivations:
            visConvLayerActivations(data, first_conv)
        if visualizeReLuActivations:
            visReLuActivations(data, first_relu)
        if visualizeMaxpoolActivations:
            visMaxpoolActivations(data, max_pool)

    # Visualize the hierarchy
    if visualizeHierarchy:
        filters = saveFilters()
        visHierarchy(filters)

    plt.show()

if __name__ == "__main__":
    main()
