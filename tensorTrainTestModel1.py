import tensorflow as tf
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, GRU, Bidirectional, Dropout
from keras.utils import plot_model

def create_model(input_window):
    '''Creates and returns the Neural Network
    '''
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(16, 4, activation='relu', input_shape=(input_window,1), padding="same", strides=1))

    #Bi-directional GRUs
    model.add(Bidirectional(GRU(64, activation='relu', return_sequences=True), merge_mode='concat'))
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(128, activation='relu', return_sequences=False), merge_mode='concat'))
    model.add(Dropout(0.5))

    # Fully Connected Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

    return model



featuresPath = '/home/nick/Downloads/low_freq/house_1/proc/channel_1.dat.npy'
labelsPath = '/home/nick/Downloads/low_freq/house_1/proc/channel_5.dat.npy'
windowSize = 10

# Load the training data into two NumPy arrays, for example using `np.load()`.
features = np.load(featuresPath)
labels = np.load(labelsPath)

features, featuresTest = np.split(features,[9000])
labels, labelsTest = np.split(labels,[9000])

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
iterator = dataset.make_initializable_iterator()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
    #print(sess.run(iterator.get_next()))
    model = create_model(windowSize)
    model.fit(features,labels,batch_size=128) 

    pred = model.predict(featuresTest)

    print(featuresTest)
    print(pred)
