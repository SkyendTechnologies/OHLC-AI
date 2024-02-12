# # pip install pandas && numpy && tensorflow && sklearn && matplotlib
# # source env/bin/activate
# import os
# import tarfile
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# from  os import open
# from six.moves import cPickle
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
# from tensorflow.python.keras.backend import backend


# # from __future__ import absolute_import
# # from __future__ import division
# # from __future__ import print_function

# def neuron_1(neuron_input):
#     set_a = np.array([7.08, 1, 1.01, 1.4], dtype=float)
#     set_b = np.array([7, 0.39, 5.5, 1], dtype=float)
#     set_c = np.array([24, 24, 58], dtype=float)
#     set_d = np.array([24, 58, 33], dtype=float)

#     for i, c in enumerate(set_a):
#         print("{} Set A = {} Set B".format(c, set_b[i]))
#     for i, c in enumerate(set_c):
#         print("{} Set C = {} Set D".format(c, set_d[i]))

#     neuron_model_D1 = Sequential(
#         [
#             layers.Dense(1, input_shape=[1], use_bias=False, activation="relu", name="neuron_input"),
#             layers.Dense(2, use_bias=False, activation="relu", name="neuron_1"),
#             layers.Dense(4, use_bias=False, activation="relu", name="neuron_2"),
#             layers.Dense(8, use_bias=False, activation="relu", name="neuron_3"),
#             layers.Dense(16, use_bias=False, activation="relu", name="neuron_4"),
#             layers.Dense(32, use_bias=False, activation="relu", name="neuron_5"),
#             # layers.Dense(64, use_bias = False, activation="relu", name="neuron_6"),
#             # layers.Dense(128, use_bias = False, activation="relu", name="neuron_7"),
#             # layers.Dense(256, use_bias = False, activation="relu", name="neuron_8"),
#             # layers.Dense(512, use_bias = False, activation="relu", name="neuron_9"),
#             # layers.Dense(1024, use_bias = False, activation="relu", name="neuron_10"),
#             # layers.Dense(2048, use_bias = False, activation="relu", name="neuron_11"),
#             # layers.Dense(4096, use_bias = False, activation="relu", name="neuron_12"),
#             # layers.Dense(8192, use_bias = False, activation="relu", name="neuron_13"),
#             # layers.Dense(16384, use_bias = False, activation="relu", name="neuron_14"),
#             # layers.Dense(32768, use_bias = False, activation="relu", name="neuron_15"),
#             # layers.Dense(65536, use_bias = False, activation="relu", name="neuron_16"),
#             layers.Dense(1, use_bias=False, name="neuron_output")

#         ]
#     )

    
#     neuron_model_D2 = Sequential(
#         [
#             layers.Dense(1, input_shape=[1], use_bias=False, activation="relu", name="neuron_input"),
#             layers.Dense(2, use_bias=False, activation="relu", name="neuron_1"),
#             layers.Dense(4, use_bias=False, activation="relu", name="neuron_2"),
#             layers.Dense(8, use_bias=False, activation="relu", name="neuron_3"),
#             layers.Dense(16, use_bias=False, activation="relu", name="neuron_4"),
#             layers.Dense(32, use_bias=False, activation="relu", name="neuron_5"),
#             # layers.Dense(64, use_bias = False, activation="relu", name="neuron_6"),
#             # layers.Dense(128, use_bias = False, activation="relu", name="neuron_7"),
#             # layers.Dense(256, use_bias = False, activation="relu", name="neuron_8"),
#             # layers.Dense(512, use_bias = False, activation="relu", name="neuron_9"),
#             # layers.Dense(1024, use_bias = False, activation="relu", name="neuron_10"),
#             # layers.Dense(2048, use_bias = False, activation="relu", name="neuron_11"),
#             # layers.Dense(4096, use_bias = False, activation="relu", name="neuron_12"),
#             # layers.Dense(8192, use_bias = False, activation="relu", name="neuron_13"),
#             # layers.Dense(16384, use_bias = False, activation="relu", name="neuron_14"),
#             # layers.Dense(32768, use_bias = False, activation="relu", name="neuron_15"),
#             # layers.Dense(65536, use_bias = False, activation="relu", name="neuron_16"),
#             layers.Dense(1, use_bias=False, name="neuron_output")

#         ]
#     )

#     neuron_model_D3 = Sequential(
#         [
#             layers.Dense(1, input_shape=[1], use_bias=False, activation="relu", name="neuron_input"),
#             layers.Dense(2, use_bias=False, activation="relu", name="neuron_1"),
#             layers.Dense(4, use_bias=False, activation="relu", name="neuron_2"),
#             layers.Dense(8, use_bias=False, activation="relu", name="neuron_3"),
#             layers.Dense(16, use_bias=False, activation="relu", name="neuron_4"),
#             layers.Dense(32, use_bias=False, activation="relu", name="neuron_5"),
#             # layers.Dense(64, use_bias = False, activation="relu", name="neuron_6"),
#             # layers.Dense(128, use_bias = False, activation="relu", name="neuron_7"),
#             # layers.Dense(256, use_bias = False, activation="relu", name="neuron_8"),
#             # layers.Dense(512, use_bias = False, activation="relu", name="neuron_9"),
#             # layers.Dense(1024, use_bias = False, activation="relu", name="neuron_10"),
#             # layers.Dense(2048, use_bias = False, activation="relu", name="neuron_11"),
#             # layers.Dense(4096, use_bias = False, activation="relu", name="neuron_12"),
#             # layers.Dense(8192, use_bias = False, activation="relu", name="neuron_13"),
#             # layers.Dense(16384, use_bias = False, activation="relu", name="neuron_14"),
#             # layers.Dense(32768, use_bias = False, activation="relu", name="neuron_15"),
#             # layers.Dense(65536, use_bias = False, activation="relu", name="neuron_16"),
#             layers.Dense(1, use_bias=False, name="neuron_output")

#         ]
#     )

#     neuron_model_D4 = Sequential(
#         [
#             layers.Dense(1, input_shape=[1], use_bias=False, activation="relu", name="neuron_input"),
#             layers.Dense(2, use_bias=False, activation="relu", name="neuron_1"),
#             layers.Dense(4, use_bias=False, activation="relu", name="neuron_2"),
#             layers.Dense(8, use_bias=False, activation="relu", name="neuron_3"),
#             layers.Dense(16, use_bias=False, activation="relu", name="neuron_4"),
#             layers.Dense(32, use_bias=False, activation="relu", name="neuron_5"),
#             # layers.Dense(64, use_bias = False, activation="relu", name="neuron_6"),
#             # layers.Dense(128, use_bias = False, activation="relu", name="neuron_7"),
#             # layers.Dense(256, use_bias = False, activation="relu", name="neuron_8"),
#             # layers.Dense(512, use_bias = False, activation="relu", name="neuron_9"),
#             # layers.Dense(1024, use_bias = False, activation="relu", name="neuron_10"),
#             # layers.Dense(2048, use_bias = False, activation="relu", name="neuron_11"),
#             # layers.Dense(4096, use_bias = False, activation="relu", name="neuron_12"),
#             # layers.Dense(8192, use_bias = False, activation="relu", name="neuron_13"),
#             # layers.Dense(16384, use_bias = False, activation="relu", name="neuron_14"),
#             # layers.Dense(32768, use_bias = False, activation="relu", name="neuron_15"),
#             # layers.Dense(65536, use_bias = False, activation="relu", name="neuron_16"),
#             layers.Dense(1, use_bias=False, name="neuron_output")

#         ]
#     )



#     x = tf.ones((4, 1))
#     neuron_model_D1(x)
#     neuron_model_D1.summary()
#     weights_A = neuron_model_D1.get_weights()
#     print("weights A = {}".format(weights_A))

#     # TODO: WORKING

#     while True:
#         try:
#             neuron_model_D1.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
#             neuron_fit = neuron_model_D1.fit(set_a, set_b, epochs=500, batch_size=32, verbose=False)
#             print("Завершили тренировку модели")
#             neuron_data = neuron_model_D1.predict(neuron_input)

#         except KeyboardInterrupt:
#             break

#         if np.bool_(neuron_data):
#             weights_get = neuron_model_D1.get_weights() # get weights for save 
#             refloat = np.float(weights_get)
#             relist = np.list(refloat)
#             with open('/save_weights/neuron_1_weights.text', 'wb') as f:
#                 f.write(relist)
#             print(neuron_data)
    
#             break
#         else:
#             print('Not data')

#             a = np.array(neuron_model_D1.get_weights())  # save weights in a np.array of np.arrays
#             r = np.random.rand(a)    #  random
#             print(r)
#             neuron_model_D1.set_weights(a + r)            # add 1 to all weights in the neural network
#             b = np.array(neuron_model_D1.get_weights())  # save weights a second time in a np.array of np.arrays
#             # print(b - a)                                   # print changes in weights
#             # weights_B = neuron_model_D1.get_weights()
#             # print('weights_B = {}'.format(weights_B))
#     neuron_output = neuron_data
#     return neuron_output, neuron_model_D1
    
# def neuron_2(neuron_input):
    
#     """Cifar100 dataset preprocessing and specifications."""

#     REMOTE_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
#     LOCAL_DIR = os.path.join("data/cifar100/")
#     ARCHIVE_NAME = "cifar-100-python.tar.gz"
#     DATA_DIR = "cifar-100-python/"
#     TRAIN_BATCHES = ["train"]
#     TEST_BATCHES = ["test"]

#     IMAGE_SIZE = 32
#     NUM_CLASSES = 100

#     def get_params():
#         """Return dataset parameters."""
#         return {
#             "image_size": IMAGE_SIZE,
#             "num_classes": NUM_CLASSES,
#         }

#     def prepare():
#         """Download the cifar 100 dataset."""
#         if not os.path.exists(LOCAL_DIR):
#             os.makedirs(LOCAL_DIR)
#         if not os.path.exists(LOCAL_DIR + ARCHIVE_NAME):
#             print("Downloading...")
#             urllib.request.urlretrieve(REMOTE_URL, LOCAL_DIR + ARCHIVE_NAME)
#         if not os.path.exists(LOCAL_DIR + DATA_DIR):
#             print("Extracting files...")
#             tar = tarfile.open(LOCAL_DIR + ARCHIVE_NAME)
#             tar.extractall(LOCAL_DIR)
#             tar.close()

#     def read(split):
#         """Create an instance of the dataset object."""
#         batches = {
#             tf.estimator.ModeKeys.TRAIN: TRAIN_BATCHES,
#             tf.estimator.ModeKeys.EVAL: TEST_BATCHES
#         }[split]

#         all_images = []
#         all_labels = []

#         for batch in batches:
#             with open("%s%s%s" % (LOCAL_DIR, DATA_DIR, batch), "rb") as fo:
#                 dict = cPickle.load(fo)
#                 images = np.array(dict["data"])
#                 labels = np.array(dict["fine_labels"])

#                 num = images.shape[0]
#                 images = np.reshape(images, [num, 3, IMAGE_SIZE, IMAGE_SIZE])
#                 images = np.transpose(images, [0, 2, 3, 1])
#                 print("Loaded %d examples." % num)

#                 all_images.append(images)
#                 all_labels.append(labels)

#         all_images = np.concatenate(all_images)
#         all_labels = np.concatenate(all_labels)

#         return tf.contrib.data.Dataset.from_tensor_slices((all_images, all_labels))

#     def parse(image, label):
#         """Parse input record to features and labels."""
#         image = tf.to_float(image) / 255.0
#         image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
#         return {"image": image}, {"label": label}

#     plt.figure()
#     plt.plot(epochs, loss, 'r', label='Training loss')
#     plt.plot(epochs, val_loss, 'bo', label='Validation loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss Value')
#     plt.ylim([0, 1])
#     plt.legend()
#     plt.show()

#     show_predictions(test_dataset, 3)

# neuron_2(neuron_input = None)
