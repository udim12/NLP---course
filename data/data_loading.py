# Download the data and save it as tfrecord files (here not sure how to access the training/testing data only)
import tensorflow_datasets as tfds
import tensorflow

ds, info = tfds.load("cnn_dailymail", split="train", try_gcs=True, with_info=True)

