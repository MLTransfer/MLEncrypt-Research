# -*- coding: utf-8 -*-
# usage: python projector.py [LOGDIR]
from os import path
from sys import argv
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

LOG_DIR = argv[1]

# Create randomly initialized embedding weights which will be trained.
N = 27783  # number of items
D = 8  # dimensions
embedding_var = tf.Variable(tf.random_normal([N, D]), name='hparams_embedding')

config = projector.ProjectorConfig()

embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = path.join(LOG_DIR, 'metadata.tsv')

summary_writer = tf.summary.FileWriter(LOG_DIR)

# write projector_config.pbtxt to LOG_DIR
projector.visualize_embeddings(summary_writer, config)
