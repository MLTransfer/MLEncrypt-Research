# -*- coding: utf-8 -*-
from torch.utils.tensorboard import SummaryWriter
from numpy import genfromtxt
data = genfromtxt(
    '/Users/suman/quantum/mltransfer/mlencrypt-research/results/analysis/hparams/rawdata.csv', delimiter=',', dtype=None)[1:-1]
writer = SummaryWriter()
writer.add_embedding(data, tag='hparams_embedding')
writer.close()
"""
import tensorflow as tf
from tensorboard.plugins import projector
from numpy import genfromtxt

TENSORBOARD_DIR = 'logs/projector/'

dataset = genfromtxt(
    '/Users/suman/quantum/mltransfer/mlencrypt-research/results/analysis/hparams/rawdata.csv', delimiter=',', dtype=None)
embeddings = tf.Variable(tf.convert_to_tensor(
    dataset[1:-1]), name='hparams_embeddings')
tf.print(
    # tf.map_fn(lambda x: tf.feature_column.embedding_column(x[0], 1), embeddings))
    tf.feature_column.embedding_column(embeddings, 1))
CHECKPOINT_FILE = TENSORBOARD_DIR + '/model.ckpt'
ckpt = tf.train.Checkpoint(embeddings=embeddings)
ckpt.save(CHECKPOINT_FILE)

reader = tf.train.load_checkpoint(TENSORBOARD_DIR)
map = reader.get_variable_to_shape_map()
key_to_use = ""
for key in map:
    if "embeddings" in key:
        key_to_use = key

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = key_to_use
# embedding.metadata_path = '/Users/suman/quantum/mltransfer/mlencrypt-transfer/results/analysis/hparams/rawdata.tsv'

writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
projector.visualize_embeddings(TENSORBOARD_DIR, config)
"""
