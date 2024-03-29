#!/usr/bin/env python
# coding: utf-8

import json
import math
import os
import time
from collections import OrderedDict
import tensorflow as tf

from utils import SEP_TOKEN, PAD_TOKEN, VOCAB_SIZE, MODEL_DIR
from data_utils import gen_batch_train_data
from model import Seq2SeqModel
from word2vec import get_word_embedding

# Data loading parameters
tf.app.flags.DEFINE_boolean('cangtou_data', False, 'Use cangtou training data')
tf.app.flags.DEFINE_boolean('rev_data', True, 'Use reversed training data')
tf.app.flags.DEFINE_boolean('align_data', True, 'Use aligned training data')
tf.app.flags.DEFINE_boolean('prev_data', True, 'Use training data with previous sentences')
tf.app.flags.DEFINE_boolean('align_word2vec', True, 'Use aligned word2vec model')

# Network parameters
tf.app.flags.DEFINE_string('cell_type', 'lstm', 'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_string('attention_type', 'bahdanau', 'Attention mechanism: (bahdanau, luong), default: bahdanau')
tf.app.flags.DEFINE_integer('hidden_units', 512, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('depth', 4, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 512, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_integer('num_encoder_symbols', 30000, 'Source vocabulary size')
tf.app.flags.DEFINE_integer('num_decoder_symbols', 30000, 'Target vocabulary size')
# NOTE(sdsuo): We used the same vocab for source and target
tf.app.flags.DEFINE_integer('vocab_size', VOCAB_SIZE, 'General vocabulary size')

tf.app.flags.DEFINE_boolean('use_residual', True, 'Use residual connection between layers')
tf.app.flags.DEFINE_boolean('attn_input_feeding', False, 'Use input feeding method in attentional decoder')
tf.app.flags.DEFINE_boolean('use_dropout', True, 'Use dropout in each rnn cell')
tf.app.flags.DEFINE_float('dropout_rate', 0.3, 'Dropout probability for input/output/state units (0.0: no dropout)')

# Training parameters
tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('max_epochs', 10000, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('max_load_batches', 20, 'Maximum # of batches to load at one time')
tf.app.flags.DEFINE_integer('max_seq_length', 50, 'Maximum sequence length')
tf.app.flags.DEFINE_integer('display_freq', 100, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('save_freq', 5000, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_integer('valid_freq', 1150000, 'Evaluate model every this iteration: valid_data needed')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
tf.app.flags.DEFINE_string('model_dir', "model/dir", 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('summary_dir', 'model/summary', 'Path to save model summary')
tf.app.flags.DEFINE_string('model_name', 'translate.ckpt', 'File name used for model checkpoints')
tf.app.flags.DEFINE_boolean('shuffle_each_epoch', True, 'Shuffle training dataset for each epoch')
tf.app.flags.DEFINE_boolean('sort_by_length', True, 'Sort pre-fetched minibatches by their target sequence lengths')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')
tf.app.flags.DEFINE_boolean('bidirectional', True, 'Use bidirectional encoder')
tf.app.flags.DEFINE_string('train_mode', 'ground_truth', 'Decode helper to use for training')
tf.app.flags.DEFINE_float('sampling_probability', 0.1, 'Probability of sampling from decoder output instead of using ground truth')

# TODO(sdsuo): Make start token and end token more robust
tf.app.flags.DEFINE_integer('start_token', SEP_TOKEN, 'Start token')
tf.app.flags.DEFINE_integer('end_token', PAD_TOKEN, 'End token')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.app.flags.FLAGS

def load_or_create_model(sess, model, saver, FLAGS):
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)  #通过checkpoint文件找到模型文件名
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print( 'Reloading model parameters...')
        model.restore(sess, saver, ckpt.model_checkpoint_path)
    else:
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        print ('Created new model parameters...')
        sess.run(tf.global_variables_initializer())


def train():
    config_proto = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,#获取到 operations 和 Tensor 被指派到哪个设备(几号CPU或几号GPU)上运行,会在终端打
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
    with tf.Session(config=config_proto) as sess:
        # Build the model
        config = OrderedDict(sorted(FLAGS.flag_values_dict().items()))   #载入所有的预设指令参数
        model = Seq2SeqModel(config, 'train')

        # Create a log writer object

        merged = tf.summary.merge_all()
        log_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        # Create a saver
        # Using var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list=None)

        # Initiaize global variables or reload existing checkpoint
        load_or_create_model(sess, model, saver, FLAGS)

        # Load word2vec embedding
        embedding = get_word_embedding(FLAGS.hidden_units, alignment=FLAGS.align_word2vec)
        model.init_vars(sess, embedding=embedding)

        step_time, loss = 0.0, 0.0
        sents_seen = 0

        start_time = time.time()

        print ('Training...')
        for epoch_idx in range(FLAGS.max_epochs):
            if model.global_epoch_step.eval() >= FLAGS.max_epochs:
                print ('Training is already complete.', \
                      'Current epoch: {}, Max epoch: {}'.format(model.global_epoch_step.eval(), FLAGS.max_epochs))
                break


            # Prepare batch training data
            # TODO(sdsuo): Make corresponding changes in data_utils
            #source  [batch_size_number,batch_size,max_len]
            for keywordes, keyword_lengths, contents,content_lengths,targets, target_lengths in gen_batch_train_data(FLAGS.batch_size,
                                                                               prev=FLAGS.prev_data,
                                                                               rev=FLAGS.rev_data,
                                                                               align=FLAGS.align_data,
                                                                               cangtou=FLAGS.cangtou_data):
                step_loss, summary = model.train(
                    sess,
                    content_inputs=contents,
                    content_inputs_length=content_lengths,
                    keyword_inputs=keywordes,
                    keyword_length=keyword_lengths,
                    decoder_inputs=targets,
                    decoder_inputs_length=target_lengths
                )

                loss += float(step_loss) / FLAGS.display_freq
                sents_seen += float(keywordes.shape[0]) # batch_size

                # Display information
                if model.global_step.eval() % FLAGS.display_freq == 0:

                    avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")

                    time_elapsed = time.time() - start_time
                    step_time = time_elapsed / FLAGS.display_freq

                    sents_per_sec = sents_seen / time_elapsed

                    print( 'Epoch ', model.global_epoch_step.eval(), 'Step ', model.global_step.eval(), \
                          'Perplexity {0:.2f}'.format(avg_perplexity), 'Step-time ', step_time, \
                           "loss",loss,
                          '{0:.2f} sents/s'.format(sents_per_sec))

                    loss = 0
                    sents_seen = 0
                    start_time = time.time()

                    # Record training summary for the current batch
                    log_writer.add_summary(summary, model.global_step.eval())

                # Save the model checkpoint
                if model.global_step.eval() % FLAGS.save_freq == 0:
                    print( 'Saving the model..')
                    checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                    model.save(sess, saver, checkpoint_path, global_step=model.global_step)
                    json.dump(model.config,
                              open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'w'),
                              indent=2)
            # Increase the epoch index of the model
            model.increment_global_epoch_step_op.eval()
            print( 'Epoch {0:} DONE'.format(model.global_epoch_step.eval()))


        print ('Saving the last model')
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
        model.save(sess, saver, checkpoint_path, global_step=model.global_step)
        json.dump(model.config,
                  open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'w'),
                  indent=2)

    print ('Training terminated')


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run() #执行main函数
