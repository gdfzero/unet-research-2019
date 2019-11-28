import time

from utils import *
from six.moves import xrange
import numpy as np
import pdb
import tensorflow as tf

import os

from PIL import Image

def dncnn(ldct_img, is_training= True):
    """
    Defines the layer configurations and parameters in the contracting,
    expanding paths and produces the residual mapping as output
    """
    net = ldct_img
    pdb.set_trace()     
    conv1, pool1 = conv_conv_pool(net, [16,16], is_training, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [32,32], is_training, name=3)
    conv3, pool3 = conv_conv_pool(pool2, [64,64], is_training, name=5)
    conv4 = conv_conv_pool(pool3, [128,128], is_training, name=7, pool=False)

    up6 = upconv_concat(conv4, conv3, 64, name=11)
    conv5 = conv_conv_pool(up6, [64,64], is_training, name=12, pool=False)

    up7 = upconv_concat(conv5, conv2, 32, name=13)
    conv6 = conv_conv_pool(up7, [32,32], is_training, name=14, pool=False)

    up8 = upconv_concat(conv6, conv1, 16, name=15)
    conv7 = conv_conv_pool(up8, [16,16], is_training, name=16, pool=False)
    
    output = tf.layers.conv2d(
        conv7,
        1, (1, 1),
        name='final',
        padding='same')
    # activation = tf.nn.relu)
    denoised_image = ldct_img - output
    return denoised_image

def conv_conv_pool(input_, n_filters, is_training, name, pool=True, activation=tf.nn.relu):
    """
    Performs convolutions on the input feature map and pools if specified
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (3, 3),
                activation=None,
                padding='same',
#                kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg))
                name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(net, training=is_training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))
        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool
def upconv_concat(inputA, input_B, n_filter,  name):
    """
    Takes the transposed convolution of the feature map from the expanding
    path and concatenates this with the input feature map from the skip layer
    """
    up_conv = upconv_2D(inputA, n_filter, name)

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))


def upconv_2D(tensor, n_filter, name):
    """
    Takes the tranposed convulution of the input feature map
    """

    return tf.layers.conv2d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        #kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
        name="upsample_{}".format(name))

def dncnn2(input, is_training=True, output_channels=1):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in xrange(2, 16 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block17'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return input - output


class denoiser(object):
    def __init__(self, sess, input_c_dim=1, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='ndct_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.X = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='ldct_image')
        self.Y = dncnn(self.X, is_training=self.is_training)
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, ndct_test_data,ldct_test_data, sample_dir, summary_merged, summary_writer, summ_img):
        # assert test_data value range is 0-255
        print("[*] Evaluating...")
        psnr_sum = 0      
        for idx in xrange(len(ldct_test_data)):
            noisy_image = ldct_test_data[idx]
            clean_image = ndct_test_data[idx]
            output_clean_image, psnr_summary, temp_img = self.sess.run(
                [self.Y, summary_merged, summ_img],
                feed_dict={self.X: noisy_image,
                           self.Y_: clean_image,
                           self.is_training: False})
            summary_writer.add_summary(psnr_summary, iter_num)
            summary_writer.add_summary(temp_img, iter_num)
            scalef= max(np.amax(clean_image), np.amax(noisy_image), np.amax(output_clean_image))
            clean_image = np.clip(255 * clean_image/scalef, 0, 255).astype('uint8')
            noisy_image = np.clip(255 * noisy_image/scalef, 0, 255).astype('uint8')
            output_clean_image = np.clip(255 * output_clean_image/scalef, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(clean_image, output_clean_image)
            print("img%d PSNR: %.2f" % (idx + 1, psnr))
            psnr_sum += psnr
            clean_image, noisy_image= arr2Img(clean_image, noisy_image)
            save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)),
                        clean_image, noisy_image, output_clean_image)
        avg_psnr = psnr_sum / len(ndct_test_data)

        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)

#    def denoise(self, data):
#        output_clean_image, noisy_image, psnr = self.sess.run([self.Y, self.X, self.eva_psnr],
#                                                              feed_dict={self.Y_: data, self.is_training: False})
#        return output_clean_image, noisy_image, psnr

    def train(self, ndct_data, ldct_data, ndct_eval_data, ldct_eval_data, batch_size, ckpt_dir, epoch, lr, sample_dir, eval_every_epoch=1):
        pdb.set_trace()
        # assert data range is between 0 and 1
        numBatch = int(ndct_data.shape[0] / batch_size)
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        img = tf.summary.image('denoised image', self.Y, max_outputs=1)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, ndct_eval_data, ldct_eval_data, sample_dir=sample_dir, summary_merged=summary_psnr,
                      summary_writer=writer, summ_img=img)  # eval_data value range is 0-255
        for epoch in xrange(start_epoch, epoch):
            p = np.random.permutation(len(ndct_data))
            ndct_data, ldct_data = ndct_data[p], ldct_data[p]    #ensure shuffling in unison
            #pdb.set_trace()
            for batch_id in xrange(start_step, numBatch):
                ndct_batch_images, ldct_batch_images = ndct_data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :], ldct_data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                _, loss, summary = self.sess.run([self.train_op, self.loss, merged],
                                                 feed_dict={self.Y_: ndct_batch_images, self.X:ldct_batch_images, self.lr: lr[epoch],
                                                            self.is_training: True})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1
                writer.add_summary(summary, iter_num)
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, ndct_eval_data, ldct_eval_data, sample_dir=sample_dir, summary_merged=summary_psnr,
                              summary_writer=writer, summ_img=img)  # eval_data value range is 0-255
                self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")

    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, ldct_files, ndct_files, ckpt_dir, save_dir):
        """Test DnCNN"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(ldct_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        print("[*] start testing...")
        rawfiles= [open(os.path.join(save_dir, "test_{num:08d}.flt".format(num=idx)), 'wb') for idx in range (len(ndct_files))]
        for idx in xrange(len(ldct_files)):
            noisy_image= ldct_files[idx]
            clean_image= ndct_files[idx]
            output_clean_image = self.sess.run(
                [self.Y],
                feed_dict={self.X: noisy_image,
                           self.Y_: clean_image,
                           self.is_training: False})
            output_clean_image= np.asarray(output_clean_image)
            #output_clean_image= output_clean_image[255, :, :, :, :]
            #scalef= max(np.amax(clean_image), np.amax(output_clean_image))
            #noisy_image = np.clip(255 * noisy_image/scalef, 0, 255).astype('uint8')
            #scaled_output = np.clip(255 * output_clean_image/scalef, 0, 255).astype('uint8')
            #clean_image = np.clip(255 * clean_image/scalef, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(clean_image, output_clean_image)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            #output_clean_image, noisy_image= arr2Img(output_clean_image, noisy_image)
            #clean_image= np.reshape(clean_image, (512, 512))
            #clean_image= Image.fromarray(clean_image, 'L')
            #save_images(os.path.join(save_dir, 'test_%d.flt' % (idx + 1)),
            #            clean_image, noisy_image, output_clean_image)
            output_clean_image.tofile(rawfiles[idx])
        avg_psnr = psnr_sum / len(ndct_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)
