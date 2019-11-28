# model.py
# U-Net architecture
# Y_ - original image, clean_image [ndct: normal dose ct]
# Y - denoised image  [Y = X - noise]
# X - noisy image [ldct: low dose ct]
# noise - our model learns noise [residual image]

import tensorflow as tf
import pdb
import os
import numpy as np
from PIL import Image
from utils import *

# pdb.set_trace()

def unet(ldct_img, is_training= True):
    """
    Defines the layer configurations and parameters in the contracting,
    expanding paths and produces the residual mapping as output
    """
    net = ldct_img
    conv1, pool1 = conv_conv_pool(net, [8,8], is_training, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [16,16], is_training, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32,32], is_training, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64,64], is_training, name=4)
    conv5 = conv_conv_pool(pool4, [128,128], is_training, name=5, pool=False)

    up6 = upconv_concat(conv5, conv4, 64, name=6)
    conv6 = conv_conv_pool(up6, [64,64], is_training, name=6, pool=False)

    up7 = upconv_concat(conv6, conv3, 32, name=7)
    conv7 = conv_conv_pool(up7, [32,32], is_training, name=7, pool=False)

    up8 = upconv_concat(conv7, conv2, 16, name=8)
    conv8 = conv_conv_pool(up8, [16,16], is_training, name=8, pool=False)
    
    up9 = upconv_concat(conv8, conv1, 8, name=9)
    conv9 = conv_conv_pool(up9, [8,8], is_training, name=9, pool=False)

    output = tf.layers.conv2d(
        conv9,
        1, (1, 1),
        name='final',
        padding='same')
        #activation = tf.nn.sigmoid)
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
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
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
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
        name="upsample_{}".format(name))

def create_dataset(CT_type_TFRecord, seed=1):
    """
    Extracts TFRecords and converts them to usable tensor format
    Generates patched training data if desired (comment/uncomment last 2 lines in this method)
    """

    def parse_fn(record):
        """extracts single TFRecord and parses its feature dictionary"""
        features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'rows': tf.FixedLenFeature([], tf.int64),
            'cols': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
        }
        example= tf.parse_single_example(record, features)   #bind empty feature dictionary with extracted TFRecord
        CT_img= tf.decode_raw(example['image'], out_type=tf.float32)
        CT_img= tf.reshape(CT_img, [512,512])
        CT_img= tf.expand_dims(CT_img, axis=-1)

        img_shape= tf.stack([example['rows'], example['cols'], example['channels']])
        filename= example['filename']
        return CT_img

    # def create_patches(record, seed=1):
    #     patches= []
    #     temp_record= tf.image.crop_to_bounding_box(record, 0, 100, 512, 412)       # reducing sample area of patches (getting rid of empty space on left of 512*512 img)
    #     for i in range(500):
    #         patch= tf.random_crop(temp_record, [128,128,1], seed=seed) #### RANDOM CROP IS A BIG PROBLEM!! Will cause mismatch in ldct/ndct data...
    #         patches.append(patch)
    #     patches= tf.stack(patches)
    #     assert patches.get_shape().dims == [500, 128, 128, 1]
    #     return patches
    if CT_type_TFRecord[-14:-9] == 'valdn':
        return tf.data.TFRecordDataset(CT_type_TFRecord).map(parse_fn)

    #--use this for patch-based learning--
    #return tf.data.TFRecordDataset(CT_type_TFRecord).map(parse_fn).map(create_patches).apply(tf.data.experimental.unbatch())

    #--use this for full-image learning--
    return tf.data.TFRecordDataset(CT_type_TFRecord).map(parse_fn)

def generateTFRecords(filenames, CT_type):
    """
    takes input binary CT image files and writes them out as TFRecords
    """

    def convert_img(file):
        with tf.io.gfile.GFile(file, 'rb') as fid:
            image_data= fid.read()
        filename= os.path.basename(file)
        example= tf.train.Example(features= tf.train.Features(feature= {
                  'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),
                  'rows': tf.train.Feature(int64_list = tf.train.Int64List(value = [512])),
                  'cols': tf.train.Feature(int64_list = tf.train.Int64List(value = [512])),
                  'channels': tf.train.Feature(int64_list = tf.train.Int64List(value = [1])),
                  'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data]))
        }))
        return example
    with tf.io.TFRecordWriter('{}.tfrecord'.format(CT_type)) as writer:
        for f in filenames:
            example = convert_img(f)
            writer.write(example.SerializeToString())

def get_loss(y_pred, y_true, batch_size):
        """
        Returns the l2 loss
        """
        loss = (1.0 / batch_size) * tf.nn.l2_loss(y_true - y_pred)
        return loss

# ------------------------ denoiser object -----------------------------------#
class denoiser(object):
    def __init__(self, sess, batch_size, input_c_dim=1):
        
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.batch_size = batch_size
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim])
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.X = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim])
        self.Y = unet(self.X, is_training=self.is_training)
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
               self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    # --------------------------------train()---------------------------------*    
    def train(self, ndct_train, ldct_train, ndct_eval_data, ldct_eval_data, lr, ckpt_dir, num_epochs, sample_dir, buffer_size):
        #generate training data from ldct and ndct datasets
        if not os.path.exists('ldct_train.tfrecord'):
            generateTFRecords(ldct_train, 'ldct_train')
        if not os.path.exists('ndct_train.tfrecord'):
            generateTFRecords(ndct_train, 'ndct_train')

        #create datasets using training data
        ldct_train_dataset = create_dataset('ldct_train.tfrecord')
        ndct_train_dataset = create_dataset('ndct_train.tfrecord')
        train_dataset = tf.data.Dataset.zip((ldct_train_dataset, ndct_train_dataset)).repeat(num_epochs).shuffle(buffer_size).batch(self.batch_size)
        #train_dataset = tf.data.Dataset.zip((ldct_train_dataset, ndct_train_dataset)).repeat(num_epochs).batch(self.batch_size)
                                   
        # summary_op = tf.summary.merge_all()
        iterator = train_dataset.make_initializable_iterator()

        next_element = iterator.get_next()
        self.sess.run(iterator.initializer)
        epoch = 0
        
        # load existing model if it exists
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            epoch = global_step
            print("[*] Model restore success!")
        else:
            epoch = 0
            print("[*] Not find pretrained model!")
        while True:
            try:
                # we're training the network on the training dataset
                num_batches = len(ndct_train) / self.batch_size # number of images used 3600

                #-----------------------------------------
                # learning rate is cut in half every 10 epoch
                #-----------------------------------------
                learning_rate = lr/(2**(epoch/10))


                for i in range(0, int(num_batches)):
                    print("batch: {}/{}".format(i,num_batches))
                    ldct_img, ndct_img = self.sess.run(next_element)

                    #-------------------------                                                                                  
                    # self.Y:  denoised image                                                                                   
                    # self.Y_: clean image                                                                                      
                    # self.X:  low dose image                                                                                   
                    #-------------------------  
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_: ndct_img, self.X: ldct_img, 
                                                             self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))
                
                    # -------------------------------- 
                    # augment image and add to training 
                    # ---------------------------------
                
                    # mode 1: flipud
                    ldct_img_1 = data_augmentation(ldct_img, 1)
                    ndct_img_1 = data_augmentation(ndct_img, 1)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_ : ndct_img_1, self.X : ldct_img_1, 
                                                            self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))

                    # mode 2: rotate 90
                    ldct_img_2 = data_augmentation(ldct_img, 2)
                    ndct_img_2 = data_augmentation(ndct_img, 2)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_ : ndct_img_2, self.X : ldct_img_2,
                                                            self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))
                    # mode 3: rotate 90 and flipud
                    ldct_img_3 = data_augmentation(ldct_img, 3)
                    ndct_img_3 = data_augmentation(ndct_img, 3)
                    _, step_loss = self.sess.run(
                               [self.train_op, self.loss],
                        feed_dict={self.Y_ : ndct_img_3, self.X : ldct_img_3, 
                                   self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))
                    
                    # mode 4: rotate 180                                                                                                                        
                    ldct_img_4 = data_augmentation(ldct_img, 4)
                    ndct_img_4 = data_augmentation(ndct_img, 4)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_ : ndct_img_4, self.X : ldct_img_4,
                                                            self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))
                    # mode 5: rotate 180  and flipud                                                                                                                   
                    ldct_img_5 = data_augmentation(ldct_img, 5)
                    ndct_img_5 = data_augmentation(ndct_img, 5)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_ : ndct_img_5, self.X : ldct_img_5,
                                                            self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))

                    # mode 6: rotate 270                                                                                                                         
                    ldct_img_6 = data_augmentation(ldct_img, 6)
                    ndct_img_6 = data_augmentation(ndct_img, 6)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_ : ndct_img_6, self.X : ldct_img_6,
                                                            self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))

                    # mode 7: rotate 270 and flipud                                                                                                              
                    ldct_img_7 = data_augmentation(ldct_img, 7)
                    ndct_img_7 = data_augmentation(ndct_img, 7)
                    #print("7", ldct_img_7[0].shape)
                    #print("7", ndct_img_7[0].shape)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_ : ndct_img_7, self.X : ldct_img_7,
                                                            self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))
                    #----------- end data augmentation----------------------
                

                     # mode 8: rotate 270 and flipud                                                                                
                    ldct_img_8, ndct_img_8 = random_crop(ldct_img, ndct_img)
                    #print("8", ldct_img_8[0].shape)
                    #print("8", ndct_img_8[0].shape)
                    _, step_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.Y_ : ndct_img_8, self.X : ldct_img_8,
                                                            self.lr: learning_rate, self.is_training : True})
                    print("Epoch: {}/{}\tLoss: {}\n".format(epoch, num_epochs, step_loss / self.batch_size))
                    
                    # want to see a cropped images
                    scalef = max(np.amax(ndct_img[0]), np.amax(ndct_img[0]), np.amax(ldct_img[0]))
                    #denoised_img = np.clip(255 * denoised_img/scalef, 0, 255).astype('uint8')
                    ndct_img_orig = np.clip(255 * ndct_img[0]/scalef, 0, 255).astype('uint8')
                    ldct_img_orig = np.clip(255*ldct_img[0]/scalef, 0, 255).astype('uint8')
                    #save_images(os.path.join(sample_dir, 'test_orig%d_%d.png' % (i,epoch)),ldct_img_orig, ldct_img_orig, ndct_img_orig)

                    ndct_img_8[0] = np.clip(255 * ndct_img_8[0]/scalef, 0, 255).astype('uint8')
                    ldct_img_8[0] = np.clip(255 * ldct_img_8[0]/scalef, 0, 255).astype('uint8')
                    #save_images(os.path.join(sample_dir, 'test_cropped%d_%d.png' % (i,epoch)), ldct_img_8[0], ldct_img_8[0], ndct_img_8[0])


                # make summary
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('lr', self.lr)
                img = tf.summary.image('denoised image', self.Y, max_outputs=1)
                writer = tf.summary.FileWriter('./logs', self.sess.graph)
                merged = tf.summary.merge_all()
                summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)

                self.evaluate(ndct_eval_data, ldct_eval_data, sample_dir, epoch, summary_merged=summary_psnr,
                     summary_writer=writer, summ_img=img)
                self.save(epoch, ckpt_dir)
                epoch = epoch + 1
                print("epoch", epoch)
            # iterator has no more data to iterate over
            except tf.errors.OutOfRangeError:
                print("Training complete")
                break
    
    # --------------------------evaluate() ---------------------#
    def evaluate(self, valdn_ndct, valdn_ldct, sample_dir, epoch, summary_merged, summary_writer, summ_img):
        print("[*] Evaluating...")

        if not os.path.exists('ldct_test.tfrecord'):
            generateTFRecords(valdn_ldct, 'ldct_test')
        if not os.path.exists('ndct_test.tfrecord'):
            generateTFRecords(valdn_ndct, 'ndct_test')

        # create dataset
        ldct_valdn_dataset= create_dataset('ldct_test.tfrecord')
        ndct_valdn_dataset= create_dataset('ndct_test.tfrecord')
        valdn_dataset = tf.data.Dataset.zip((ldct_valdn_dataset, ndct_valdn_dataset)).batch(1)
        
        iterator = valdn_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        self.sess.run(iterator.initializer)
        psnr_sum = 0
        
        for i in range(0,len(valdn_ldct)):
            ldct_img, ndct_img = self.sess.run(next_element)
        
            #-------------------------
            # self.Y:  denoised image
            # self.Y_: clean image
            # self.X:  low dose image
            #-------------------------            
            denoised_img,psnr_summary, temp_img  = self.sess.run([self.Y, summary_merged, summ_img],
                            feed_dict = {self.X: ldct_img,
                                         self.Y_:ndct_img,
                                         self.is_training: False})
            
            summary_writer.add_summary(psnr_summary, epoch)
            summary_writer.add_summary(temp_img, epoch)


            psnr = cal_psnr(ndct_img, denoised_img)
            psnr_sum += psnr
            print("Test img {}/{} PSNR: {}\n".format(i, len(valdn_ldct), psnr))

            #------------------------------------------------------------------                                                                                                        
            # we are outputing 3 images in the output_samples folder                                                                                                                   
            # first: clean image, second: denoised image, third: low dose image                                                                                                        
            # only saved the first image                                                                                                                                               
            #------------------------------------------------------------------ 
            scalef = max(np.amax(denoised_img), np.amax(ndct_img), np.amax(ldct_img))
            denoised_img = np.clip(255 * denoised_img/scalef, 0, 255).astype('uint8')
            ndct_img = np.clip(255 * ndct_img/scalef, 0, 255).astype('uint8')
            ldct_img = np.clip(255*ldct_img/scalef, 0, 255).astype('uint8')
            # change i to any image number that we want to save
            # we can't save all images because we have 3600 images
            # 227 has psnr 31
            if i == 227:
                save_images(os.path.join(sample_dir, 'test%d_%d.png' % (i,epoch)),ldct_img, denoised_img, ndct_img)
        print("Avg PSNR: {}".format(psnr_sum / len(valdn_ndct)))
    
    # --------------------------test() ---------------------#
    # 1. Calculates average psnr for all test images
    # 2: Saves the denoised image as a float numpy array
    #    print(denoised_img.shape) 
    #    (1, 1, 512, 512, 1)
    #-------------------------------------------------------#
    def test(self, test_ldct, test_ndct, ckpt_dir, save_dir):
        assert len(test_ldct) != 0, 'No testing data!'                                                                                                      
        assert len(test_ndct) != 0, 'No testing data!'  
        # pdb.set_trace() 
        print("[*] Evaluating...")

        if not os.path.exists('ldct_test.tfrecord'):
            generateTFRecords(test_ldct, 'ldct_test')
        if not os.path.exists('ndct_test.tfrecord'):
            generateTFRecords(test_ndct, 'ndct_test')

        # create dataset                                                                                                                                         
        ldct_test_dataset= create_dataset('ldct_test.tfrecord')
        ndct_test_dataset= create_dataset('ndct_test.tfrecord')
        test_dataset = tf.data.Dataset.zip((ldct_test_dataset, ndct_test_dataset)).batch(1)
        
        iterator = test_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        self.sess.run(iterator.initializer)

        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        print("[*] start testing...")
        
        rawfiles = [open(os.path.join(save_dir, "test_{num:08d}.flt".format(num=idx)), 'wb') for idx in range (len(test_ndct))]
        
        for idx in range(0,len(test_ldct)):
            ldct_img, ndct_img = self.sess.run(next_element)
            denoised_img = self.sess.run([self.Y],
                                         feed_dict = {self.X: ldct_img,
                                                    self.Y_: ndct_img,
                                                    self.is_training: False})

            #--------------------------------                                                                                                                                          
            #save image to a test folder                                                                                                                                               
            #--------------------------------  
            denoised_img = np.asarray(denoised_img)
            denoised_img.tofile(rawfiles[idx])
            
            # calculate PSNR
            psnr = cal_psnr(ndct_img, denoised_img)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
        avg_psnr = psnr_sum / len(test_ndct)
        print("--- Average PSNR %.2f ---" % avg_psnr)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int( full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def save(self, iter_num, ckpt_dir, model_name='UNet-tensorflow'):
        saver = tf.compat.v1.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)
