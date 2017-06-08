import os,sys,shutil
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import math,glob,random,time,csv

class BatchGenerator:
    def __init__(self):
        pass

    def loadFiles(self,fpaths, cols_to_use):
        self.nfiles = len(fpaths)
        self.fpaths = fpaths
        self.orig   = {}
        self.data   = {}
        self.mask   = {}
        self.ndim   = len(cols_to_use)

        for fpath in fpaths:
            d = pd.read_csv(fpath)
            self.data[fpath] = d.as_matrix(columns=cols_to_use)
            self.orig[fpath] = d.as_matrix(columns=cols_to_use)
            self.mask[fpath] = (~d.isnull()).as_matrix(columns=cols_to_use)
        return

    def getBatch(self,nLen,nBatch):
        x   = np.zeros( (nBatch, 1, nLen, self.ndim ), dtype=np.float32)
        m   = np.zeros( (nBatch, 1, nLen, self.ndim ), dtype=np.bool   )
        self.last_iFile = []
        self.last_iIdx  = []
        self.last_nLen  = []
        for i in range(nBatch):
            iFile = np.random.choice(self.fpaths)
            iIdx  = np.random.randint(0, self.data[iFile].shape[0] - nLen - 1)
            self.last_iFile.append(iFile)
            self.last_iIdx.append(iIdx)
            self.last_nLen.append(nLen)
            x[i,0,:,:] = self.data[iFile][iIdx:iIdx+nLen,:]
            m[i,0,:,:] = self.mask[iFile][iIdx:iIdx+nLen,:]
        return x,m

    def updateData(self,data):
        for i in range(self.ndim):
            iFile = self.last_iFile[i]
            iIdx  = self.last_iIdx[i]
            nLen  = self.last_nLen[i]
            m = self.mask[iFile][iIdx:iIdx+nLen,:]
            np.place(self.data[iFile],~m,data[~m])
            #self.data[iFile][iIdx:iIdx+nLen][~m] = data[~m]

class Interpol:
    def __init__(self,args,ndim,nLen):
        self.nBatch = args.nBatch
        self.learnRate = args.learnRate
        self.saveFolder = args.saveFolder
        self.reload = args.reload
        self.isTraining = False
        self.ndim = ndim # number of channels
        self.nLen = nLen # input length
        self.buildModel()

        return

    def _fc_variable(self, weight_shape,name="fc"):
        with tf.variable_scope(name):
            # check weight_shape
            input_channels  = int(weight_shape[0])
            output_channels = int(weight_shape[1])
            weight_shape    = (input_channels, output_channels)

            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer())
            bias   = tf.get_variable("b", [weight_shape[1]], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            input_channels  = int(weight_shape[2])
            output_channels = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [output_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _deconv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            output_channels = int(weight_shape[2])
            input_channels  = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape    , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [input_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, 1, stride, 1], padding = "SAME")

    def _deconv2d(self, x, W, output_shape, stride=1):
        # x           : [nBatch, height, width, in_channels]
        # output_shape: [nBatch, height, width, out_channels]
        return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,1,stride,1], padding = "SAME",data_format="NHWC")

    def leakyReLU(self,x,alpha=0.1):
        return tf.maximum(x*alpha,x) 

    def calcImageSize(self,dh,dw,stride):
        return int(math.ceil(float(dh)/float(stride))),int(math.ceil(float(dw)/float(stride)))

    def calcDeconvSize(self,d,stride):
        return int(math.ceil(float(d)/float(stride)))

    def loadModel(self, model_path=None):
        if model_path: self.saver.restore(self.sess, model_path)

    def buildEncoder(self,x,reuse=False):
        with tf.variable_scope("Encoder") as scope:
            if reuse: scope.reuse_variables()

            h = x

            # conv1
            self.e_conv1_w, self.e_conv1_b = self._conv_variable([1,10,self.ndim,64],name="conv1")
            h = self._conv2d(h,self.e_conv1_w, stride=1) + self.e_conv1_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="eNorm1")
            h = self.leakyReLU(h)

            # conv2
            self.e_conv2_w, self.e_conv2_b = self._conv_variable([1,10,64,128],name="conv2")
            h = self._conv2d(h,self.e_conv2_w, stride=1) + self.e_conv2_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="eNorm2")
            h = self.leakyReLU(h)

            # conv3
            self.e_conv3_w, self.e_conv3_b = self._conv_variable([1,10,128,256],name="conv3")
            h = self._conv2d(h,self.e_conv3_w, stride=1) + self.e_conv3_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="eNorm3")
            h = self.leakyReLU(h)

            ### summary
            if not reuse:
                tf.summary.histogram("e_conv1_w"   ,self.e_conv1_w)
                tf.summary.histogram("e_conv1_b"   ,self.e_conv1_b)
                tf.summary.histogram("e_conv2_w"   ,self.e_conv2_w)
                tf.summary.histogram("e_conv2_b"   ,self.e_conv2_b)
                tf.summary.histogram("e_conv3_w"   ,self.e_conv3_w)
                tf.summary.histogram("e_conv3_b"   ,self.e_conv3_b)

        return h

    def buildGenerator(self,y,reuse=False):
        dim_0 = self.nLen
        dim_1 = self.calcDeconvSize(dim_0, stride=1)
        dim_2 = self.calcDeconvSize(dim_1, stride=1)
        dim_3 = self.calcDeconvSize(dim_2, stride=1)

        with tf.variable_scope("Generator") as scope:
            if reuse: scope.reuse_variables()

            h = y

            # deconv3
            self.g_deconv3_w, self.g_deconv3_b = self._deconv_variable([1,10,256,128],name="deconv3")
            h = self._deconv2d(h,self.g_deconv3_w, output_shape=[self.nBatch,1,dim_2,128], stride=1) + self.g_deconv3_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="gNorm3")
            h = tf.nn.relu(h)

            # deconv2
            self.g_deconv2_w, self.g_deconv2_b = self._deconv_variable([1,10,128,64],name="deconv2")
            h = self._deconv2d(h,self.g_deconv2_w, output_shape=[self.nBatch,1,dim_1,64], stride=1) + self.g_deconv2_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="gNorm2")
            h = tf.nn.relu(h)

            # deconv1
            self.g_deconv1_w, self.g_deconv1_b = self._deconv_variable([1,10,64,self.ndim],name="deconv1")
            h = self._deconv2d(h,self.g_deconv1_w, output_shape=[self.nBatch,1,dim_0,self.ndim], stride=1) + self.g_deconv1_b

            ### summary
            if reuse:
                tf.summary.histogram("g_deconv1_w"   ,self.g_deconv1_w)
                tf.summary.histogram("g_deconv1_b"   ,self.g_deconv1_b)
                tf.summary.histogram("g_deconv2_w"   ,self.g_deconv2_w)
                tf.summary.histogram("g_deconv2_b"   ,self.g_deconv2_b)
                tf.summary.histogram("g_deconv3_w"   ,self.g_deconv3_w)
                tf.summary.histogram("g_deconv3_b"   ,self.g_deconv3_b)

        return h

    def buildModel(self):
        # define variables
        #self.x      = tf.placeholder(tf.float32, [self.nBatch, 1, self.nLen, self.ndim],name="x")
        self.x      = tf.placeholder(tf.float32, [self.nBatch, 1, self.nLen, self.ndim],name="x")
        self.m      = tf.placeholder(tf.bool   , [self.nBatch, 1, self.nLen, self.ndim],name="m")

        self.y = self.buildEncoder  (self.x)
        self.z = self.buildGenerator(self.y)

        # define loss
        self.loss = tf.sqrt(tf.reduce_sum( tf.pow((self.x-self.z),2) * tf.cast(self.m,tf.float32) ) / tf.maximum(tf.reduce_sum(tf.cast(self.m,tf.float32)),1))
        #self.loss = tf.nn.l2_loss(self.x-self.z)

        # define optimizer
        self.optimizer   = tf.train.AdamOptimizer(self.learnRate).minimize(self.loss)

        ### summary
        tf.summary.scalar("loss"      ,self.loss)

        #############################
        # define session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.30))
        self.sess = tf.Session(config=config)

        #############################
        ### saver
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.saveFolder: self.writer = tf.summary.FileWriter(self.saveFolder, self.sess.graph)

        return

    def train(self,bGen):
        initOP = tf.global_variables_initializer()
        self.sess.run(initOP)

        self.loadModel(self.reload)

        step = -1
        start = time.time()

        while True:
            step += 1
            batch_x,batch_m = bGen.getBatch(self.nLen,self.nBatch)
            batch_x[~batch_m] = 0. # dummy

            # update generator
            _,z,loss,summary = self.sess.run([self.optimizer,self.z,self.loss,self.summary],feed_dict={self.x:batch_x,self.m:batch_m})
            bGen.updateData(z)

            if step>0 and step%10==0:
                self.writer.add_summary(summary,step)

            if step%100==0:
                print "%6d: loss=%.4e; time/step = %.2f sec"%(step,loss,time.time()-start)
                start = time.time()
                self.saver.save(self.sess,os.path.join(self.saveFolder,"model.ckpt"),step)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nBatch","-b",dest="nBatch",type=int,default=64)
    parser.add_argument("--nLen"  ,"-n",dest="nLen",type=int,default=64)
    parser.add_argument("--learnRate","-r",dest="learnRate",type=float,default=1e-3)
    parser.add_argument("--saveFolder","-s",dest="saveFolder",type=str,default="models")
    parser.add_argument("--reload","-l",dest="reload",type=str,default=None)

    args = parser.parse_args()

    bGen = BatchGenerator()
    bGen.loadFiles(["data.csv"],cols_to_use=["col1","col2"])
    p = Interpol(args,bGen.ndim,args.nLen)
    p.train(bGen)
