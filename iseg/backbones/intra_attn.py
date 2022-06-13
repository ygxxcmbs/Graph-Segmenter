#20220517 zhanghan
import tensorflow as tf
import numpy as np

class intra_win_attn(tf.keras.Model):

    def __init__(self,dim,window_size,name=None,):
        super().__init__(name=name)
        self.outdim = int(dim / 16)
        self.h = int(window_size[0])

        self.conv1=tf.keras.layers.Conv2D(self.outdim, kernel_size=(1,1), strides=(1,1))
        self.W = tf.Variable(tf.initializers.GlorotUniform()([self.outdim,self.outdim]))
        self.conv2 = tf.keras.layers.Conv2D(dim, kernel_size=(1, 1), strides=(1, 1),
                                            kernel_initializer=tf.zeros_initializer(),
                                            bias_initializer=tf.zeros_initializer()
                                            )
        self.softmax=tf.keras.layers.Softmax()

        #zhanghan
        self.permute1=tf.keras.layers.Permute([3,1,2])
        self.permute2=tf.keras.layers.Permute([2,1])
        self.permute3 = tf.keras.layers.Permute([2,3,1])

    # def build(self, input_shape):
    #     super().build(input_shape)

    def call(self, x, batchsize=1):
        #B_, L, C = x.shape  # 1568, 49, 192              # 121, 144, 192
        B_, L, C = tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2]

        # x=self.conv1( self.permute1(tf.reshape(x,[B_, self.h, self.h, C])) )  # (121, 192, 12, 12)
        x1 = tf.reshape(x, (B_, self.h, self.h, C))
        x1 = self.permute1(x1)

        x1 = tf.transpose(x1, [0, 2, 3, 1])
        x1 = self.conv1(x1)
        x1 = tf.transpose(x1, [0, 3, 1, 2])

        # x_wins=self.permute2(tf.reshape(x,[B_, self.outdim, L]))
        x_wins = self.permute2(tf.reshape(x1, [B_, self.outdim, L]))

        A = self.softmax(tf.matmul(x_wins, self.permute2(x_wins) / np.sqrt(x_wins.shape[2])))
        
        #zhanghan 2022 06 08  Sparse A
        zero_vec=tf.zeros_like(A)
        A=tf.where(A>tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(A,2),1),0)/4,A,zero_vec)
        
        
        x2 = tf.matmul(A, x_wins)
        x2=tf.cast(x2,dtype=tf.float32)
        x2 = tf.matmul(tf.reshape(x2, [B_ * L, -1]), self.W)
        # x=self.conv2(  tf.reshape(self.permute2(tf.reshape(x,[B_, L, self.outdim])),[B_, self.outdim, self.h, self.h]) )
        x2 = tf.reshape(x2, [B_, L, self.outdim])
        x2 = self.permute2(x2)
        x2 = tf.reshape(x2, [B_, self.outdim, self.h, self.h])

        x2 = tf.transpose(x2, [0, 2, 3, 1])
        x2 = self.conv2(x2)
        x2 = tf.transpose(x2, [0, 3, 1, 2])

        x2 = tf.reshape(self.permute3(x2), [B_, L, C])

        return x2
