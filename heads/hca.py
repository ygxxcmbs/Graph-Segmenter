#zhanghan 2022 06 07
# hard class attention

import tensorflow as tf
import numpy as np

class HardClassAttention(tf.keras.Model):
    def __init__(self,in_planes,name=None,):
        super().__init__(name=name)
        hidden_dim = int(in_planes / 2)
        #self.conv1 = nn.Conv2d(in_planes, hidden_dim, kernel_size=1, bias = False)
        self.conv1 = tf.keras.layers.Conv2D(hidden_dim, kernel_size=(1, 1), strides=(1, 1))
        #self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding = 3, bias = False)
        self.conv2 = tf.keras.layers.Conv2D(hidden_dim, kernel_size=(7, 7), strides=(1, 1),padding='SAME')
        #self.relu = nn.GELU()
        #self.relu=tf.nn.gelu()
        #self.conv3 = nn.Conv2d(hidden_dim, in_planes, kernel_size=1, bias = False)
        self.conv3 = tf.keras.layers.Conv2D(in_planes, kernel_size=(1, 1), strides=(1, 1))
        #self.sigmoid = nn.Sigmoid()
        #self.sigmoid=tf.nn.sigmoid()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = tf.nn.gelu(x)
        x = self.conv3(x)
        ct_atten = tf.nn.sigmoid(x)
        return ct_atten
        
