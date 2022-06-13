#20220517 zhanghan
import tensorflow as tf
import numpy as np

from .cross_attn import cross_win_attn
from .intra_attn import intra_win_attn

class cross_intra_attn(tf.keras.Model):

    def __init__(self,dim,window_size,name=None,):
        super().__init__(name=name)


        self.cross_attn = cross_win_attn(dim, (window_size, window_size))
        self.intra_attn = intra_win_attn(dim, (window_size, window_size))


    # def build(self, input_shape):
    #     super().build(input_shape)

    def call(self, x, batchsize=1):
        #print('zhanghan 0' ,x.shape)  #(121, 144, 192)
        y1 = self.intra_attn(x, batchsize) + x
        y2 = self.cross_attn(y1, batchsize) + y1
        return y2

