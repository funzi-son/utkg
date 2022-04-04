"""
Time embedding
@Son N. Tran
"""
import numpy as np
import tensorflow.compat.v1 as tf

class YearEmb():
    def __init__(self,dim=100,name=None):
        self.dim = dim
        self.name = name
    def embed(self,x):
        """
        embedding year
        xyzt
        """
        scope = "year_emb"
        if self.name is not None:
            scope = self.name+"_"+scope
         
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as vs:
            t1000 = tf.slice(x,[0,0],[-1,1])
            t100  = tf.slice(x,[0,1],[-1,1])
            t10   = tf.slice(x,[0,2],[-1,1])
            t1    = tf.slice(x,[0,3],[-1,1])
        
            # convert to one-hot vectors
            mtx = tf.eye(10)

            t1000 = tf.nn.embedding_lookup(mtx,t1000)
            t100  = tf.nn.embedding_lookup(mtx,t100)
            t10   = tf.nn.embedding_lookup(mtx,t10)
            t1    = tf.nn.embedding_lookup(mtx,t1)

            t1000 = tf.reshape(t1000,[-1,10])
            t100 = tf.reshape(t100,[-1,10])
            t10 = tf.reshape(t10,[-1,10])
            t1 = tf.reshape(t1,[-1,10])
            
            # feed to an 1-layer LSTM with n_hidden units.
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.dim)
        
            # generate inference
            outputs, _ = tf.nn.static_rnn(rnn_cell, (t1000,t100,t10,t1), dtype=tf.float32)
        
            print("[Modelling] Time embedding is complete")
        return outputs[-1]
