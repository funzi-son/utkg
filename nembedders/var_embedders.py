"""
Variable embedding
@Son N. Tran
"""

import tensorflow.compat.v1 as tf


class VarEmb():
    def __init__(self,name,vocab_size,dim=500):
        self.name=name
        self.vocab_size = vocab_size
        self.dim = dim
        self.onehot_mtx = tf.eye(vocab_size)
        with tf.variable_scope("var_emb_"+self.name) as vs:
            self.emb_mtx = tf.get_variable("weights",shape=[vocab_size,dim])
        
    def embed(self,x):
        onehot = tf.nn.embedding_lookup(self.onehot_mtx,x)
        onehot = tf.reshape(onehot,[-1,self.vocab_size])
        emb    = tf.matmul(onehot,self.emb_mtx)
        print("[Modelling] Variable embedding for "+self.name+" is complete")
        return emb

    def save(self):
        print("TODO")
        
    def load(self):
        print("TODO")
        
