"""
Predicate embedding
@Son N. Tran
"""

import tensorflow.compat.v1 as tf


class PredEmb():
    def __init__(self,name,vocab_size,dim=500):
        self.name=name
        self.vocab_size = vocab_size
        self.dim = dim
        self.onehot_mtx = tf.eye(vocab_size)
        with tf.variable_scope(name) as vs:
            self.emb_mtx = tf.get_variable("weight",shape=[vocab_size,dim])
        
    def embed(self,x):
        onehot = tf.nn.embedding_lookup(self.onehot_mtx,x)
        emb    = tf.matmul(onehot,self.emb_mtx)
        emb    = tf.reshape(emb,[self.dim])
        print("[Modelling] Predicate embedding is complete")
        return emb

    def save(self):
        print("TODO")
        
    def load(self):
        print("TODO")
        
