import numpy as np
class NeuralPred(object):
    """
    Generic class for neural predicate
    """
    def __init__(self,var_classes,all_var_dims,name="nn_pred"):
        self.var_classes  = var_classes
        self.all_var_dims = all_var_dims
        self.mb_size      = FLAGS.bsize
        self.name = name
        self.trainable = True
        
        if FLAGS.optimizer=="adam":
            self.optimizer = tf.train.AdamOptimizer
        elif FLAGS.optimizer=="rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer

        i = 0    
        self.x = {}
        self.y = {}
        self.z = {}
        self.losses = {}
        self.ops = {}
        self.dims= {}

        with tf.variable_scope("npred_"+name) as scope:
            for var_class in self.var_classes:
                self.dims[i] = all_var_dims[var_class]
                shape = [None] + self.dims[i]
                self.x[i] = tf.placeholder(tf.float32, shape=shape)
                i+=1        

            i = 0
            for var_class in self.var_classes:
                if var_class[0]=="u":
                    self.losses[i],self.ops[i],self.y[i] = self.upart(i)
                i+=1

        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"npred_"+name)
        self.saver = tf.train.Saver(vars)
        self.ckpt_name = self.name+"_vars.ckpt"
       
