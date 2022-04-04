"""
Neural Net Probabilistic/Regression Predicate
@SNT
"""
import numpy as np
from chizson.npredicates.neural_predicate import NeuralPred
from utkg.utils.negsampl import *

import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS

                      
def minibatch(data,bsize):
    """
    extract a minibatch of facts and create positive and negative sample
    """
    inds = np.random.randint(data.shape[0],size=bsize)
    mb_x_pos = data[inds,:]
    mb_x_neg = negative_sampling(inds,data)

    mb_y_pos = np.ones((mb_x_pos.shape[0]))
    mb_y_neg = np.zeros((mb_x_neg.shape[0]))
    
    mb_x = np.append(mb_x_pos,mb_x_neg,axis=0)
    mb_y = np.append(mb_y_pos,mb_y_neg,axis=0)
    
    return mb_x,mb_y

class NeuralRegPred(NeuralPred):
    """
    Neural probabilistic/regression as a predicate. A neural net N present the truth value as a probablistic/regression function.
    """
    def __init__(self,kb,embedders=None,name="neural_prob_pred"):
        if name not in kb.predicates:
            raise ValueError("Predicate name does not exist!")
        
        self.var_classes  = kb.predicates[name]["var_classes"]
        self.var_dims = kb.db.var_dims
        self.pred_inx = (list(kb.predicates.keys())).index(name) 
        self.embedders    = embedders
        self.mb_size      = FLAGS.bsize
        self.name = name

        # set up place holders
        self.place_holders()

        # Embedding
        self.embs = self.embedding()
        # link to truth value
        self.truth = self.nn()
        
    def place_holders(self):
        self.x = {}
        for i in range(len(self.var_classes)):
            var_class =self.var_classes[i]
            shape = [None] + self.var_dims[var_class]
            if var_class[0]=="u" or var_class[0]=="t":
                self.x[i] = tf.placeholder(tf.int32, shape=(None,1))
            else:
                self.x[i] = tf.placeholder(tf.float32, shape=shape)


    def embedding(self):
        print("neural pred " + self.name)
        embs = {}
        for var_name in self.var_classes:
            if self.embedders is not None and self.embedders[var_name] is not None:
                if var_name[0] == "t":
                    temb = self.embedders[var_name].embed(self.x[var_name])
                    pemb = self.embedders["predicate"].embed([self.pred_inx])
                    fc = tf.tensordot(temb,pemb,axes=0) # CHECK IF THAT DOES OUTER PRODUCT PROPERLY
                    fc = tf.layers.flatten(fc)
                else:
                    fc = self.embedders[var_name].embed(self.x[var_name])
            else:
                fc = self.x[var_name]

            embs[var_name] = tf.cast(fc,tf.float32)
        return embs
    
    def nn(self):
        """
        Infer truth value from inputs
        """
        fc_all = None
        for var_name in self.embs:
            print(var_name)
            fc = self.embs[var_name]
            print(fc.get_shape())
            if fc_all is None:
                fc_all = fc
            else:
                fc_all = tf.concat([fc_all,fc],axis=1)

        scope = "npred_"+self.name
        with tf.variable_scope(scope) as sv:
            truth_val = tf.layers.dense(fc_all,1,activation="sigmoid")
            
            # checkpoint save model
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
            self.saver = tf.train.Saver(vars)
            self.ckpt_name = self.name+"_vars.ckpt"

        return truth_val

    def train(self,session,data):
        if FLAGS.optimizer=="adam":
            self.optimizer = tf.train.AdamOptimizer
        elif FLAGS.optimizer=="rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer

        # truth label
        self.y = tf.placeholder(tf.float32,shape=(None,1))
        # loss & optimization
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y,self.truth))

        self.op = self.optimizer(learning_rate=FLAGS.lr).minimize(self.loss)

        
        iter = 0
        while True:
            iter+=1
            mb_x,mb_y = minibatch(data,bsize=FLAGS.bsize)
            print(mb_x.shape)
            print(mb_y.shape)
            input("")
            dict = {}
            for var_name in self.var_classes: # checking
                dict[self.x[var_name]] = mb_x[var_name]
                
            dict[self.y] = mb_y

            _,loss = session.run([self.op,self.loss],dict)
            print("[iter %d] loss=%f"%(iter,loss))

            if iter>=100000:
                break
         
    def save(self,session,fpath):
        self.saver.save(session,os.path.join(fpath,self.ckpt_name))

    def load(self,session,fpath):
        self.saver.store(session,os.path.join(fpath,self.ckpt_name))
                     
class NTNPred(NeuralRegPred):
    def __init__(self,var_classes,all_var_dims,name="ntn_pred"):
        print("TODO")

    def save(self,session,fpath):
        print("TODO")

   
      
