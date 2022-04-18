"""
Truth estimation neural predicates
@Son Tran 
"""
import numpy as np
from utkg.npredicates.npred_generic import NeuralPred
from utkg.sampling.negsampl import *

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

class TruthNNPred(NeuralPred):
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

        self.trainable = True
        # optimizers
        if FLAGS.optimizer=="adam":
            self.optimizer = tf.train.AdamOptimizer
        elif FLAGS.optimizer=="rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer

        # set up place holders
        self.place_holders()
        # truth label
        self.y = tf.placeholder(tf.float32,shape=(None,1))

        # Embedding
        self.embs = self.embedding()
        # link to truth value
        self.truth = self.nn()

        # loss & optimization
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y,self.truth))
        with tf.variable_scope("npred_op", reuse=tf.AUTO_REUSE) as vs:
            self.op = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(self.loss)

    def place_holders(self):
        self.x = {}
        for i in range(len(self.var_classes)):
            var_class =self.var_classes[i]
            shape = [None] + self.var_dims[var_class]
            if var_class[0]=="u":
                self.x[i] = tf.placeholder(tf.int32, shape=(None,1))
            elif var_class[0]=="t":# currently for year
                self.x[i] = tf.placeholder(tf.int32, shape=(None,4))
            else:
                self.x[i] = tf.placeholder(tf.float32, shape=shape)


    def embedding(self):
        print("neural pred " + self.name)
        embs = {}
        for i in range(len(self.var_classes)):
            var_name = self.var_classes[i]
            if self.embedders is not None and self.embedders[var_name] is not None:
                if var_name[0] == "t":
                    temb = self.embedders[var_name].embed(self.x[i])
                    pemb = self.embedders["predicate"].embed([self.pred_inx])
                    fc = tf.tensordot(temb,pemb,axes=0) # CHECK IF THAT DOES OUTER PRODUCT PROPERLY
                    fc = tf.layers.flatten(fc)
                else:
                    fc = self.embedders[var_name].embed(self.x[i])
            else:
                fc = self.x[i]

            embs[var_name] = tf.cast(fc,tf.float32)
        return embs
    
    def nn(self):
        """
        Infer truth value from inputs
        """
        fc_all = None
        for var_name in self.embs:
            #print(var_name)
            fc = self.embs[var_name]
            #print(fc.get_shape())
            if fc_all is None:
                fc_all = fc
            else:
                fc_all = tf.concat([fc_all,fc],axis=1)

        scope = "npred_"+self.name
        with tf.variable_scope(scope) as sv:
            h1  = tf.layers.dense(fc_all,FLAGS.npred_hid,activation="relu")
            truth_val = tf.layers.dense(fc_all,1,activation="sigmoid")
            
            # checkpoint save model
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
            self.saver = tf.train.Saver(vars)
            self.ckpt_name = self.name+"_vars.ckpt"

        return truth_val

    def train(self,session,data):
        # loss & optimization
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y,self.truth))
        self.op = self.optimizer(learning_rate=FLAGS.lr).minimize(self.loss)

        
        iter = 0
        while True:
            iter+=1
            mb_x,mb_y = minibatch(data,bsize=FLAGS.bsize)
            #print(mb_x.shape)
            #print(mb_y.shape)
            #input("")
            dict = {}
            for var_name in self.var_classes: # checking
                dict[self.x[var_name]] = mb_x[var_name]
                
            dict[self.y] = mb_y

            _,loss = session.run([self.op,self.loss],dict)
            print("[iter %d] loss=%f"%(iter,loss))

            if iter>=100000:
                break

    def update(self,session,data):
        #print("Update neural predicates %s"%(self.name))
        
        mb_x,mb_y = minibatch(data,bsize=FLAGS.bsize)
        #print(mb_x)
        #print(mb_y)
        #input("")
        dict = {}
        for i in range(len(self.var_classes)):
            var_class =self.var_classes[i]
            vals = mb_x[:,i][:,np.newaxis] 
            if var_class[0]=="t":# year format                                                                    
                t1000,vals  = np.divmod(vals,1000)
                t100,vals = np.divmod(vals,100)
                t10,t1  = np.divmod(vals,10)
                vals = np.concatenate((t1000,t100,t10,t1),axis=1)

            #print(vals.shape)
            #print(self.x[i].get_shape())
            dict[self.x[i]] = vals
            #input("+++")
        #print(input("+-+0d-9i-98"))
        dict[self.y] = mb_y[:,np.newaxis]

        _,loss = session.run([self.op,self.loss],dict)
        #print("loss=%f"%(loss))

        return loss

    def infer(self,session,query):
        dict = {}
        for i in range(len(self.var_classes)):
            var_class =self.var_classes[i]
            vals = query[:,i][:,np.newaxis] 
            if var_class[0]=="t":# year format                                                                    
                t1000,vals  = np.divmod(vals,1000)
                t100,vals = np.divmod(vals,100)
                t10,t1  = np.divmod(vals,10)
                vals = np.concatenate((t1000,t100,t10,t1),axis=1)

            #print(vals.shape)
            #print(self.x[i].get_shape())
            dict[self.x[i]] = vals
        truth = session.run(self.truth,dict)
        return truth
    
    def save(self,session,fpath):
        self.saver.save(session,os.path.join(fpath,self.ckpt_name))

    def load(self,session,fpath):
        self.saver.store(session,os.path.join(fpath,self.ckpt_name))
                     
class TruthNTNPred(NeuralPred):
    def __init__(self,var_classes,all_var_dims,name="ntn_pred"):
        print("TODO")

    def save(self,session,fpath):
        print("TODO")


class TruthNTNTEmbPred(NeuralPred):
    def __init__(self,var_classes,all_var_dims,name="tnt_time_pred"):
        print("TODO")

    def save(self,sessoin,fpath):
        print("TODO")
        
