from pathlib import Path

import tensorflow.compat.v1 as tf
#import tensorflow as tf

from utkg.kbs.kg import GraphKB, GraphDB
from utkg.models.utkg import UTKG

home = str(Path.home())
flags = tf.app.flags
FLAGS = flags.FLAGS

# Modelling params
flags.DEFINE_integer("thid_num",10,"number of hidden unit in time embedding")
flags.DEFINE_integer("phid_num",50,"number of hidden unit in predicate embedding")
flags.DEFINE_integer("vhid_num",50,"number of hidden unit in object embedding")

flags.DEFINE_integer("npred_hid",50,"number of hidden unit neural predicate")


# Learning params
flags.DEFINE_integer("iter",100,"max iteration")
flags.DEFINE_float("lr",0.001,"learning rate")
flags.DEFINE_integer("nneg_per_var",1,"number of negative samples generated from changing values of a variable")
flags.DEFINE_integer("bsize",32,"mini-batch size")
flags.DEFINE_string("optimizer","adam","training alg")


if __name__=="__main__":
    with  tf.Graph().as_default():
        kb = GraphKB("../../data/footballdb/")
        dataset = GraphDB(kb)
        model = UTKG(kb)

        print("[UTKG] Complete constructing models. Start trainning ...")

        
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        # Train neural predicate "wasBornOn" # make it to CNLP in the next version
        #model.train_npred("wasBornOn",session)
        # Train neural predicate "playFor"
        #model.train_npred("playFor",dataset) 
        # Run the model
        model.train(session)
        
