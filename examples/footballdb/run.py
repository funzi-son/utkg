import tensorflow as tf

from utkg.kbs.kg import GraphKB
from utkg.model.utkg import UTKG

home = str(Path.home())
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("iter",100,"max iteration")
flags.DEFINE_float("lr",0.001,"discriminative learning rate")

flags.DEFINE_integer("bsize",2,"mini-batch size")
flags.DEFINE_string("optimizer","adam","training alg")
        
if __name__=="__main__":
    kb = GraphKB("../../../../data/footballdb/")
    dataset = GraphDB("../../../../data/footballdb/")
    model = UTKG(kb)
    # Train neural predicate "playFor"
    model.train_npred("playFor",dataset) # make it to CNLP in the next version
    # Train neural predicate "wasBornOn"
    #model.train_npred("wasBornOn")

    # Run the model
    #run(model)
