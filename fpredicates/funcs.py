"""
Functional predicates
@SNT
"""
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS
class Before():
    """
    Just apply to years now
    """
    def __init__(self,kb,name="func_pred"):
        if name not in kb.predicates:
            raise ValueError("Predicate name does not exist!")
        
        self.var_classes  = kb.predicates[name]["var_classes"]
        self.var_dims = kb.db.var_dims
        
        self.name = name

        # set up place holders
        self.place_holders()

        # link to truth value
        self.truth = self.func()
        
    def place_holders(self):
        self.x = {}
        if len(self.var_classes)>2:
            raise ValueError("Number of variables cannot largerr than 2")
        for i in range(len(self.var_classes)):
            var_class = self.var_classes[i]
            shape = [None] + self.var_dims[var_class]
            if var_class[0]=="u":
                self.x[i] = tf.placeholder(tf.int32, shape=(None,1))
            elif var_class[0]=="t":
                self.x[i] = tf.placeholder(tf.int32, shape=(None,4))
            else:
               raise ValueError("Variable must be comparable!!!")
               
    def func(self):
        """
        Infer truth value from inputs
        """
        #print("func predicate-===================")
        #print(self.var_classes)
        x0 = tf.reduce_sum(self.x[0]*[1000,100,10,1],axis=1)
        x1 = tf.reduce_sum(self.x[1]*[1000,100,10,1],axis=1)
        #print(x0.get_shape())
        #print(x1.get_shape())
        truth = tf.less(x0,x1)
        
        return tf.cast(truth,tf.float32)

    def sample(self,assigned_inds,cll):
        
        if len(assigned_inds) >=2:
            return None

        if assign_inds[0] ==0:
            # sample a value after cll[0]
            return cll[0]+np.random.randint(10,size=1)+1
        else:
            # sample a value before cll[0]
            return cll[0]-np.random.randint(10,size=1)+1
