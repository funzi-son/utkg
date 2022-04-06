"""
Neural-symbolic Uncertain Temporal Knowledge Graph
@SNT
"""

from chizson.npredicates.truth_neural_predicates import TruthNNPred
from chizson.nembedders.var_embedders import VarEmb
from chizson.nembedders.time_embedders import YearEmb
from chizson.nembedders.pred_embedders import PredEmb
from chizson.sampling.negsampl import *

from utkg.fpredicates.funcs import *
from utkg.utils.fuzzy_norms import *


import tensorflow.compat.v1 as tf
FLAGS = tf.flags.FLAGS

tf.reset_default_graph()

""" TODO
def sample_with_shared_vars(kb,p1name,p2name,p1shared,p2shared):
    if p1name in kb.db.facts:
        d1 = kb.db.facts[p1name]
    else:
        d1  = kb.db.facts[p1name].generate()


    if p2name in kb.db.facts:
        d2 = kb.db.facts[p2name]
    else:
        d2  = kb.db.facts[p2name].generate()
            
    d2 = kb.db.facts[p2name]

    # sample some of d1[:,p1shared]  that match d2[:,p2shared]
    sample_inds1 = np.random.randint(d1.shape[0],size=1)
    d1shared     = d1[sample_inds,p1shared]
    match        = np.equal(d1shared,d2[:,p2shared])
    for i in range(match.shape[0]):
        m = match[i]
        if m:
            print(d1shared)
            print(d2shared[i,:])
            input("")
    
    input("sample from share var")
    
def minibatch_rule(f,kb):
    mb = {}

    for i in range(len(f)-2):
        # Get shared variables between two preds
        curr_l = "l" + str(i)
        curr_p = f[curr_l]
        
        next_l = "l" + str(i+1)
        next_p = f[next_l]
        
        svars = list(set(curr_p[1]) & set(next_p[1]))        
        # Sample the assignments for the two preds
        p1shared = [curr_p[1].index(x) for x in svars]
        p2shared = [curr_p[1].index(x) for x in svars]
        
        sample_with_shared_vars(kb,curr_p[0],next_p[0],p1shared,p2shared)
        print(svars)
"""

def minibatch_rule(f,kb):
    mb = {}
    var_assignment = {}
    # 1. negative sampling from head
    lh = f["head"]
    sign,name= extract(lh[0])
    data = kb.db.facts[name]
    neg_candidates = negative_sampling_rand(data,1)
    mb[name] = {}
    for i in range(len(lh[1])):
        v =lh[1][i]
        var_assignment[v]  = neg_candidates[:,i]
        mb[name][i] = var_assignment[v][np.newaxis,:]
        
    print(var_assignment)
    # 2. sample other assignments for other variables in the formula
    for i in range(len(f)-1): 
        p = f["l"+str(i)]
        sign,pname = extract(p[0])
        mb[pname] = {}
        cll = []

        assigned_inds = []
        unassigned = {}
        for j in range(len(p[1])):
            var = p[1][j]
            print(p[1])
            if var in var_assignment:
                cll.append(var_assignment[var][0])
                assigned_inds.append(j)
            else:
                unassigned[j] = None
        
        # samples the tuple with assigned variables        
        if type(kb.db.facts[pname]).__module__ == np.__name__:
            sinds  = np.where(kb.db.facts[pname][:,assigned_inds]==cll)[0]
            fsampl = kb.db.facts[pname][sinds,:]            
        else:
            fsampl = kb.db.facts[pname].sample(assigned_inds,cll)
        
    
        # extract and assign values to unassigned variables
        for j in unassigned:
            var = p[1][j]
            var_assignment[var] = fsampl[:,j]
        # add to minibatch
        for j in range(len(p[1])):
            var = p[1][j]
            mb[pname][j] = var_assignment[var][np.newaxis,:]
        print(mb[pname])
    input("")
    return mb

def extract(lit):
    sign = 1
    
    if "NOT" in lit:
        sign = 0
    name = lit.replace("NOT","").strip()
    return sign,name

funcs  = {"before":Before}

class UTKG(object):
    def __init__(self,kb,reasoner="lbm",npred_types=None):
        self.reasoner = reasoner
        self.kb = kb
        self.funcs = funcs
        
        # Setting up embedders
        self.embedders = {}
        for c in self.kb.classes:
            #print(c)
            if c[0]=="t": # time variable
                self.embedders[c] = YearEmb(dim=FLAGS.thid_num)
            else:
                self.embedders[c] = VarEmb(c,self.kb.db.var_dims[c][0],dim=FLAGS.vhid_num)

        # define embedders
        self.embedders["predicate"] = PredEmb("pred_emb",len(self.kb.predicates),dim=FLAGS.phid_num)

        # define neural predicates
        self.build_model()
    
        
    def build_model(self):
        # Step 1: construct neural predicate
        self.build_neural_predicates()

        # Step 2: plausibility score
        self.plausibility_scores()
        input("Completed building model")

        # Step 3: plausibility loss and ops
        self.plausibility_ops()
        
    def plausibility_scores(self):
        """
        Compute pausibility score of all formulas
        """
        print("plausibility")
        self.plausibilities = {}
        for rname in self.kb.rules:
            formulas =  self.kb.rules[rname]
            for f in formulas:
                clause = [] # conjunctive clause
                for e in f:
                    sign,pname = extract(f[e][0])
                    print(sign)
                    print(pname)
                    if e =="head":
                        p = literal(sign,self.npreds[pname].truth)
                    else:
                        clause.append([sign,self.npreds[pname].truth])
                    
                score = tf.math.square(p-prod_norm_conj(clause))
                if rname in self.plausibilities:
                    self.plausibilities[rname].append(score)
                else:
                    self.plausibilities[rname] = [score]
        
        
    def build_neural_predicates(self):
        # construct neural predicates
        self.npreds  = {}

        for pred_name in self.kb.predicates:
            if pred_name in self.kb.db.facts:
                self.npreds[pred_name] = TruthNNPred(self.kb,
                                                       embedders=self.embedders,
                                                       name=pred_name)
            else:
                self.npreds[pred_name] = self.funcs[pred_name](self.kb,name=pred_name)
                self.kb.db.facts[pred_name] = self.npreds[pred_name]
        print("[UTKG] Neural Predicates built")
        # construct formulas
                    
        
    def infer(self,session,bquery):
        # find formulas
        print(bquery)

                
    def train_npred(self,pred_name,session):
        """
        Pre-train each neural predicate from facts 
        """
        pred_data = self.kb.extract(pred_name)
        
        self.npreds[pred_name].train(session,pred_data)


    def plausibility_ops(self):
        if FLAGS.optimizer=="adam":
            self.optimizer = tf.train.AdamOptimizer
        elif FLAGS.optimizer=="rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer

        c = 0
        self.rloss = 0
        for rname in self.plausibilities:
            for score in self.plausibilities[rname]:
                self.rloss += score
                c+=1
        self.rloss = self.rloss/c
        self.rop   = self.optimizer(learning_rate=FLAGS.lr).minimize(self.rloss)
        
    def train(self,session):
        """
        Train the whole network 
        """
        print("[UTKG] Start training ...")
        iter = 0
        while True:
            loss1=0
            """
            for rname in self.kb.rules:
                formulas =  self.kb.rules[rname]
                
                for f in formulas:                    
                    # Update rule constrants
                    print("Get a minibatch from rule")
                    mb = minibatch_rule(f,self.kb)
                    print(mb)
                    rdict = {}
                    for lname  in f:
                        p = f[lname]
                        sign,pred_name = extract(p[0])
                        print(pred_name)
                        print(p[1])
                        for i in range(len(p[1])):
                            print("****")
                            #print(self.npreds[pred_name].x[i].get_shape())
                            #print(mb[pred_name][i].shape)
                            var_class = self.kb.predicates[pred_name]["var_classes"][i][0]
                            vals = mb[pred_name][i]
                            if var_class=="t":# year format
                                 t1000,vals  = np.divmod(vals,1000)
                                 t100,vals = np.divmod(vals,100)
                                 t10,t1  = np.divmod(vals,10)
                                 vals = np.concatenate((t1000,t100,t10,t1),axis=1)
                                 #print(vals.shape)
                                                
                            rdict[self.npreds[pred_name].x[i]]  = vals
                            #print(vals)
                            #input("")
                    print(rdict)
                    input("")
                    _, loss1 = session.run([self.rop,self.rloss],rdict)
            """     
            # Update predicates from facts
            loss2 = 0
            
            for pred_name in self.npreds:
                pred_loss = self.npreds[pred_name].update(session,self.kb.db.facts[pred_name])
                loss2+=pred_loss
            
            # Plot learning
            print("[UTKG] iter %d loss1=%.5f loss2=%.5f",(iter,loss1,loss2))
                    
