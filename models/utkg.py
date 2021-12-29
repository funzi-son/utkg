"""
Neural-symbolic Uncertain Temporal Knowledge Graph
@SNT
"""

from npredicates.reg_net_pred import NeuralRegPred

class UTKG(object):
    def __init__(self,kb,var_dims=None,reasoner="lbm",npred_types=None):
        self.reasoner = reasoner

    def build_model(self):
        # construct neural predicates
        self.preds  = {}

        print("Build model")

        # construct formulas



    def infer(self,session,bquery):
        # find formulas
        print(bquery)

        # 
        
        
    def train_npred(self,pred_name,session):
        pred_data = self.kb.extract(pred_name)# TODO
        self.npreds[pred_name].train(session,pred_data)
        
