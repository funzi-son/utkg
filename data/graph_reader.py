import numpy as np
import os

class GReader():
    def __init__(self,ddir):
        self.ddir = ddir

    
    def load_data(self,partition="train"):
        if partition=="train":
            fname = "train.txt"
        elif partition=="validation":
            fname = "valid.txt"
        elif partition=="test":
            fname = "test.txt"
        else:
            raise ValueError("Data name is not specified!!!")

        dfile = os.path.join(self.ddir,fname)
        
        f = open(dfile,"r")
        data = np.empty((0,5))
        while True:
            line = f.readline()
            if line is None or len(line)<=0:
                break
            
            els = line.strip().split("\t")
            d =[[int(els[0]), int(els[1]), int(els[2]), int(els[3]), int(els[4])]]
            data = np.append(data,d,axis=0)
        
        print(data.shape)
        


       
if __name__=="__main__": 
    greader = GReader("../baselines/RE-Net/data/YAGO")

    greader.load_data(partition="train")
