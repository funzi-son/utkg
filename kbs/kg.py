"""
Knowledge base from graphs 
@SNT
"""
import os
import numpy as np

from chizson.kbs.firstorder import KB
class GraphKB(KB):
    def __init__(self,kb_folder_path):
        self.kb_dir = kb_folder_path
        cfile = os.path.join(self.kb_dir,"kb.czs")
        super().__init__(cfile)
        self.objects = self.load_objects(os.path.join(self.kb_dir,"entity2id.txt"))
    def load_graph(self,partition="train"):
        if partition=="train":
            fname = "train.txt"
        elif partition=="validation":
            fname = "valid.txt"
        elif partition=="test":
            fname = "test.txt"
        else:
            raise ValueError("Data name is not specified!!!")
        
        dfile = os.path.join(self.kb_dir,fname)

        self.load_graph_file(dfile)

    def load_graph_file(self,rfile):
        """
        Read relation file
        1st collumn: subject
        2nd collumn: relation
        3rd collumn: object
        4th collumn: time/order of time
        5th collumn: certainty/weight (if available)
        """
        f = open(rfile,"r")
        pred_names = list(self.predicates.keys())
        while True:
            line = f.readline()
            if line is None or len(line)<=0:
                break
            
            els = line.strip().split("\t")
            s = int(els[0])
            r = int(els[1])
            o = int(els[2])
            t = int(els[3])

            rel = pred_names[r]
            # update classes
            var_type = self.predicates[rel]["var_classes"][0]
            if s not in self.classes[var_type]:
                self.classes[var_type].append(s)
            
            var_type = self.predicates[rel]["var_classes"][1]
            if o not in self.classes[var_type]:
                self.classes[var_type].append(o)
            # update facts
            if rel not in self.facts:
                self.facts[rel]  = np.array([[s,o,t]])
            else:
                self.facts[rel] = np.append(self.facts[rel],[[s,o,t]],axis=0)
            
        #print(self.facts)
    def load_objects(self,fname):
        f = open(fname,"r")
        objects = {}
        while True:
            line = f.readline()
            if line is None or len(line)<=0:
                break
            strs = line.strip().split("\t")
            obj_name = strs[0].replace("<","").replace(">","")
            obj_id   = int(strs[1])
            objects[obj_id] = obj_name
        return objects

    def print_(self,relation,num=10):
        facts = self.facts[relation]
    
        for i in range(num):
            print(relation+"("+self.objects[facts[i,0]]+","+self.objects[facts[i,1]]+")")
        
if __name__=="__main__": 
    greader = GraphKB("../data/YAGO")
    """
    print(greader.classes)
    print(greader.predicates)
    print(greader.rules)
    print(greader.facts)
    print(greader.head_list)
    print(greader.body_list)
    print(greader.objects)
    """
    greader.load_graph()
    greader.print_("worksAt")
