"""
Knowledge base from graphs 
@SNT
"""
import os
import numpy as np
from datetime import datetime
import time

from utkg.kbs.firstorder import KB

class GraphDB(object):
    def __init__(self,kb,partition="train",indexing=True):
        """
        indexing=True: re-index the entities, else use the currrent indexing in the data files
        """
        self.kb = kb
        self.facts = {}
        self.indexing = indexing
        self.load_graph(partition)
        self.kb.set_db(self)
        self.var_dims = self.get_var_dims()

    def get_var_dims(self):
        var_dims = {}
        for var_name in self.kb.classes:
            var_dims[var_name] = [len(self.kb.classes[var_name])]
        
        return var_dims
            
    def load_graph(self,partition):
        if partition=="train":
            fname = "train.txt"
        elif partition=="validation":
            fname = "valid.txt"
        elif partition=="test":
            fname = "test.txt"
        else:
            raise ValueError("Data name is not specified!!!")
        
        dfile = os.path.join(self.kb.kb_dir,fname)

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
        pred_names = list(self.kb.predicates.keys())
        while True:
            line = f.readline()
            if line is None or len(line)<=0 or len(line.strip())<=0:
                break

            els = line.strip().split("\t")
            s = int(els[0])
            r = int(els[1])
            if els[2]!= "#":
                o = int(els[2])
            else:
                o=-1

            rel = pred_names[r]
            
            time_format = self.kb.predicates[rel]["var_classes"][-1]
            
            if time_format == "tindex":
                t = int(els[-1])
            else:
                """
                # To timestamp
                t = datetime.strptime(els[-1], time_format.replace("t","%").replace("-","-%"))
                t= time.mktime(t.timetuple())
                """
                t = int(els[-1][:4])# only take year
                
            # update classes (subject)
            var_type = self.kb.predicates[rel]["var_classes"][0]
            sinx = s
            if s not in self.kb.classes[var_type]:
                self.kb.classes[var_type].append(s)
            if self.indexing:
                sinx = self.kb.classes[var_type].index(s)
            
            # update classes (object if applicable)
            var_type = self.kb.predicates[rel]["var_classes"][1]
            oinx = o
            if o!=-1:
                if o not in self.kb.classes[var_type]:
                    self.kb.classes[var_type].append(o)
                    
                if self.indexing:
                    oinx = self.kb.classes[var_type].index(o)
            
                tple = np.array([[sinx,oinx,t]])
            else:
                tple = np.array([[sinx,t]])
            # update facts
            
            if rel not in self.facts:
                self.facts[rel]  = tple
            else:
                self.facts[rel] = np.append(self.facts[rel],tple,axis=0)
        
    def print_(self,relation,num=10):
        facts = self.facts[relation]
        for i in range(num):
            strl = self.kb.objects[facts[i,0]]
            if facts[i,1]!=-1:
                strl+= ","+self.kb.objects[facts[i,1]]
            
            print(relation+"("+strl+ "," + str(datetime.fromtimestamp(facts[i,2])) + ")")

            
class GraphKB(KB):
    def __init__(self,kb_folder_path):
        self.kb_dir = kb_folder_path
        cfile = os.path.join(self.kb_dir,"meta.czs")
        super().__init__(cfile)
        
        self.objects = self.load_objects(os.path.join(self.kb_dir,"entity2id.txt"))
           
    def load_objects(self,fname):
        f = open(fname,"r")
        objects = {}
        while True:
            line = f.readline()
            if line is None or len(line)<=0:
                break
            strs = line.strip().split("\t")
            obj_name = strs[0].replace("<","").replace(">","").replace(",","")
            obj_id   = int(strs[1])
            objects[obj_id] = obj_name
        return objects

    def set_db(self,db):
        """
        reference to the database where facts are kept
        """
        self.db = db
        
    def extract(self,pred_name):
        facts = self.db.facts[pred_name]
        return facts
    
if __name__=="__main__": 
    #greader = GraphKB("../../../data/YAGO")
    kb = GraphKB("../../../data/footballdb/")
    #print(kb.classes)
    #print(kb.predicates)
    print(kb.rules)
    #print(kb.facts)
    #print(kb.head_list)
    #print(kb.body_list)
    db = GraphDB(kb,partition="train")

    db.print_("playFor")
    db.print_("wasBornOn")
    """
    cnames = ["uhuman","uplace","uorganisation","uteam","uprize","ucompany","uparty","uproduct"]
    for i in range(len(cnames)):
        cname1 = cnames[i]
        for n in greader.classes[cname1]:
            for j in range(i+1,len(cnames)):
                cname2 = cnames[j]
                if n in greader.classes[cname2]:
                    print(cname1)
                    print(cname2)
                    print(n)
                    print(greader.objects[n])
    """
