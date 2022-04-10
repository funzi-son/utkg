"""
Knowledge Base 
@SNT
"""
import numpy as np
import re 

def read_pred(str):
    sinx = str.find("(")
    einx = str.find(")")
    
    if einx!=-1:
        content = str[sinx+1:einx].split(",")
    else:
        content = str[sinx+1:].split(",")
    pred_name = str[:sinx]

    return pred_name,content

class KB():
    def __init__(self,kb_file_path):
        self.classes    = {}
        self.predicates = {}
        self.rules      = {}
        self.facts      = {}
        self.head_list  = []
        self.body_list  = []

        self.load_kb(kb_file_path)
        self.rules = self.organise_rules()
    def load_kb(self,filename):
        f = open(filename,"r")
        while True:
            str = f.readline()
            if str is None or len(str)==0:
                break
            
            if ":-" in str:
                self.add_rule(str)
            elif "pred" == str[:4]:
                str = str.split("pred:")[1].strip()
                self.add_pred(str)
            else:
                self.add_fact(str)
        f.close()
        
    def add_pred(self,str):
        sinx = str.find("(")
        einx = str.find(")")
        
        pred_name = str[:sinx]
        self.predicates[pred_name] = {}
        self.predicates[pred_name]["name"] = pred_name
        self.predicates[pred_name]["var_classes"] = []
        for type_var in str[sinx+1:einx].split(","):
            if type_var not in self.classes:
                self.classes[type_var] = []
            self.predicates[pred_name]["var_classes"].append(type_var)

    def add_rule(self,line):
        [hstr,bstr] = line.split(":-")
        hpred,var_list = read_pred(hstr)
        r = {}
        r["head"] = (hpred,var_list)
        c = 0
        self.head_list.append(hpred)
        bstr = re.sub(r"\s+", "", bstr)
        for bs in bstr.split("),"):            
            pred,vlist = read_pred(bs)
            self.body_list.append(pred)
            r["l"+str(c)] = (pred,vlist)
            c+=1
        
        if hpred in self.rules:
            self.rules[hpred].append(r)
        else:
            self.rules[hpred] = [r]
    
    def update_facts(self,name):
        self.facts[name] = None

    def add_fact(self,str):
        sinx = str.find("(")
        einx = str.find(")")

        if einx!=-1:
            objs = str[sinx+1:einx].split(",")
        else:
            objs = str[sinx+1:].split(",")
            print(objs)
        pred_name = str[:sinx]
        # Add objects to class
        f_obj_lst = []
        #print(self.predicates)
        for i in range(len(objs)):
            obj = objs[i]
            var_type = self.predicates[pred_name]["var_classes"][i]
            if obj not in self.classes[var_type]:
                self.classes[var_type].append(obj)
            
            # add obj index to fact object list
            f_obj_lst.append(self.classes[var_type].index(obj))
        
        # Add facts
        if pred_name not in self.facts:
            self.facts[pred_name] = np.array([f_obj_lst]) 
        else:
            self.facts[pred_name] = np.append(self.facts[pred_name],[f_obj_lst],axis=0)
    
    def organise_rules(self,by_level=False):
        if not by_level:
            return self.rules
        # organise rules
        self.stacked_rules = {}
        head_list = self.rules.keys()
        self.depth = 0
        for head in head_list:
            level=0
            for l in self.rules[head]:
                if l=="head":
                    continue
                (pred,vlist) = self.rules[head][l]
                if pred in head_list:
                    level+=1
            if level not in self.stacked_rules:
                self.stacked_rules[level] = {}

            self.stacked_rules[level][head] = self.rules[head]
            if self.depth < level:
                self.depth=level
        return self.stacked_rules

if __name__=="__main__":
    kb = KB("/Users/sntran/WORK/projects/deepsymbolic/code/gnlp/examples/single_digit/gnlp/kb.gnlp")
    #print(kb.facts)
    #print(kb.facts["sum"].shape)
    #print(kb.predicates)
    #print(kb.classes)
    print(kb.rules)
