"""
Convert football dataset to temporal kb format
@SNT
"""
import glob
import bisect

fs = glob.glob("./files/*.csv")

#players    = []
#teams      = []
#positions  = []
#colleges   = []

objects = []

relations = ["playsFor","wasBornOn","playsAs","attend"]

data = {}
playfor = {}
wasbornon = []
playas = {}
attend = []

for fname in fs:
    f = open(fname)
    print(fname)
    f.readline() # first line
    while True:
        line = f.readline()
        if line is None or len(line)<=0:
            f.close()
            break
        strs = line.replace("\"","").split(",")

        # Get information about time 
        yr = strs[6] #year
        if yr not in data:
            data[yr]= {"playFor":[],
                       "wasBornOn":[],
                       "playAs":[],
                       "attend":[]}

        bd = strs[4] # birthdate
        if bd not in data:
            data[bd] = {"playFor":[],
                       "wasBornOn":[],
                       "playAs":[],
                       "attend":[]}
            
        # add player to the list
        pl = strs[0]
        if pl not in objects:
            objects.append(pl)
        #else:
        #    print("same name " + pl)
            
        # add team to the list
        tm = strs[1]
        if tm not in objects:
            objects.append(tm)

        # add position to the list
        po = strs[3]
        if po not in objects:
            objects.append(po)
        
        
        # add college
        cl = strs[5]
        if cl not in objects:
            objects.append(cl)


        hid = objects.index(pl)
        tid = objects.index(tm)
        pid = objects.index(po)
        cid = objects.index(cl)
        
        if (hid,tid) not in data[yr]["playFor"]:
            data[yr]["playFor"].append((hid,tid)) # note: next version should use numpy for efficiency
        #else:
        #    raise ValueError("Duplicate data for playfor, check!!!!")

        if hid not in data[bd]["wasBornOn"]:
            data[bd]["wasBornOn"].append(hid)
        #else:
        #    raise ValueError("Duplicate data for wasbornon, check!!!!")
            
        if (hid,po) not in data[yr]["playAs"]:
           data[yr]["playAs"].append((hid,po)) # note: next version should use numpy for efficiency
        #else:
        #    raise ValueError("Duplicate data for playas, check!!!!")
    
        if (hid,cid) not in attend:
            attend.append((hid,cid)) # note: next version should use numpy for efficiency

print("Complete reading files ....")
print(len(objects))

# relation2id
f = open("relation2id.txt","w")
for i in range(len(relations)):
    r = relations[i] 
    f.write("<"+r+">"+"\t"+str(i)+"\n")
f.close()

# entity2id
f = open("entity2id.txt","w")
for i in range(len(objects)):
    o = objects[i]
    f.write("<"+o+">" + "\t" + str(i) + "\n")
f.close()

# save data
train_f = open("train.txt","w")
test_f  = open("test.txt","w")

data = sorted(data.items())

for k in data:
    #print(k[0])
    #input("")
    if len(k[1]["playFor"])>0:
        pf = k[1]["playFor"] # relation id 0
        for (o,s) in pf:
            line = str(o)+ "\t"+ "0" + "\t" + str(s) + "\t" + k[0].strip() + "\n"
            train_f.write(line)
            
    if len(k[1]["wasBornOn"])>0:
        wb = k[1]["wasBornOn"] # relation id 1
        for h in wb:
            line = str(h) + "\t"+ "1"  + "\t" + "#" + "\t" + k[0].strip() + "\n"
            train_f.write(line)
            
train_f.close()
test_f.close()
