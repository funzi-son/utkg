"""
plot results
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

rs_dir = "./footballdb/log"

fnum = len(glob.glob(os.path.join(rs_dir,"log*.csv")))

rs = None
count = 0
while True:
    count+=1
    f = os.path.join(rs_dir,"log_%d.csv"%(count))
    if os.path.isfile(f) and os.path.getsize(f):
        d = np.loadtxt(f)
        if rs is None:
            rs = d
        else:
            rs = np.append(rs,d,axis=0)
        
    if count==fnum:
        break

fig,axes = plt.subplots(2,2)

axes[0,0].plot(rs[:,0],rs[:,1])
axes[0,1].plot(rs[:,0],rs[:,2])
axes[1,0].plot(rs[:,0],rs[:,3])
axes[1,1].plot(rs[:,0],rs[:,4])

plt.show()

#save image
fig = plt.figure(1)
plt.plot(rs[:,0],rs[:,1])
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.savefig("loss1.png")
plt.close()
plt.cla()
plt.clf()


fig = plt.figure(1)
plt.plot(rs[:,0],rs[:,2])
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.savefig("loss2.png")
plt.close()
plt.cla()
plt.clf()


fig = plt.figure(1)
plt.plot(rs[:,0],rs[:,3])
plt.xlabel("Iteration")
plt.ylabel("wasBornIn")
plt.savefig("wbi.png")
plt.close()
plt.cla()
plt.clf()


fig = plt.figure(1)
plt.plot(rs[:,0],rs[:,4])
plt.xlabel("Iteration")
plt.ylabel("playFor")
plt.savefig("pf.png")


plt.close()
plt.cla()
plt.clf()
