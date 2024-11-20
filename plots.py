import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import os
# collectng data from results
inpath = './Buss118/' # path of results 
outpath = 'Plots' # path to save figures

G0 = {}
G1 = {}
G2 = {}
G3 = {}

for i,j,k in os.walk(inpath):
    for i0 in k:
        if 'netB.csv' in i0:
            g = pd.read_csv(i+'/'+i0)
            t0 = G0.get(i.split('/')[-1],[])
            t1 = G1.get(i.split('/')[-1],[])
            t2 = G2.get(i.split('/')[-1],[])
            t3 = G3.get(i.split('/')[-1],[])

            t0 += [g['Pd'].sum()]
            t1 += [g['Pg'].sum()]
            t2 += [g['Renewable'].sum()]
            t3 += [g['USED'].sum()]

            G0[i.split('/')[-1]] = t0
            G1[i.split('/')[-1]] = t1
            G2[i.split('/')[-1]] = t2
            G3[i.split('/')[-1]] = t3



# #####################################[effects of extra capacity on Injected and suppied power]####################################3
gh = []

j = 0
for j in G0:	
	k = np.mean(G3[j]),np.mean(G0[j])
	gh.append(k)



ht = np.array(gh)
ht[:,1] -= ht[:,1].min()
keynumber = len(list(G3.keys()))
labels = ["_".join(i.split("_")[0:-1]) for i in list(G3.keys())]
plt.figure(figsize=(6,4),dpi=150)
plt.plot(ht[:,0],label='Injected',linestyle='--',linewidth=1.4, marker='p',markersize=5)
plt.plot(ht[:,1],label='Supplied',linestyle='--',linewidth=1.4, marker='p',markersize=5)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylabel('Power (MW)')
plt.xticks(range(keynumber),labels);
plt.xticks(rotation=45);
plt.legend()
plt.xlabel('Scenarios')
plt.savefig(outpath+'/plot1.png')

plt.show()

# ##############################[cascade bigger than 30%]#############################33

C = {}

for i,j,k in os.walk(inpath):
    for i0 in k:
        if 'REAU' in i0:
            C[i.split('/')[-1]] = i+'/'+i0

sorted_dict_by_key = dict(sorted(C.items()))
C2 = {}
for i in C:
    C2[i]=np.load(C[i])

ROBS = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])/100

number_of_probs = C2[list(C2.keys())[0]].shape[0]
h = ROBS[1:number_of_probs+1]

max_pd = 4242

kl2 = np.zeros((keynumber,number_of_probs))

for cv in range(number_of_probs):
    for knd,k in enumerate(C2):
        kl2[knd][cv] = (((max_pd - C2[k][cv])/max_pd) >= 0.30).sum()/100



plt.figure(figsize=(6,4),dpi=150)
for ind,i in enumerate(C2):
    plt.plot(h,kl2[ind].T, linestyle='--',linewidth=1.4, marker='p',markersize=5,label=f'{i}')
    
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(h);
plt.xlabel('The fraction of transmission lines removed due to initial failure')
plt.ylabel('P(Cascade > 30%)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.xticks(h);
plt.tight_layout()
plt.savefig(outpath+'/plot2.png')

plt.show()

# ###############################################[default plots]###########################################3
plt.figure(figsize=(6,4),dpi=150)
for ind,i in enumerate(C2):
    plt.plot(h,(max_pd-C2[i].mean(1))/max_pd, linestyle='--',linewidth=1.4, marker='p',markersize=5,label=f'{i}')
plt.xlabel('The fraction of transmission lines removed due to initial failure')
plt.ylabel('Cascade size')
# plt.title('$15$% Extra Line Capacity')
        # plt.grid(True)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.xticks(h);
plt.tight_layout()
plt.savefig(outpath+'/plot3.png')
plt.show()