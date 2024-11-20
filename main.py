from modules import (Initial,Init,B2LL2B,betweenLayercascade,
CascadeInPow,CascadeInCom,add_renew)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import time
# time.sleep(4*60*60)
from copy import deepcopy
import networkx as nx
from tqdm import tqdm
global DEBUG


DEBUG = False

# load probs for removing edges
PROBS = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])/100



NetworkCase = 118 #network case:/
Edges = np.load('EDGES.npy',allow_pickle=True) #load possible edges for reproducability

prePath = f'Buss{NetworkCase}'
os.makedirs(f'{prePath}',exist_ok=True) 
ALL = ['hbtc', 'hdegree']
Sample_Edge = Edges[1:5:1,1::25] #choose a sample of possible removal edges
renew = 0.1 #fraction of nodes with renewable capacity
renew2 = 0.1 # amount of renewable capacity 
case=1 #case for each network max=5
base = False
if renew==0 or renew2==0: #for situation where there is not extra or slack capacity
    base = True
print(f"\n[starting the process on {NetworkCase} buss. with fracton of capable nodes for extra capacity as {renew} and amount of capacity as {renew2},for {Sample_Edge.shape[0]} Probabalities and {Sample_Edge.shape[1]} Iterations.]\n")
for Slack_Kind in ALL:
    print(f'\n[for senario: {Slack_Kind}]\n' if not base else '[for Base senario]\n')
    #for each senario:
    #senario1 is for choosing nodes for adding renewable capacity 
    # senario2 is for choosing load or generator for addig renewable capacity
    busses,lines,bussesC,linesC,index_reset = Initial(casename=f'{NetworkCase}',case=case) # network initialization
    busses,lines = Init(lines,busses) # calculate flow and capacity and check whether powerflow work or not
    busses = add_renew(busses,senario1=Slack_Kind,senario2="_",renew=renew,renew2=renew2) #this function add renewable power to nodes

    if base:
        Slack_Kind =  "Base"+"_" +f'{renew}_{renew2}_{NetworkCase}'
    else:
        Slack_Kind =  Slack_Kind + "_" +f'{renew}_{renew2}_{NetworkCase}'

    os.makedirs(f'{prePath}/{Slack_Kind}',exist_ok=True)
    os.makedirs(f'{prePath}/{Slack_Kind}_details',exist_ok=True)
    path = f'{prePath}/{Slack_Kind}/' # path for general results for every network
    path2 = f'{prePath}/{Slack_Kind}_details/' #path for detailed results like for every iterations
    nodes = []
    edges = []
    powersU = []
    powersG = []
    nodes0 = []
    edges0 = []
    powersU0 = []
    powersG0 = []
    for EdgeIter,prob in enumerate(Sample_Edge):
        node = []
        edge = []
        powerU = []
        powerG = []
        node0 = []
        edge0 = []
        powerU0 = []
        powerG0 = []
        sampleIter = 0
        for j in tqdm(prob):
            sampleIter+=1
            busses_,lines_ = deepcopy(busses),deepcopy(lines)
            bussesC_,linesC_ = deepcopy(bussesC),deepcopy(linesC)
            lines_ = lines_.drop(j) #drop initial lines\enges
            
            netB,netL = B2LL2B(busses_,lines_) #apply removed enges on nodes
            edge0.append(netL.shape[0])
            node0.append(netB.shape[0])
            powerU0.append(netB['Pd'].sum())
            powerG0.append(netB['Pg'].sum())


            netB,netL,netC,netCl= betweenLayercascade(netB,netL,bussesC,linesC) #apply cascade in intralayer mode
            netC,netCl = CascadeInCom(netC,netCl) #apply cascade in communication layer
            netB,netL,netC,netCl= betweenLayercascade(netB,netL,netC,netCl) #apply cascade in intralayer mode

            d = 0

            Cascade = True
            while Cascade:
                d+=1

                a1,a2,a3 = netB['Pg'].sum(),netB['Pd'].sum(),set(netL['tuple'])
                netB,netL = CascadeInPow(netB,netL) #apply cascade in power network
                b1,b2,b3 = netB['Pg'].sum(),netB['Pd'].sum(),set(netL['tuple'])
                
                netC,netCl = CascadeInCom(netC,netCl) #apply cascade in communication network

                netB,netL,netC,netCl= betweenLayercascade(netB,netL,netC,netCl) #apply casce intralayer

                #check if no enges or nodes has been removed last iteration if not so the cascadeing process ends
                if abs(a1-a2)<=1 and abs(a1-b1)<=1 and abs(a2-b2)<=1 and a3 == b3:
                    Cascade = False
                if d==20: #if in 20 iteratin solver didnt converge kill it

                    netB = pd.DataFrame(columns=netB.columns)
                    netL = pd.DataFrame(columns=netL.columns)
                    netC = pd.DataFrame(columns=netC.columns)
                    netCl = pd.DataFrame(columns=netCl.columns)

            if DEBUG: print(netB['Pd'].sum())
            netB.to_csv(f'{path2}_{EdgeIter}_{sampleIter}_{len(j)}_netB.csv',index=False)
            netL.to_csv(f'{path2}_{EdgeIter}_{sampleIter}_{len(j)}_netL.csv',index=False)
            netC.to_csv(f'{path2}_{EdgeIter}_{sampleIter}_{len(j)}_netC.csv',index=False)
            netCl.to_csv(f'{path2}_{EdgeIter}_{sampleIter}_{len(j)}_netCl.csv',index=False)
            edge.append(netL.shape[0])
            node.append(netB.shape[0])
            powerU.append(netB['Pd'].sum())
            powerG.append(netB['Pg'].sum())
        if DEBUG: print('powerU,powerG: ',np.array(powerU).mean(),np.array(powerG).mean())
        print('powerU,powerG: ',np.array(powerU).mean(),np.array(powerG).mean())
        edges0.append(edge0)
        nodes0.append(node0)
        powersG0.append(powerG0)
        powersU0.append(powerU0)

        edges.append(edge)
        nodes.append(node)
        powersG.append(powerG)
        powersU.append(powerU)

    po =1
    pu = -1
    POP = np.array(powersU).shape[1]
    np.save(f'{path}REAU_{POP}',np.array(powersU))
    np.save(f'{path}REAG_{POP}',np.array(powersG))
    np.save(f'{path}REAE_{POP}',np.array(edges))
    np.save(f'{path}REAN_{POP}',np.array(nodes))
    np.save(f'{path}REAU0_{POP}',np.array(powersU0))
    np.save(f'{path}REA0_{POP}',np.array(powersG0))
    np.save(f'{path}REAE0_{POP}',np.array(edges0))
    np.save(f'{path}REAN0_{POP}',np.array(nodes0))
    if base:
        break