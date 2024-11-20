import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import warnings
from copy import deepcopy
import sys
import random
import time
import cvxpy as cp
import numpy as np
import sys
warnings.filterwarnings('ignore')


global DEBUG

DEBUG = False

#network initialization, this function do every thing is needed to use power grid network
def Initial(casename='118',case=1):
    path = f'working_on_{casename}/case_{case}/'
    # load the data
    busses = pd.read_csv(f'{path}busses.csv')
    busses.drop(busses.columns[0],axis=1,inplace=True)
    gens = pd.read_csv(f'{path}gens.csv')
    comm =  pd.read_csv(f'{path}comm.csv')
    lines =  pd.read_csv(f'{path}lines.csv')
    lines.drop('from_to',axis=1,inplace=True)

    minlincap = lines[lines['r']>0]['r'].min()
    lines['r'][lines['r']==0] = minlincap
    minlincap = lines[lines['x']>0]['x'].min()

    gens.index = list(gens.bus)
    busses.index = list(busses['bus_i'])
    busses['Pg'] = gens['Pg']
    busses['Pmin'] = gens['Pmin']
    busses['Pmax'] = gens['Pmax']
    busses['state'] = [1] * busses.shape[0]
    busses['Cstate'] = [1] * busses.shape[0] #accessable via comm network
    # busses = add_renew(busses)
    busses.fillna(0,inplace=True)
    busses['Pdemand'] = busses['Pg'] - busses['Pd']
    reset_index = {i:ind for ind,i in enumerate(list(busses.index))}
    index_reset = {i:j for j,i in reset_index.items()}
    busses['bus_i'] = busses['bus_i'].replace(reset_index)
    busses.index= list(busses['bus_i'])
    busses['USED'] = 0
    busses['Renewable'] = 0
    lines['fbus'] = lines['fbus'].replace(reset_index)
    lines['tbus'] = lines['tbus'].replace(reset_index)
    tpl = []
    for i,j in zip(lines['fbus'],lines['tbus']):
        tpl.append((i,j))
    lines['tuple'] = tpl
    links = tpl
    G = nx.Graph()
    G.add_edges_from(links)
    btc = nx.betweenness_centrality(G)
    # Create Communication network
    busses['Btc'] = busses['bus_i'].map(btc)
    degrees = dict(G.degree())
    busses['Dgre'] = busses['bus_i'].map(degrees)


    a = []
    b = []
    for i in comm['edges']:
        i = i.replace(')','')
        i0 = int(i.replace('(','').split(',')[0])
        i1 = int(i.replace('(','').split(',')[1])
        a.append(i0)
        b.append(i1)

    linesC = pd.DataFrame()
    linesC['fbus'] = a
    linesC['tbus'] = b
    linesC['fbus'] = linesC['fbus'].replace(reset_index)
    linesC['tbus'] = linesC['tbus'].replace(reset_index)
    tpl = []
    for i,j in zip(linesC['fbus'],linesC['tbus']):
        tpl.append((i,j))
    linesC['tuple'] = tpl
    links = linesC['tuple'].tolist()
    G = nx.Graph()
    G.add_edges_from(links)
    btc = nx.betweenness_centrality(G)
    bussesC = pd.DataFrame()
    bussesC['bus_i'] = busses['bus_i']
    bussesC['Btc'] = bussesC['bus_i'].map(btc)
    bussesC['base_Btc'] = bussesC['Btc']
    mean = bussesC[bussesC['base_Btc']!=0]['base_Btc'].mean()

    bussesC['max_norm_Btc'] = bussesC['base_Btc']*2.5
    bussesC['max_crit_Btc'] = bussesC['base_Btc']*2.5
    bussesC['X'] = busses['X']
    bussesC['Y'] = busses['Y']
    bussesC['Operator'] = [False] * bussesC.shape[0]
    bussesC['Operator'][bussesC['max_norm_Btc']==bussesC['max_norm_Btc'].max()] = True

    return busses,lines,bussesC,linesC,index_reset

# add reneable capacity to network
def add_renew(bus,senario1='hbtc',senario2='_',renew=0.1,renew2=0.1):
    RN = bus['Pd'].sum() * renew # fraction to be renewable power
    if senario2 == 'load':
        num = int(bus[bus['type']==1].shape[0] * renew2) # how many renewable buss
        RN2 = RN / num
        bus['Renewable'] = [0] * bus.shape[0]
    elif senario2 == 'gen':
        num = int(bus[bus['type']!=1].shape[0] * renew2) # how many renewable buss
        RN2 = RN / num
        bus['Renewable'] = [0] * bus.shape[0]    
    else:
        num = int(bus.shape[0] * renew2) # how many renewable buss
        RN2 = RN / num
        bus['Renewable'] = [0] * bus.shape[0]   


    if senario1 == 'hsort':
        if senario2 == 'load':
            u = bus[bus['type']==1].sort_values(by='Pd',ascending=False)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']

        elif senario2 == 'gen':
            u = bus[bus['type']!=1].sort_values(by='Pg',ascending=False)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']
        
        else:
            u = bus.sort_values(by='Pg',ascending=False)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']



    elif senario1 == 'lsort':
        if senario2 == 'load':
            u = bus[bus['type']==1].sort_values(by='Pd',ascending=True)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']

        elif senario2 == 'gen':
            u = bus[bus['type']!=1].sort_values(by='Pg',ascending=True)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']
        else:
            u = bus.sort_values(by='Pg',ascending=True)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']
        
    elif senario1 == 'hbtc':
        if senario2 == 'load':
            u = bus[bus['type']==1].sort_values(by='Btc',ascending=False)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']

        elif senario2 == 'gen':
            u = bus[bus['type']!=1].sort_values(by='Btc',ascending=False)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']
        else:
            u = bus.sort_values(by='Btc',ascending=False)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']
        
    elif senario1 == 'hdegree':
        if senario2 == 'load':
            u = bus[bus['type']==1].sort_values(by='Dgre',ascending=False)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']

        elif senario2 == 'gen':
            u = bus[bus['type']!=1].sort_values(by='Dgre',ascending=False)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']
        else:
            u = bus.sort_values(by='Dgre',ascending=False)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']
        

    elif senario1 == 'lbtc':
        if senario2 == 'load':
            u = bus[bus['type']==1].sort_values(by='Btc',ascending=True)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']

        elif senario2 == 'gen':
            u = bus[bus['type']!=1].sort_values(by='Btc',ascending=True)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']
        else:
            u = bus.sort_values(by='Btc',ascending=True)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']
        
    elif senario1 == 'ldegree':
        if senario2 == 'load':
            u = bus[bus['type']==1].sort_values(by='Dgre',ascending=True)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']

        elif senario2 == 'gen':
            u = bus[bus['type']!=1].sort_values(by='Dgre',ascending=True)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']
        else:
            u = bus.sort_values(by='Dgre',ascending=True)
            u['Renewable'].iloc[0:num] = RN2
            bus['Renewable'].loc[u.index] = u['Renewable']

    return bus



# find_slack_bus
def slackid(busses):
    busses_ = deepcopy(busses)
    busses_['slack'] = busses_['Pmax'] - busses_['Pg']
    busses_ = busses_[busses_['type']!=1]

    c = busses_[busses_['Cstate']==1].sort_values(by='slack',ascending=False)
    if c.shape[0] > 0:
        slack_bus = int(c.iloc[0]['bus_i'])
    else:
        slack_bus = None
    return slack_bus


# find_slack_bus_value
def slackvalue(busses):
    busses_ = deepcopy(busses)
    busses_['slack'] = busses_['Pmax'] - busses_['Pg']
    return busses_


# update demands
def demand(busses):
    busses_ = deepcopy(busses)
    busses_['Pdemand'] = busses_['Pg'] - busses_['Pd']
    return busses_

# a simple test data 
def test():
    busses_test = pd.DataFrame()
    busses_test['bus_i'] = [0,  1,  2,  3,   4,  5,  6]
    busses_test['Pmin'] =  [0,  0,  0,  0,   0,  0,  0]
    busses_test['Pmax'] =  [0,200,  0,400,   0,900,  0]
    busses_test['Pd'] =    [0,100,200,200, 400,100, 60]
    busses_test['Pg'] =    [0,  0,  0,400,   0,500,  0]
    busses_test['type'] =  [1,  2,  1,  2,   1,  2,  1]
    busses_test['Pdemand'] = busses_test['Pg'] - busses_test['Pd']
    lines_test = pd.DataFrame()
    lines_test['fbus'] = [1,0,2,4,4,1,2,5,6,1]
    lines_test['tbus'] = [2,2,3,3,1,0,0,6,0,6]
    lines_test['r'] = [0.3] * 10
    lines_test['x'] = [0.3] * 10

    return busses_test,lines_test





# Load balamcing 

# Senario 1
# simply just make slack goes up or down

def loadBalancingS(bus,slack_bus_):
    busses_= deepcopy(bus)
    busses_ = slackvalue(busses_)
    busses_ = demand(busses_)
    tot = busses_['Pdemand'].sum()
    senario = np.array([False,False,False,False,False])
    if tot==0:
        senario[0] = True 
        busses_ = demand(busses_)
        busses_ = deepcopy(bus)
        return senario, busses_
    
    elif tot < 0 and abs(tot) <= busses_['slack'].iloc[slack_bus_]:
        senario[1] = True
        busses_['Pg'].iloc[slack_bus_] += -tot
        busses_['slack'].iloc[slack_bus_] += tot
        busses_ = demand(busses_)
        return senario, busses_

    elif tot > 0 and tot <= busses_['Pg'].iloc[slack_bus_]:
        senario[2] = True
        busses_['Pg'].iloc[slack_bus_] += -tot
        busses_ = demand(busses_)
        return senario, busses_     

    elif tot > 0 and tot > busses_['Pg'].iloc[slack_bus_]:
        busses_ = demand(busses_)
        return senario, deepcopy(bus)   
           
    elif tot < 0 and abs(tot) >= busses_['slack'].iloc[slack_bus_]:
        busses_ = demand(busses_)
        return senario, deepcopy(bus)
        # here we have nothing to do
    else: 
        return np.array([False,False,False,False,False]),deepcopy(bus)





# run first power flow on data and 
# calculate the flow of lines with max flow line and 
# critical max flow line

def Init(lines,busses):
    # Example input data
    # Power generated (positive) and consumed (negative) at each bus
    P = busses['Pdemand']  # Example: 5 buses
    slack_bus = slackid(busses) # finding slack bus

    senario , busses = loadBalancingS(busses,slack_bus)
    if senario.any():
        P = busses['Pdemand']
    else:
        sys.exit()

    num_buses = P.shape[0] # number of busses

    R = np.zeros((num_buses,num_buses)) # reactance or resistance for all lines

    for i,j,r in zip(lines['fbus'],lines['tbus'],lines['x']): #filling R with small reactance
        R[i,j] = r
        R[j,i] = r

    B = np.zeros((num_buses, num_buses)) # like before but this time to calculate 1/R

    for i in range(num_buses): #filling B
        for j in range(num_buses):
            if i != j and R[i, j] != 0:
                B[i, j] = -1 / R[i, j]

    for i in range(num_buses): #filling B diagonally
        B[i, i] = -np.sum(B[i, :])

    B_prime = np.delete(np.delete(B, slack_bus, axis=0), slack_bus, axis=1) #calculate B Prime
    P_prime = np.delete(P, slack_bus) #calculate C Prime

    theta_prime = np.linalg.solve(B_prime, P_prime) #Solving eqation

    # Insert the slack bus angle (0 degrees)
    theta = np.insert(theta_prime, slack_bus, 0) # finding theta or degrees

    # Calculate the power flow on each line
    power_flows = np.zeros_like(R)
    for i in range(num_buses):
        for j in range(num_buses):
            if i != j and R[i, j] != 0:
                power_flows[i, j] = (theta[i] - theta[j]) / R[i, j]
    P_prime_check = np.matmul(B_prime, theta_prime)
    if DEBUG: print("Verification passed:", np.allclose(P_prime, P_prime_check))

    flow = []
    for ind,(i,j) in enumerate(zip(lines['fbus'],lines['tbus'])):
        flow.append(abs(power_flows[i,j]))
        # flow.append(power_flows[i,j])

    lines['flow'] = np.around(flow,3)

    lines['base_flow'] = lines['flow']
    k = np.around(lines['base_flow'].mean(),3)
    lines['max_norm_flow'] = lines['flow'] * 2
    lines['max_norm_flow'].loc[lines[lines['flow']<(2*k)].index] = 2 * k
    lines['max_crit_flow'] = lines['flow'] * 3.5
    return busses,lines



def DCPower(busses,lines,slack_bus):
    # passed = True
    busses_,lines_ = deepcopy(busses),deepcopy(lines)
    n2o = {ind:i for ind,i in enumerate(list(busses_.bus_i))}
    o2n = {j:i for i,j in n2o.items()}
    busses_['bus_i'].replace(o2n,inplace=True)
    lines_['fbus'].replace(o2n,inplace=True)
    lines_['tbus'].replace(o2n,inplace=True)
    busses_ = demand(busses_)
    slack_bus = o2n[slack_bus]
    if slack_bus in set(busses_['bus_i']):
        pass
    else:
        slack_bus = slackid(busses_)

    P = busses_['Pdemand']
    num_buses = P.shape[0] # number of busses

    R = np.zeros((num_buses,num_buses)) # reactance or resistance for all lines

    for i,j,r in zip(lines_['fbus'],lines_['tbus'],lines_['x']): #filling R with small reactance
        R[i,j] = r
        R[j,i] = r

    B = np.zeros((num_buses, num_buses)) # like before but this time to calculate 1/R

    for i in range(num_buses): #filling B
        for j in range(num_buses):
            if i != j and R[i, j] != 0:
                B[i, j] = -1 / R[i, j]

    for i in range(num_buses): #filling B diagonally
        B[i, i] = -np.sum(B[i, :])


    if slack_bus != None:
        B_prime = np.delete(np.delete(B, slack_bus, axis=0), slack_bus, axis=1) #calculate B Prime
        P_prime = np.delete(P, slack_bus) #calculate C Prime

        theta_prime = np.linalg.solve(B_prime, P_prime) #Solving eqation

        # Insert the slack bus angle (0 degrees)
        theta = np.insert(theta_prime, slack_bus, 0) # finding theta or degrees

        # Calculate the power flow on each line
        power_flows = np.zeros_like(R)
        for i in range(num_buses):
            for j in range(num_buses):
                if i != j and R[i, j] != 0:
                    power_flows[i, j] = (theta[i] - theta[j]) / R[i, j]
        P_prime_check = np.matmul(B_prime, theta_prime)
        if DEBUG:
            if DEBUG: print("Verification passed:", np.allclose(P_prime, P_prime_check))

        flow = []
        for ind,(i,j) in enumerate(zip(lines_['fbus'],lines_['tbus'])):
            flow.append(abs(power_flows[i,j]))
            # flow.append(power_flows[i,j])

        lines_['flow'] = np.around(flow,3)
        busses_['bus_i'].replace(n2o,inplace=True)
        lines_['fbus'].replace(n2o,inplace=True)
        lines_['tbus'].replace(n2o,inplace=True)
        return True, lines_,busses_
    return False, lines_,busses_


# ----------------------------------------

# some useful functions

# update network with component
def component2net(bus,lin,component):
    busi = set(component)
    bus = bus[bus['bus_i'].isin(busi)]
    lin = lin[(lin['fbus'].isin(busi)) & (lin['tbus'].isin(busi))]
    return bus,lin

# some nodes are failed so we should update lines
def bus2lin(bus,lin):
    busi = set(bus.bus_i)
    lin = lin[(lin['fbus'].isin(busi)) & (lin['tbus'].isin(busi))]
    return bus,lin

# some lines are failed so we should update nodes
def lin2bus(bus,lin):
    fb = set(lin['fbus'])
    tb = set(lin['tbus'])
    ftb = fb | tb
    bus = bus[bus['bus_i'].isin(ftb)]
    return bus,lin

# update lines and nodes
# use it after each node/edge removal
def B2LL2B(bus,lin):
    b = set(bus['bus_i'])
    f = set(lin['fbus'])
    t = set(lin['tbus'])
    ft = f|t
    while b!= ft:
        bus,lin = lin2bus(bus,lin)
        bus,lin = bus2lin(bus,lin)
        b = set(bus['bus_i'])
        f = set(lin['fbus'])
        t = set(lin['tbus'])
        ft = f|t

    return bus,lin

# calculate equlidion distance
def distan(n1,n2):
    a1,b1 = n1[0],n1[1]
    a2,b2 = n2[0],n2[1]
    return np.sqrt((a2-a1)**2+(b2-b1)**2)


# create graph from network
def net2top(line):
    line_ = deepcopy(line)
    edges = line_['tuple'].to_list()
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

# graph to network
def top2net(G):
    G_ = deepcopy(G)
    bus = pd.DataFrame()
    pos = nx.spring_layout(G_, seed=42)
    def posx(x):
        return pos[x][0]
    
    def posy(x):
        return pos[x][1]
        
    bus['bus_i'] = list(G_.nodes())
    bus['X'] = bus['bus_i'].apply(posx)
    bus['Y'] = bus['bus_i'].apply(posy)

    lines = pd.DataFrame()
    lines['fbus'] = np.array(list(G_.edges()))[:,0]
    lines['tbus'] = np.array(list(G_.edges()))[:,1]
    lines['tuple'] = list(G.edges())
    return bus,lines


# give a sorted list component
def sortedcomponent(G):
    G_ = deepcopy(G)
    connected_components = list(nx.connected_components(G_))
    if len(connected_components) != 0: 
        sorted_components = sorted(connected_components, key=len, reverse=True)
    else:
        sorted_components = [{}]
    return sorted_components


# ---------------- [cascading failure] -----------

# first impact of power outage 
    # this is because we say for each node in power grid which is 
    # removed the identical one on communication will be removed

def betweenLayercascade(netB,netL,netC,netCl):
    
    powNSet = set(netB[netB['Cstate']==1]['bus_i'])
    comNSet = set(netC['bus_i'])
    if DEBUG: print('len(powNSet) ',len(powNSet),file=open('logs.txt','a'))
    if DEBUG: print('len(comNSet) ',len(comNSet),file=open('logs.txt','a'))
    
    diff = comNSet & powNSet
    netC = netC[netC['bus_i'].isin(diff)]
    netB['Cstate'][~netB['bus_i'].isin(diff)] = 0
    netCl = netCl[netCl['fbus'].isin(diff) & netCl['tbus'].isin(diff)]
    netC,netCl = B2LL2B(netC,netCl)
    
    powNSet = set(netB[netB['Cstate']==1]['bus_i'])
    comNSet = set(netC['bus_i'])

    diff = comNSet & powNSet
    netC = netC[netC['bus_i'].isin(diff)]
    netB['Cstate'][~netB['bus_i'].isin(diff)] = 0
    return netB,netL,netC,netCl

# V3
def CascadeInPow(net,line):
    net,line = B2LL2B(net,line)

    G = net2top(line)
    largest_component = sortedcomponent(G)
    Lexist = set()
    Lexist2 = set()
    if DEBUG: print('Power Components:', len(largest_component),file=open('logs.txt','a'))
    conm = 0
    for comp in largest_component:
        g = G.subgraph(comp)
        nodes = set(g.nodes())
        net_ = net[net['bus_i'].isin(nodes)]
        net_,line_ = B2LL2B(net_,line)
        slack_bus_ = slackid(net_)
        situation = False
        if slack_bus_ != None:
            if abs(net_['Pd'].sum() - net_['Pg'].sum()) > 0:
                    situation, net_, line_ = DCPowerOP(net_,line_,slack_bus=slack_bus_)
            else:
                situation, line_, net_ = DCPower(net_,line_,slack_bus_)
                mln = line_[line_['flow'] > line_['max_norm_flow'] + 1e-5].shape[0]
                mosess = line_[line_['flow'] > line_['max_norm_flow'] + 1e-5]
                if mln > 0:
                    mosess_ = (mosess['flow'] - mosess['max_norm_flow']).sum()
                    if DEBUG: print(f'Components N.O: {conm} Passed but {mln} lines will failed with {mosess_} overloads',file=open('logs.txt','a'))
                    situation, net_, line_ = DCPowerOP(net_,line_,slack_bus=slack_bus_)
                    mln = line_[line_['flow'] > line_['max_norm_flow'] + 1e-5].shape[0]
                    if DEBUG: print(f'Components N.O: {conm} Passed but', "it is figure it out" if mln == 0 else "still failing",file=open('logs.txt','a'))
        else:
            situation = False
        if situation:
            mosess = line_[line_['flow'] > line_['max_norm_flow'] + 1e-5]
            mln = mosess.shape[0]
            mosess_ = (mosess['flow'] - mosess['max_norm_flow']).sum()

            line_ = line_[line_['flow'] <= line_['max_norm_flow'] + 1e-5] 

            net_,line_ = B2LL2B(net_,line_)
            Lsl = set(line_['fbus']) | set(line_['tbus'])
            Lexist = Lexist | Lsl
            Lexist2 = Lexist2 | set(line_['tuple'])
            net.loc[net_.index] = net_
            line['flow'].loc[line_.index] = line_['flow']

            if DEBUG: print(f'Components N.O: {conm} Passed but {mln} lines failed with {mosess_} overloads',file=open('logs.txt','a'))
        else:
            if DEBUG: print(f'Components N.O: {conm} Failed with slack number {slack_bus_}',file=open('logs.txt','a'))
            net['Pd'].loc[net_.index] = 0
            net['Pdemand'].loc[net_.index] = 0
            net['Pg'].loc[net_.index] = 0
            net['Renewable'].loc[net_.index] = 0
        conm += 1
    net = net[net['bus_i'].isin(Lexist)]
    line = line[line['tuple'].isin(Lexist2)]
    net,line = B2LL2B(net,line)
    return net,line


def CascadeInCom(net,line):
    net,line = B2LL2B(net,line)
    G = net2top(line)
    if DEBUG: print('[IN1]',len(G.nodes()),len(G.edges()),file=open('logs.txt','a'))
    largest_component = sortedcomponent(G)
    for component in largest_component[:1]:
        net_ = deepcopy(net)
        line_ = deepcopy(line)

        g = G.subgraph(component)
        btc = nx.betweenness_centrality(g)
        if DEBUG: print('[IN2]',len(g.nodes()),len(g.edges()),file=open('logs.txt','a'))
        
        net_['Btc'] = net_['bus_i'].map(btc)
        net_.fillna(np.inf,inplace=True)
        if DEBUG: print('[IN3]',net_.shape[0],file=open('logs.txt','a'))
        net_ = net_[net_['Btc'] <= net_['max_norm_Btc']]
        if DEBUG: print('[IN4]',net_.shape[0],file=open('logs.txt','a'))

        node = set(net_['bus_i'])
        net_,line_ = B2LL2B(net_,line_)
        if DEBUG: print('[IN5]',net_.shape[0],file=open('logs.txt','a'))



        G = net2top(line_)
        largest_component = sortedcomponent(G)
        for component in largest_component[:1]:
            g = G.subgraph(component)
            if DEBUG: print('[IN6]',len(g.nodes()),len(g.edges()),file=open('logs.txt','a'))
            nodes = set(g.nodes())
            # if net_[net_['bus_i'].isin(nodes)]['Operator'].sum()==1:
            net_ = net_[net_['bus_i'].isin(nodes)]
            net_,line = B2LL2B(net_,line_)
            if DEBUG: print('[IN7]',net_.shape[0],file=open('logs.txt','a'))


            return net_,line_
        
        
    return pd.DataFrame(columns=net.columns),pd.DataFrame(columns=line.columns)




# ---------------------------------------------
# where solver begin to work and find optimal load shedding
def optc(busses,lines,slack_bus):
    num_buses = busses.shape[0]
    num_lines = lines.shape[0]

    ConstBusses = busses[busses['Cstate']==0]
    busses['Renewable'] = busses['Renewable'] * busses['Cstate']
    Sg =busses['Pg'].iloc[slack_bus]
    Sd =busses['Pd'].iloc[slack_bus]
    Pd = np.delete(busses['Pd'], slack_bus)
    Pg = np.delete(busses['Pg'], slack_bus)
    Pr = np.delete(busses['Renewable'], slack_bus)
    CST = np.delete(busses['Cstate'], slack_bus)
    busses_ = deepcopy(busses)
    lines_ = deepcopy(lines)
    P_gen_max = busses_['Pmax']
    CONSTPG = np.array(busses['Pg'] * (1-busses['Cstate']))
    CONSTPD = np.array(busses['Pd'] * (1-busses['Cstate']))
    CONSTPG = np.delete(CONSTPG, slack_bus)
    CONSTPD = np.delete(CONSTPD, slack_bus)
    constPGA = CONSTPG.sum() 
    constPDA = CONSTPD.sum()
    Pgm = (np.delete(busses['Pmax'], slack_bus) - (np.delete(busses['Pg'], slack_bus)))* CST

    P = busses_['Pg'] - busses_['Pd']
    R = np.zeros((num_buses, num_buses))
    R[lines_['fbus'], lines_['tbus']] = lines_['x']
    R[lines_['tbus'], lines_['fbus']] = lines_['x']
    B = np.where(R != 0, 1 / R, 0)
    for i in range(B.shape[0]):
        B[i, i] = -np.sum(B[i, :]) + B[i, i]
    P_demand_reduced = np.delete(P, slack_bus)
    B = np.delete(np.delete(B, slack_bus, axis=0), slack_bus, axis=1)
    P = np.delete(P, slack_bus)
    B_reduced = B
    C = Pd
    Calpha = cp.Variable(num_lines)

    BR = cp.Variable(num_buses - 1) #this is added
    theta = cp.Variable(num_buses - 1)  # Voltage angles, excluding slack bus
    booU = cp.multiply(cp.Variable(num_buses - 1,boolean=True),CST)
    booG = cp.multiply(cp.Variable(num_buses - 1,boolean=True),CST)
    booR = cp.multiply(BR,CST) #this is added

    C2 = 10e2
    flow_vars = cp.Variable(num_lines)  # Power flows for each line

    P_r = booR @ Pr
    P_usage = booU @ Pd 
    P_generated = booG @ Pg
    objective = cp.Maximize(cp.sum(P_usage) + constPDA + Sd - cp.sum(Calpha)*C2)#-cp.sum(cp.multiply(rampup,Pgm))*0.1)
    P_generated_slack = cp.Variable()  # Slack bus power generation

    constraints = [
        (B_reduced @ theta - (cp.multiply(booG ,Pg) + CONSTPG - CONSTPD + cp.multiply(booR ,Pr) - cp.multiply(booU, Pd) )) == 0,  # Power balance equation

        (cp.sum(P_generated) + constPGA + cp.sum(P_r) + P_generated_slack - cp.sum(P_usage)-Sd-constPDA) == 0,  # Power usage + generated power = demand
    
    BR <=1,
    BR >=0, #this two are added

    ]


    # Line power flow constraints
    # if DEBUG: print(lines_.shape)
    for idx, (i, j) in enumerate(zip(lines_['fbus'], lines_['tbus'])):
        if i != slack_bus and j != slack_bus:
            i_red = i if i < slack_bus else i - 1
            j_red = j if j < slack_bus else j - 1
            constraints += [flow_vars[idx] == ((theta[i_red] - theta[j_red]) / lines_['x'].iloc[idx])]
        elif i == slack_bus:
            j_red = j if j < slack_bus else j - 1
            constraints += [flow_vars[idx] == -(theta[j_red] / lines_['x'].iloc[idx])]
        
        elif j == slack_bus:
            i_red = i if i < slack_bus else i - 1
            constraints += [flow_vars[idx] == (theta[i_red] / lines_['x'].iloc[idx])]

    # Maximum flow constraints
    for idx in range(num_lines):
        constraints += [cp.abs(flow_vars[idx]) - ( cp.multiply((1), (lines_['max_norm_flow'].iloc[idx]))) <= Calpha[idx]]# +cp.multiply(lines_['max_norm_flow'].iloc[idx],alpha[idx])
        

        constraints +=[Calpha[idx] >= 0]

    # Slack bus power generation constraint
    constraints += [
        P_generated_slack >= 0,  # Slack bus power generation should be non-negative
        P_generated_slack <= P_gen_max.iloc[slack_bus],  # Slack bus power generation should not exceed max capacity
    ]

    # Sum of generated power should meet the demand

    # Define the problem and solve
    prob = cp.Problem(objective, constraints)

    prob.solve(solver='CBC',maximumSeconds=60)
    # prob.solve(solver=cp.SCIPY)
    # check whether the solution was optimal or nat
    if prob.status != cp.OPTIMAL:
        return False,busses,lines

    PG_ = cp.multiply(booG ,Pg).value + cp.multiply(booR ,Pr).value
    PG_ = np.insert(PG_, slack_bus, P_generated_slack.value)
    PD_ = cp.multiply(booU, Pd).value
    booUt = booU.value
    booUt = np.insert(booUt, slack_bus,1)
    booGt = booG.value
    booGt = np.insert(booGt, slack_bus,1)
    
    booRt = booR.value
    booRt = np.insert(booRt, slack_bus,1)
    
    PD_ = np.insert(PD_, slack_bus,Sd)

    PR_ = (1 - booR.value)
    PR_ = np.insert(PR_,slack_bus,1)
    RMPT = 0

    busses_['Pg'] = PG_
    busses_['Pd'] = PD_
    busses_['USED'] += busses_['Renewable'] * (1-PR_)
    busses_['Renewable'] *= PR_
    lines_['flow'] = np.around(abs(np.array(flow_vars.value)),3)
    busses_['Pmax'] = busses_['Pmax'] - RMPT
    busses_['Pg'] +=  RMPT
    busses_['Cstate'] = np.array(booUt,dtype=int) | np.array(booGt,dtype=int) |(np.array(booRt)>0)
    busses_.loc[ConstBusses.index] = ConstBusses
    if DEBUG: print('[OUTER]',busses_['Pg'].sum(),busses_['Pd'].sum(),file=open('logs.txt','a'))

    return True, busses_, lines_


def DCPowerOP(busses,lines,slack_bus):
    busses_,lines_ = deepcopy(busses),deepcopy(lines)
    n2o = {ind:i for ind,i in enumerate(list(busses_.bus_i))}
    o2n = {j:i for i,j in n2o.items()}
    busses_['bus_i'].replace(o2n,inplace=True)
    lines_['fbus'].replace(o2n,inplace=True)
    lines_['tbus'].replace(o2n,inplace=True)
    busses_ = demand(busses_)

    slack_bus = o2n[slack_bus]
    if slack_bus in set(busses_['bus_i']):
        pass
    else:
        slack_bus = slackid(busses_)



    Stat, busses_, lines_ = optc(busses_,lines_,slack_bus)
    busses_['bus_i'].replace(n2o,inplace=True)
    lines_['fbus'].replace(n2o,inplace=True)
    lines_['tbus'].replace(n2o,inplace=True)
    return Stat, busses_, lines_
