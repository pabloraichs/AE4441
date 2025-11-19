import numpy as np
import os
import pandas as pd
import time
from gurobipy import Model,GRB,LinExpr
import gurobipy as gp
import pickle
from copy import deepcopy

model = Model("ChooChoo")

class Node:
    def __init__(self, sio, sid, tio, tid, li, ti, id):
        self.sio=sio #origin station
        self.sid=sid #destination station
        self.tio=tio #departure time
        self.tid=tid #arrival time
        self.li=li # distance
        self.ti=ti #travel time
        self.id = id #v id

class Arc:
    def __init__(self, sigma, station, tjo, tid, i, j):
        self.i = i
        self.j = j
        self.sigma = sigma
        self.station = station
        if tjo-tid>=self.sigma:
            self.tij = tjo-tid
        else:
            self.tij = tjo-tid+24

V = []
V.append(Node(1,2,6,8,600,2,0))
V.append(Node(2,1,6,8,600,2,1))
V.append(Node(1,3,10,14,800,4,2))
V.append(Node(3,1,17,21,800,4,3))
V.append(Node(2,3,14,16,600,1,4))
V.append(Node(3,2,19,21,600,1,5))

A = []
for i in range(len(V)):
    for j in range(len(V)):
        if V[i].sid == V[j].sio:
            A.append(Arc(1.5, V[i].sid, V[j].tio, V[i].tid, i, j))

x = {}
for arc in A:
    x[arc.i, arc.j] = model.addVar(obj=arc.tij, vtype=GRB.BINARY,name = ''.join(['Arc(', str(arc.i+1), ',', str(arc.j+1), ')']))

y = {}
for arc in A:
    y[arc.i, arc.j] = model.addVar(vtype=GRB.BINARY,name = ''.join(['y(', str(arc.i+1), ',', str(arc.j+1), ')']))

a = {}
for v in V:
    a[v.id] = model.addVar(vtype=GRB.CONTINUOUS,name = ''.join(['a(', str(v.id+1), ')']))

b = {}
for v in V:
    b[v.id] = model.addVar(vtype=GRB.CONTINUOUS,name = ''.join(['b(', str(v.id+1), ')']))

model.update()

c1l = {}
for i in V:
    c1l[i.id] = model.addConstr(
        gp.quicksum(x[arc.i, arc.j] for arc in A if arc.i == i)
        <= 1,
        name=f"c1({i.id+1})"
    )

c1u = {}
for i in V:
    c1u[i.id] = model.addConstr(
        gp.quicksum(x[arc.i, arc.j] for arc in A if arc.i == i)
        >= 1,
        name=f"c1({i.id+1})"
    )

model.update()