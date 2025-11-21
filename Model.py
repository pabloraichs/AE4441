import numpy as np
import os
import pandas as pd
import time
from gurobipy import Model,GRB,LinExpr
import gurobipy as gp
import pickle
from copy import deepcopy

model = Model("ChooChoo")
M = 100
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
    def __init__(self, sigma, station, tjo, tid, i, j, id):
        self.i = i
        self.j = j
        self.sigma = sigma
        self.station = station
        if tjo-tid>=self.sigma:
            self.tij = tjo-tid
        else:
            self.tij = tjo-tid+24
        if self.station == 2 and self.tij > 4:
            self.theta = 2
        else:
            self.theta = 1
        self.id= id

V = []
V.append(Node(1,2,6,8,600,2,0))
V.append(Node(2,1,6,8,600,2,1))
V.append(Node(1,3,10,14,800,4,2))
V.append(Node(3,1,17,21,800,4,3))
V.append(Node(2,3,14,16,600,1,4))
V.append(Node(3,2,19,21,600,1,5))

A = []
count = 0
for i in range(len(V)):
    for j in range(len(V)):
        if V[i].sid == V[j].sio:
            A.append(Arc(1.5, V[i].sid, V[j].tio, V[i].tid, i, j, count))
            count += 1

Am = [arc for arc in A if arc.theta == 2]  
Ac = [arc for arc in A if arc.theta == 1]  

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
        gp.quicksum(x[arc.i, arc.j] for arc in A if arc.i == i.id)
        <= 1,
        name=f"c1({i.id+1})"
    )

c1u = {}
for i in V:
    c1u[i.id] = model.addConstr(
        gp.quicksum(x[arc.i, arc.j] for arc in A if arc.i == i.id)
        >= 1,
        name=f"c1({i.id+1})"
    )


c2 = {}
for i in V:
    c2[i.id] = model.addConstr(
        gp.quicksum(x[arc.i, arc.j] for arc in A if arc.i == i.id)
                               == gp.quicksum(x[arc.i, arc.j] for arc in A if arc.j == i.id),
                               name=f"c2({i.id+1})")

c3 = {}
for i in V:
    c3[i.id] = model.addConstr(
        a[i.id] <= 48,
        name=f"c3({i.id+1})"
    )

c4 = {}
for arc in Am:
    c4[arc.id] = model.addConstr(
        a[arc.j] <= V[arc.j].ti + M * (1-y[arc.i, arc.j]),
        name=f"c4({arc.id+1})"
    )

c5 = {}
for arc in Am:
    c5[arc.id] = model.addConstr(
        a[arc.j] >= V[arc.j].ti - M * (1-y[arc.i, arc.j]),
        name=f"c5({arc.id+1})"
    )

c6 = {}
for arc in Am:
    c6[arc.id] = model.addConstr(
        a[arc.j] <= a[arc.i] + arc.tij + V[arc.j].ti + M * (1+y[arc.i, arc.j]-x[arc.i, arc.j]),
        name=f"c6({arc.id+1})"
    )

c7 = {}
for arc in Am:
    c7[arc.id] = model.addConstr(
        a[arc.j] >= a[arc.i] + arc.tij + V[arc.j].ti - M * (1+y[arc.i, arc.j]-x[arc.i, arc.j]),
        name=f"c7({arc.id+1})"
    )

c8 = {}
for arc in Ac:
    c8[arc.id] = model.addConstr(
        a[arc.j] <= a[arc.i] + arc.tij + V[arc.j].ti + M * (1 - x[arc.i, arc.j]),
        name=f"c8({arc.id+1})"
    )

c9 = {}
for arc in Ac:
    c9[arc.id] = model.addConstr(
        a[arc.j] >= a[arc.i] + arc.tij + V[arc.j].ti - M * (1 - x[arc.i, arc.j]),
        name=f"c9({arc.id+1})"
    )

c10 = {}
for arc in Am:
    c10[arc.id] = model.addConstr(
        x[arc.i, arc.j] >= y[arc.i,arc.j],
        name=f"c10({arc.id+1})"
    )


model.setParam("LogFile", 'log_file')
model.update()
model.write('model_file.lp')
model.optimize()

if model.SolCount > 0:
    print("Status:", model.status)
    print("Objective:", model.objVal)
    print("\nSelected x arcs:")
    for (i, j), var in x.items():
        if var.x > 0.5:
            print(f"Arc({i+1},{j+1})  x = {var.x:.0f}  tij={next(a.tij for a in A if a.i==i and a.j==j)}")

    print("\nSelected y arcs:")
    for (i, j), var in y.items():
        if var.x > 0.5:
            print(f"y({i+1},{j+1}) = {var.x:.0f}")

    print("\nNode variables a and b:")
    for vid, var in a.items():
        print(f"a({vid+1}) = {var.x}")
    for vid, var in b.items():
        print(f"b({vid+1}) = {var.x}")

    # write a simple text file with selected arcs
    with open('solution.txt', 'w') as f:
        f.write(f"Objective: {model.objVal}\n")
        f.write("Selected x arcs:\n")
        for (i, j), var in x.items():
            if var.x > 0.5:
                f.write(f"Arc({i+1},{j+1})\n")
else:
    print("No solution available. Status:", model.status)