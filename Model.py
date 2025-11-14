import numpy as np
import os
import pandas as pd
import time
from gurobipy import Model,GRB,LinExpr
import pickle
from copy import deepcopy

Model("ChooChoo")

class Node:
    def __init__(self, sio, sid, tio, tid, li, ti):
        self.sio=sio #origin station
        self.sid=sid #destination station
        self.tio=tio #departure time
        self.tid=tid #arrival time
        self.li=li # distance
        self.ti=ti #travel time

V = []
V.append(Node(1,2,6,8,600,2))
V.append(Node(2,1,6,8,600,2))
V.append(Node(1,3,10,14,800,4))
V.append(Node(3,1,17,21,800,4))
V.append(Node(2,3,14,16,600,1))
V.append(Node(3,2,19,21,600,1))

print(V)
