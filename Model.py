import numpy as np
import os
import pandas as pd
import time
from gurobipy import Model,GRB,LinExpr
import gurobipy as gp
import pickle
from copy import deepcopy
import random
from collections import Counter
import matplotlib.pyplot as plt

model = Model("ChooChoo")
M = 500
turnaround_time = 1.5
maintenance_time = 4
Tm = 48
Lm = 40
maintenance_stations = [2, 3, 4]
# ...existing code...

def plot_space_time(V, stations=None, time_range=(0, 24), figsize=(10, 6), filename=None, show_labels=True):
    """
    Draw a time-space diagram from a list/array of Node objects (like V).
    - V: iterable of Node objects with attributes sio, sid, tio, tid, id
    - stations: iterable/list of station ids to plot (defaults to unique station ids found in V)
    - time_range: tuple (tmin, tmax) in hours; trips that wrap-around (arrival <= departure) are drawn across midnight (arrival+24)
    - filename: if provided, image is saved to this path (PNG)
    """
    if stations is None:
        # collect unique station ids from V and sort
        stations = sorted({v.sio for v in V} | {v.sid for v in V})

    # map station id -> y coordinate (S1 at bottom)
    stations_sorted = list(stations)
    y_map = {s: idx + 1 for idx, s in enumerate(stations_sorted)}  # 1..n bottom->top

    tmin, tmax = time_range
    fig, ax = plt.subplots(figsize=figsize)

    # horizontal lines for stations
    for s in stations_sorted:
        y = y_map[s]
        ax.hlines(y, tmin, tmax, colors='lightgray', linestyles='dotted', linewidth=1)
        ax.text(tmin - (tmax - tmin) * 0.02, y, f"S{s}", va='center', ha='right', fontsize=10)

    # plot trips
    cmap = plt.get_cmap('tab20')
    for idx, v in enumerate(V):
        t0 = float(v.tio)
        t1 = float(v.tid)
        # if arrival is not after departure, assume next day
        if t1 <= t0:
            t1_plot = t1 + 24
        else:
            t1_plot = t1
        # if t0 is outside time_range, still draw if overlaps; simple clipping for display
        x = [t0, t1_plot]
        y = [y_map[v.sio], y_map[v.sid]]
        color = cmap(idx % 20)
        ax.plot(x, y, color=color, linewidth=2)
        # markers at endpoints
        ax.scatter([t0], [y_map[v.sio]], color=color, s=20, zorder=3)
        ax.scatter([t1_plot], [y_map[v.sid]], color=color, s=20, zorder=3)
        if show_labels:
            # label near mid-point
            xm = x[0] + 0.25 * (x[1] - x[0])
            ym = y[0] + 0.25 * (y[1] - y[0])
            ax.text(xm, ym, f"G{v.id+1}", fontsize=8, ha='center', va='center', backgroundcolor='white', alpha=0.8)

    ax.set_xlim(tmin, tmax if tmax > tmin else tmin + 24)
    ax.set_ylim(0.5, len(stations_sorted) + 0.5)
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("stations")
    ax.set_yticks([])  # station labels already drawn
    ax.set_title("Space-Time Diagram")
    ax.grid(False)

    # tidy ticks for x axis (hours)
    xticks = np.arange(int(tmin), int(max(tmax, tmax + 1)) + 1, max(1, int((tmax - tmin) // 12 or 1)))
    ax.set_xticks(np.arange(int(tmin), int(tmin) + 25, 2))
    ax.set_xlim(tmin, tmin + (tmax - tmin))
    plt.show()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    return fig, ax

class Node:
    def __init__(self, sio, sid, tio, tid, li, ti, id):
        self.sio=sio #origin station
        self.sid=sid #destination station
        self.tio=tio #departure time
        self.tid=tid #arrival time
        self.li=li # distance
        self.ti=ti #travel time
        self.id = id #v id
    def __repr__(self):
        return f"Node({self.sio},{self.sid},{self.tio},{self.tid},{self.li},{self.ti},{self.id})"

# V = []
# V.append(Node(1,2,6,8,6,2,0))
# V.append(Node(2,1,6,8,6,2,1))
# V.append(Node(1,3,10,14,8,4,2))
# V.append(Node(3,1,17,21,8,4,3))
# V.append(Node(2,3,14,16,6,2,4))
# V.append(Node(3,2,19,21,6,2,5))

def generate_strict_balanced_network(
        n_stations=6,
        trains_per_station=4,
        min_dist=2, max_dist=8,
        min_time=1, max_time=4,
        min_depart=5, max_depart=22,
        rng_seed=None,
        allow_self_loops=False,
        max_shuffles=10000):

    if rng_seed is not None:
        random.seed(rng_seed)

    if n_stations < 1:
        raise ValueError("n_stations must be >= 1")
    if n_stations == 1 and not allow_self_loops and trains_per_station > 0:
        raise ValueError("Impossible: single station with self-loops disallowed.")

    # Pre-generate time and distance for each OD pair
    global od_times
    global od_dists
    od_times = {}
    od_dists = {}
    for o in range(1, n_stations + 1):
        for d in range(1, n_stations + 1):
            if o == d and not allow_self_loops:
                continue
            # ensure symmetric: if (o,d) already set, use it; otherwise generate new
            if (d, o) in od_times:
                od_times[(o, d)] = od_times[(d, o)]
                od_dists[(o, d)] = od_dists[(d, o)]
            else:
                od_times[(o, d)] = random.randint(min_time, max_time)
                od_dists[(o, d)] = random.randint(min_dist, max_dist)

    # build the origin slots and destination slots with identical counts
    origins = []
    destinations = []
    for s in range(1, n_stations + 1):
        origins += [s] * trains_per_station
        destinations += [s] * trains_per_station

    # produce a random permutation of destinations that yields no origin==dest pairs
    attempts = 0
    while True:
        attempts += 1
        random.shuffle(destinations)
        if allow_self_loops:
            break
        # check for any self-loop
        bad = False
        for o, d in zip(origins, destinations):
            if o == d:
                bad = True
                break
        if not bad:
            break
        if attempts >= max_shuffles:
            if n_stations == 1:
                raise RuntimeError("Cannot avoid self-loops with 1 station")
            destinations = []
            for s in range(1, n_stations + 1):
                dest_station = (s % n_stations) + 1
                destinations += [dest_station] * trains_per_station
            break

    # now pair origins[i] -> destinations[i], using pre-generated OD times/distances
    V = []
    id_counter = 0
    for origin, dest in zip(origins, destinations):
        tio = random.randint(min_depart, max_depart)
        ti = od_times[(origin, dest)]  # use OD pair's time
        tid = tio + ti
        li = od_dists[(origin, dest)]  # use OD pair's distance
        V.append(Node(origin, dest, tio, tid, li, ti, id_counter))
        id_counter += 1

    # quick sanity checks
    out_counts = Counter(v.sio for v in V)
    in_counts  = Counter(v.sid for v in V)
    assert all(out_counts[s] == trains_per_station for s in range(1, n_stations+1)), "out-degree mismatch"
    assert all(in_counts[s]  == trains_per_station for s in range(1, n_stations+1)), "in-degree mismatch"

    return V

if __name__ == "__main__":
    V = generate_strict_balanced_network(n_stations=5, trains_per_station=3, rng_seed=1)
    for v in V:
        print(v)

    # Print arrival/departure counts to demonstrate balance
    from collections import Counter
    print("Departures per station:", Counter(v.sio for v in V))
    print("Arrivals   per station:", Counter(v.sid for v in V))

class Arc:
    def __init__(self, sigma, i, j, id, node1, node2):
        self.i = i
        self.j = j
        self.sigma = sigma
        self.station = node1.sid
        tjo = node2.tio
        tid = node1.tid
        if node1.sid == node2.sio:
            if tjo-tid>=self.sigma:
                self.tij = tjo-tid
            else:
                self.tij = tjo-tid+24
            if self.station in maintenance_stations and self.tij > maintenance_time:
                self.theta = 2
            else:
                self.theta = 1
            self.connected = 1
        else:
            if tjo-tid>=self.sigma + od_times[(node1.sid, node2.sio)]:
                self.tij = tjo-tid
            else:
                self.tij = tjo-tid+24
            if (self.station in maintenance_stations or node2.sio in maintenance_stations) and self.tij > maintenance_time + od_times[(node1.sid, node2.sio)]:
                self.theta = 2
            else:
                self.theta = 1
            self.connected = 0
        self.id = id

A = []
count = 0
for i in range(len(V)):
    for j in range(len(V)):
#        if V[i].sid == V[j].sio:
            A.append(Arc(turnaround_time, i, j, count, V[i], V[j]))
            count += 1

plot_space_time(V, filename='space_time_diagram.png')

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
        name=f"c1l({i.id+1})"
    )

c1u = {}
for i in V:
    c1u[i.id] = model.addConstr(
        gp.quicksum(x[arc.i, arc.j] for arc in A if arc.i == i.id)
        >= 1,
        name=f"c1u({i.id+1})"
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
        a[i.id] <= Tm,
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

c11 = {}
for i in V:
    c11[i.id] = model.addConstr(
        b[i.id] <= Lm,
        name=f"c11({i.id+1})"
    )

c12 = {}
for arc in Am:
    c12[arc.id] = model.addConstr(
        b[arc.j] <= V[arc.j].li + M * (1-y[arc.i, arc.j]) + M * (1-arc.connected),
        name=f"c12({arc.id+1})"
    )

c12 = {}
for arc in Am:
    c12[arc.id] = model.addConstr(
        b[arc.j] <= V[arc.j].li + M * (1-y[arc.i, arc.j]) + od_dists[V[arc.i].sid, V[arc.j].sio] + M * (arc.connected),
        name=f"c12({arc.id+1})"
    )

c13 = {}
for arc in Am:
    c13[arc.id] = model.addConstr(
        b[arc.j] >= V[arc.j].li - M * (1-y[arc.i, arc.j]) - M * (1-arc.connected),
        name=f"c13({arc.id+1})"
    )

c13 = {}
for arc in Am:
    c13[arc.id] = model.addConstr(
        b[arc.j] >= V[arc.j].li - M * (1-y[arc.i, arc.j]) + od_dists[V[arc.i].sid, V[arc.j].sio] - M * (arc.connected),
        name=f"c13({arc.id+1})"
    )

c14 = {}
for arc in Am:
    c14[arc.id] = model.addConstr(
        b[arc.j] <= b[arc.i] + V[arc.j].li + M * (1+y[arc.i, arc.j]-x[arc.i, arc.j]) + M * (1-arc.connected),
        name=f"c14({arc.id+1})"
    )

c14 = {}
for arc in Am:
    c14[arc.id] = model.addConstr(
        b[arc.j] <= b[arc.i] + V[arc.j].li + M * (1+y[arc.i, arc.j]-x[arc.i, arc.j]) + od_dists[V[arc.i].sid, V[arc.j].sio] + M * (arc.connected),
        name=f"c14({arc.id+1})"
    )

c15 = {}
for arc in Am:
    c15[arc.id] = model.addConstr(
        b[arc.j] >= b[arc.i] + V[arc.j].li - M * (1+y[arc.i, arc.j]-x[arc.i, arc.j]) - M * (1-arc.connected),
        name=f"c15({arc.id+1})"
    )

c15 = {}
for arc in Am:
    c15[arc.id] = model.addConstr(
        b[arc.j] >= b[arc.i] + V[arc.j].li - M * (1+y[arc.i, arc.j]-x[arc.i, arc.j]) + od_dists[V[arc.i].sid, V[arc.j].sio] - M * (arc.connected),
        name=f"c15({arc.id+1})"
    )

c16 = {}
for arc in Ac:
    c16[arc.id] = model.addConstr(
        b[arc.j] <= b[arc.i] + V[arc.j].li + M * (1 - x[arc.i, arc.j]) + M * (1-arc.connected),
        name=f"c16({arc.id+1})"
    )

c16 = {}
for arc in Ac:
    c16[arc.id] = model.addConstr(
        b[arc.j] <= b[arc.i] + V[arc.j].li + M * (1 - x[arc.i, arc.j]) + od_dists[V[arc.i].sid, V[arc.j].sio] + M * (arc.connected),
        name=f"c16({arc.id+1})"
    )

c17 = {}
for arc in Ac:
    c17[arc.id] = model.addConstr(
        b[arc.j] >= b[arc.i]+ V[arc.j].li - M * (1 - x[arc.i, arc.j]) - M * (1-arc.connected),
        name=f"c17({arc.id+1})"
    )

c17 = {}
for arc in Ac:
    c17[arc.id] = model.addConstr(
        b[arc.j] >= b[arc.i]+ V[arc.j].li - M * (1 - x[arc.i, arc.j]) + od_dists[V[arc.i].sid, V[arc.j].sio] - M * (arc.connected),
        name=f"c17({arc.id+1})"
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