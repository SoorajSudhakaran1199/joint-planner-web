#!/usr/bin/env python3
from __future__ import annotations
import csv, io, math
from dataclasses import dataclass
from typing import List
from urllib.parse import quote

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# ----------------- Robot + planner core -----------------

@dataclass
class RobotModel:
    name: str
    dof: int
    theta_offset: np.ndarray
    d: np.ndarray
    a: np.ndarray
    alpha: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray

    def fk_all(self, q: np.ndarray) -> List[np.ndarray]:
        T = np.eye(4); frames = []
        for i in range(self.dof):
            th = q[i] + self.theta_offset[i]
            d, a, al = float(self.d[i]), float(self.a[i]), float(self.alpha[i])
            ct, st = math.cos(th), math.sin(th)
            ca, sa = math.cos(al), math.sin(al)
            A = np.array([[ct, -st*ca,  st*sa,  a*ct],
                          [st,  ct*ca, -ct*sa,  a*st],
                          [0.0,    sa,     ca,    d],
                          [0.0,   0.0,    0.0,  1.0]])
            T = T @ A
            frames.append(T.copy())
        return frames

    @staticmethod
    def ur5e() -> "RobotModel":
        a = np.array([0, -0.425, -0.39225, 0, 0, 0], float)
        d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996], float)
        alpha = np.array([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0], float)
        theta_offset = np.zeros(6)
        q_min = np.radians([-360]*6); q_max = np.radians([360]*6)
        return RobotModel("UR5e", 6, theta_offset, d, a, alpha, q_min, q_max)

@dataclass
class Box:
    center: np.ndarray  # meters
    size:   np.ndarray  # meters
    def bounds(self):
        s2 = self.size/2.0
        return self.center - s2, self.center + s2

def collision(robot: RobotModel, q: np.ndarray, boxes: List[Box], margin: float) -> bool:
    frames = robot.fk_all(q)
    pts = [T[:3,3] for T in frames]
    for i in range(len(frames)-1):
        pts.append((frames[i][:3,3] + frames[i+1][:3,3]) * 0.5)
    for b in boxes:
        bmin, bmax = b.bounds()
        for p in pts:
            if np.all(p >= (bmin - margin)) and np.all(p <= (bmax + margin)):
                return True
    return False

def segment_free(robot, qa, qb, boxes, margin, res=0.03):
    n = max(2, int(np.linalg.norm(qb-qa)/res))
    for t in np.linspace(0,1,n):
        q = qa + t*(qb-qa)
        if collision(robot, q, boxes, margin): return False
    return True

class Node:
    __slots__=("q","parent")
    def __init__(self, q, parent): self.q, self.parent = q, parent

def rrt_connect(robot, q_start, q_goal, boxes, margin,
                step=0.10, iters=30000, goal_bias=0.25, res=0.03):
    if collision(robot, q_start, boxes, margin) or collision(robot, q_goal, boxes, margin):
        return None
    if segment_free(robot, q_start, q_goal, boxes, margin, res):
        return [q_start, q_goal]

    trees = [[Node(q_start, None)], [Node(q_goal, None)]]
    def nearest(T, q): return min(range(len(T)), key=lambda i: np.linalg.norm(T[i].q - q))
    def steer(a, b):
        d = b - a; n = np.linalg.norm(d)
        return b if n <= step else a + (d/n)*step

    for k in range(iters):
        a, b = (0,1) if k%2==0 else (1,0)
        qr = trees[b][0].q if np.random.rand() < goal_bias else np.random.uniform(robot.q_min, robot.q_max)
        ia = nearest(trees[a], qr); qa = trees[a][ia].q
        q_new = steer(qa, qr)
        if not segment_free(robot, qa, q_new, boxes, margin, res): continue
        trees[a].append(Node(q_new, ia))

        ib = nearest(trees[b], q_new); qb = trees[b][ib].q
        q_probe = q_new; parent = len(trees[a]) - 1
        while True:
            q_next = steer(q_probe, qb)
            if not segment_free(robot, q_probe, q_next, boxes, margin, res): break
            trees[a].append(Node(q_next, parent))
            parent = len(trees[a]) - 1; q_probe = q_next
            if np.allclose(q_probe, qb, atol=1e-3):
                def path(T, i):
                    out=[]
                    while i is not None:
                        out.append(T[i].q); i=T[i].parent
                    return out[::-1]
                p1 = path(trees[a], parent)
                p2 = path(trees[b], ib)
                if a==1: p1, p2 = p2, p1
                return p1 + p2[::-1][1:]
    return None

def shortcut(path, robot, boxes, margin, res=0.03, attempts=400):
    if len(path)<=2: return path
    p=list(path)
    for _ in range(attempts):
        i = np.random.randint(0,len(p)-2)
        j = np.random.randint(i+2,len(p))
        if segment_free(robot,p[i],p[j],boxes,margin,res):
            p = p[:i+1] + p[j:]
    return p

def resample_to_n(path: List[np.ndarray], N: int) -> List[np.ndarray]:
    if len(path)==1: return [path[0]]*N
    segs=[(path[i],path[i+1]) for i in range(len(path)-1)]
    lens=[np.linalg.norm(b-a) for a,b in segs]; L=float(np.sum(lens))
    if L<1e-9: return [path[0] for _ in range(N)]
    s_targets=np.linspace(0,L,N); out=[]; acc=0.0; i=0; a,b=segs[0]
    for s in s_targets:
        while i<len(lens)-1 and acc+lens[i]<s:
            acc+=lens[i]; i+=1; a,b=segs[i]
        u=0.0 if lens[i]==0 else (s-acc)/lens[i]
        out.append(a+u*(b-a))
    return out

def plan(start_deg, goal_deg, n_waypoints, obstacles_mm,
         safety_margin=0.005, step=0.10, iters=30000, goal_bias=0.25, res=0.03):
    robot = RobotModel.ur5e()
    q_start = np.radians(np.asarray(start_deg, float))
    q_goal  = np.radians(np.asarray(goal_deg,  float))
    boxes=[Box(center=np.array([cx,cy,cz])/1000.0, size=np.array([L,W,H])/1000.0)
           for (cx,cy,cz,L,W,H) in obstacles_mm]
    path = rrt_connect(robot, q_start, q_goal, boxes, safety_margin, step, iters, goal_bias, res)
    if path is None:
        raise RuntimeError("No collision-free path. Adjust goal/obstacles/safety margin.")
    path = shortcut(path, robot, boxes, safety_margin, res)
    return [list(np.degrees(q)) for q in resample_to_n(path, n_waypoints)]

# ----------------- Web routes -----------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html",
        n_waypoints=5,
        start_deg="-1.72, -98.96, -126.22, -46.29, 91.39, -1.78",
        goal_deg="80.0, -90.0, -90.0, -90.0, 90.0, 0.0",
        obstacles="73.865, 639.614, 360.0, 800.0, 500.0, 720.0",
        safety_margin=0.005, rrt_step=0.10, rrt_iters=30000, rrt_goal_bias=0.25, seg_res=0.03
    )

def _angles(s: str):
    parts=[p for p in s.replace(","," ").replace(";"," ").split() if p]
    vals=[float(x) for x in parts]
    if len(vals)!=6: raise ValueError("Provide exactly 6 joint angles (deg).")
    return vals

def _boxes(s: str):
    out=[]
    for ln in s.splitlines():
        ln=ln.strip()
        if not ln: continue
        parts=[p for p in ln.replace(","," ").replace(";"," ").split() if p]
        if len(parts)!=6: raise ValueError("Each obstacle line must be 6 numbers: cx cy cz L W H (mm).")
        out.append(tuple(float(x) for x in parts))
    if not out: raise ValueError("Provide at least one obstacle.")
    return out

@app.route("/plan", methods=["POST"])
def do_plan():
    try:
        n_waypoints = int(request.form["n_waypoints"])
        start_deg   = _angles(request.form["start_deg"])
        goal_deg    = _angles(request.form["goal_deg"])
        obstacles   = _boxes(request.form["obstacles"])
        safety_margin = float(request.form["safety_margin"])
        rrt_step      = float(request.form["rrt_step"])
        rrt_iters     = int(request.form["rrt_iters"])
        rrt_goal_bias = float(request.form["rrt_goal_bias"])
        seg_res       = float(request.form["seg_res"])

        wps = plan(start_deg, goal_deg, n_waypoints, obstacles,
                   safety_margin=safety_margin, step=rrt_step, iters=rrt_iters,
                   goal_bias=rrt_goal_bias, res=seg_res)

        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow([f"J{i+1}" for i in range(6)])
        for q in wps: w.writerow([f"{v:.6f}" for v in q])
        csv_url = "data:text/csv;charset=utf-8," + quote(buf.getvalue())

        return render_template("results.html", waypoints=wps, csv_url=csv_url, error=None)
    except Exception as e:
        return render_template("results.html", waypoints=[], csv_url=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
