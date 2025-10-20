import os
import io
import csv
import math
import time
import numpy as np
from flask import Flask, request, render_template, send_file, abort

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------
# Utilities
# ---------------------------
def _to_deg(path_rad):
    return [[math.degrees(x) for x in q] for q in path_rad]

def _validate_shape(q, name):
    if not isinstance(q, (list, tuple)) or len(q) != 6:
        abort(400, f"{name} must be 6 joint values (rad).")

def _parse_boxes_text(boxes_text):
    obstacles = []
    if boxes_text:
        for line in boxes_text.splitlines():
            if not line.strip():
                continue
            cx, cy, cz, L, W, H = [float(v) for v in line.split()]
            obstacles.append({"cx": cx, "cy": cy, "cz": cz, "L": L, "W": W, "H": H})
    return obstacles

def _resample_path(path_rad, n_points):
    """Joint-space linear interpolation to N points (baseline)."""
    path = np.asarray(path_rad, dtype=float)
    if path.size == 0:
        return []
    if len(path) == 1 or n_points <= 1:
        return [path[0].tolist() for _ in range(max(1, n_points))]
    seg = np.linalg.norm(path[1:] - path[:-1], axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] == 0:
        return [path[0].tolist() for _ in range(n_points)]
    t /= t[-1]
    t_new = np.linspace(0.0, 1.0, n_points)
    out = []
    for tn in t_new:
        i = np.searchsorted(t, tn) - 1
        i = max(0, min(i, len(path) - 2))
        denom = (t[i+1] - t[i]) if (t[i+1] > t[i]) else 1.0
        alpha = (tn - t[i]) / denom
        out.append((path[i]*(1 - alpha) + path[i+1]*alpha).tolist())
    return out

# ---------------------------
# Planner (baseline)
# Replace this function body with your real goal-biased RRT + smoothing.
# ---------------------------
def plan_rrt_connect(start_q, goal_q, obstacles, knobs):
    """
    Return dict:
      success: bool
      path_rad: list[list[float]]  # joint waypoints in radians
      iters: int
      time_sec: float
      reason: str|None
    """
    # Fallback: straight-line joint path (keeps the UI working).
    t0 = time.time()
    path = [list(start_q), list(goal_q)]
    return {
        "success": True,
        "path_rad": path,
        "iters": 1,
        "time_sec": time.time() - t0,
        "reason": None,
    }

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/plan", methods=["POST"])
def results():
    try:
        start_deg = [float(x) for x in request.form.get("start_deg","").replace(",", " ").split()]
        goal_deg  = [float(x) for x in request.form.get("goal_deg","").replace(",", " ").split()]
        n_waypoints = int(request.form.get("n_waypoints","20"))
        boxes_text  = request.form.get("boxes","").strip()
        knobs = {
            "max_iters": int(request.form.get("max_iters","2000")),
            "step_rad": float(request.form.get("step_rad","0.05")),
            "bias": float(request.form.get("bias","0.2")),
            "safety_margin_m": float(request.form.get("safety_margin_m","0.003")),
            "shortcut_passes": int(request.form.get("shortcut_passes","30")),
        }
        start_q = [math.radians(x) for x in start_deg]
        goal_q  = [math.radians(x) for x in goal_deg]
        obstacles = _parse_boxes_text(boxes_text)
    except Exception as e:
        return render_template("results.html", error=f"Bad input: {e}")

    _validate_shape(start_q, "start")
    _validate_shape(goal_q, "goal")

    plan = plan_rrt_connect(start_q, goal_q, obstacles, knobs)
    if not plan.get("success", False) or not plan.get("path_rad"):
        return render_template("results.html", error=f"No path: {plan.get('reason','unknown')}")

    path_rad = _resample_path(plan["path_rad"], n_waypoints)
    path_deg = _to_deg(path_rad)

    return render_template(
        "results.html",
        path_deg=path_deg,
        n=len(path_deg),
        iters=plan.get("iters"),
        time_sec=plan.get("time_sec"),
        reason=plan.get("reason"),
    )

@app.route("/download.csv", methods=["POST"])
def download_csv():
    data = request.get_json(silent=True) or {}
    path_deg = data.get("path_deg")
    if not isinstance(path_deg, list) or not path_deg:
        abort(400, "path_deg missing.")
    buf = io.StringIO()
    cw = csv.writer(buf)
    cw.writerow([f"J{i+1} (deg)" for i in range(6)])
    for q in path_deg:
        cw.writerow([f"{float(v):.6f}" for v in q])
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="waypoints.csv",
    )

@app.route("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
