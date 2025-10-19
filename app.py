# app.py
import os
import io
import csv
import math
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, abort

# Optional CORS
try:
    from flask_cors import CORS
    CORS_ENABLED = True
except Exception:
    CORS_ENABLED = False

app = Flask(__name__, template_folder="templates", static_folder="static")
if CORS_ENABLED:
    CORS(app)

# -----------------------------------------------------------------------------
# Hook these to your existing implementations.
# Expect:
#   plan_rrt_connect(start_q, goal_q, obstacles, knobs) -> dict
#   resample_path(path_rad, n_points) -> list[list[float]]
#   fk_ee_mm(q_rad) -> (x, y, z) in millimeters  (rename here if needed)
#
# If they live in another module, import them at the top.
# -----------------------------------------------------------------------------

# ---------- Advisory helpers ----------
UR5E_REACH_M = 0.85  # ~850 mm reach (meters)

def _box_clearance_to_point_mm(box, p):
    """
    Min distance from point p=(x,y,z) mm to axis-aligned box.
    Negative => inside/penetrating.
    """
    cx, cy, cz, L, W, H = box["cx"], box["cy"], box["cz"], box["L"], box["W"], box["H"]
    min_x, max_x = cx - L/2.0, cx + L/2.0
    min_y, max_y = cy - W/2.0, cy + W/2.0
    min_z, max_z = cz - H/2.0, cz + H/2.0
    dx = max(min_x - p[0], 0, p[0] - max_x)
    dy = max(min_y - p[1], 0, p[1] - max_y)
    dz = max(min_z - p[2], 0, p[2] - max_z)
    if (min_x <= p[0] <= max_x) and (min_y <= p[1] <= max_y) and (min_z <= p[2] <= max_z):
        pen = min(p[0]-min_x, max_x-p[0], p[1]-min_y, max_y-p[1], p[2]-min_z, max_z-p[2])
        return -pen
    return (dx*dx + dy*dy + dz*dz) ** 0.5

def _recommend_waypoints_from_path(path_rad):
    """
    Heuristic: keep <= ~5 deg per-joint step and ~0.12 rad step length.
    Returns recommended minimum waypoint count.
    """
    import numpy as np, math
    if not path_rad or len(path_rad) < 2:
        return 10
    arr = np.array(path_rad)
    segs = np.linalg.norm(arr[1:] - arr[:-1], axis=1)  # joint-space step (rad)
    L = float(segs.sum())
    max_per_joint_deg = float(np.max(np.abs(np.rad2deg(arr[1:] - arr[:-1]))))
    n_len = max(2, int(math.ceil(L / 0.12)))
    n_joint = max(2, int(math.ceil(max_per_joint_deg / 5.0)) + 1)
    return max(n_len + 1, n_joint + 1)

def _recommend_knobs_on_failure(knobs):
    """Conservative tweaks that usually help without huge runtime."""
    k = dict(knobs or {})
    k.setdefault("max_iters", 2000)
    k.setdefault("shortcut_passes", 30)
    recs = []
    if k["max_iters"] < 4000:
        recs.append(("max_iters", max(3000, k["max_iters"] * 2)))
    if k.get("safety_margin_m", 0.003) > 0.002:
        recs.append(("safety_margin_m", 0.002))
    if k.get("bias", 0.2) < 0.35:
        recs.append(("bias", 0.35))
    if k.get("step_rad", 0.05) > 0.06:
        recs.append(("step_rad", 0.05))
    return recs

def _workspace_advice(start_q, goal_q, obstacles, fk_func):
    """
    Spatial suggestions based on reach and clearances near the goal.
    fk_func(q) -> (x,y,z) in mm.
    """
    adv = {"notes": [], "obstacle_moves": []}
    try:
        ee_goal = fk_func(goal_q)
    except Exception:
        ee_goal = None

    adv["notes"].append(
        f"UR5e nominal reach ≈ {int(UR5E_REACH_M*1000)} mm from base. "
        "Keep obstacle faces ≥ 120 mm away from the end-effector target if possible."
    )

    if ee_goal is not None and obstacles:
        min_c, min_box = None, None
        for b in obstacles:
            c = _box_clearance_to_point_mm(b, ee_goal)
            if (min_c is None) or (c < min_c):
                min_c, min_box = c, b
        if min_c is not None:
            if min_c < 0:
                adv["notes"].append(
                    "Goal appears to intersect an obstacle volume. Shift the goal or move that box."
                )
            elif min_c < 80:
                adv["notes"].append(
                    f"Goal clearance to the closest box is tight (~{int(min_c)} mm). "
                    "Increase spacing to ≥ 120–150 mm."
                )
                dx = ee_goal[0] - min_box["cx"]
                dy = ee_goal[1] - min_box["cy"]
                dz = ee_goal[2] - min_box["cz"]
                axis, sign = max(
                    [(abs(dx), ("x", 1 if dx >= 0 else -1)),
                     (abs(dy), ("y", 1 if dy >= 0 else -1)),
                     (abs(dz), ("z", 1 if dz >= 0 else -1))],
                    key=lambda t: t[0]
                )[1]
                shift = max(150 - int(min_c), 60)
                adv["obstacle_moves"].append(
                    f"Move the closest box ~{shift} mm along {'+' if sign>0 else '-'}{axis} away from the goal."
                )
    return adv
# ---------- /Advisory helpers ----------

# ---------- Utils ----------
def _to_deg(path_rad):
    return [[math.degrees(x) for x in q] for q in path_rad]

def _validate_shape(q, name):
    if not isinstance(q, (list, tuple)) or len(q) != 6:
        abort(400, f"{name} must be length-6 joint array (rad).")

def _parse_boxes_text(boxes_text):
    obstacles = []
    if boxes_text:
        for line in boxes_text.splitlines():
            if not line.strip():
                continue
            cx, cy, cz, L, W, H = [float(v) for v in line.split()]
            obstacles.append({"cx":cx,"cy":cy,"cz":cz,"L":L,"W":W,"H":H})
    return obstacles

def _parse_boxes_json(obstacles):
    out = []
    for i, b in enumerate(obstacles or []):
        try:
            cx = float(b["cx"]); cy = float(b["cy"]); cz = float(b["cz"])
            L  = float(b["L"]);  W  = float(b["W"]);  H  = float(b["H"])
        except Exception:
            abort(400, f"Obstacle {i} missing/invalid numeric fields.")
        out.append({"cx": cx, "cy": cy, "cz": cz, "L": L, "W": W, "H": H})
    return out
# ---------- /Utils ----------

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/plan", methods=["POST"])
def results():
    # HTML form
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
        return render_template("results.html", error=f"Bad input: {e}", advice=None)

    _validate_shape(start_q, "start")
    _validate_shape(goal_q, "goal")

    plan = plan_rrt_connect(start_q, goal_q, obstacles, knobs)

    # Build advice (best effort if FK present)
    advice = {"notes": [], "obstacle_moves": [], "recommend_waypoints": None}
    try:
        advice.update(_workspace_advice(start_q, goal_q, obstacles, fk_func=fk_ee_mm))
    except Exception:
        pass

    if not plan.get("success", False):
        advice["notes"].append("Planner failed. Try these knob changes:")
        for kname, v in _recommend_knobs_on_failure(knobs):
            advice["notes"].append(f"• {kname} → {v}")
        return render_template("results.html",
                               error=f"No path: {plan.get('reason','unknown')}",
                               advice=advice)

    path_rad = plan["path_rad"]
    path_rad = resample_path(path_rad, n_waypoints)
    path_deg = _to_deg(path_rad)

    try:
        advice["recommend_waypoints"] = _recommend_waypoints_from_path(path_rad)
    except Exception:
        advice["recommend_waypoints"] = None

    return render_template("results.html",
                           path_deg=path_deg,
                           n=len(path_deg),
                           iters=plan.get("iters"),
                           time_sec=plan.get("time_sec"),
                           reason=plan.get("reason"),
                           csv_download=True,
                           advice=advice)

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

@app.route("/api/plan", methods=["POST"])
def api_plan():
    body = request.get_json(silent=False, force=True)
    if body is None:
        abort(400, "JSON body required.")
    start_q = body.get("start_rad")
    goal_q  = body.get("goal_rad")
    _validate_shape(start_q, "start_rad")
    _validate_shape(goal_q,  "goal_rad")

    n_waypoints = int(body.get("n_waypoints", 20))
    knobs = body.get("knobs", {}) or {}
    obstacles = _parse_boxes_json(body.get("obstacles", []))
    return_degrees = bool(body.get("return_degrees", True))

    plan = plan_rrt_connect(start_q, goal_q, obstacles, knobs)

    # advice
    advice = None
    try:
        advice = _workspace_advice(start_q, goal_q, obstacles, fk_func=fk_ee_mm)
    except Exception:
        advice = None

    resp = {
        "success": bool(plan.get("success", False)),
        "iters": int(plan.get("iters", 0)),
        "time_sec": float(plan.get("time_sec", 0.0)),
        "reason": plan.get("reason", None),
        "advice": advice,
    }
    if not resp["success"]:
        if advice:
            resp["advice"]["knob_tweaks"] = _recommend_knobs_on_failure(knobs)
        return jsonify(resp), 200

    path_rad = plan["path_rad"]
    path_rad = resample_path(path_rad, n_waypoints)

    if return_degrees:
        resp["path"] = _to_deg(path_rad)
        resp["units"] = "deg"
    else:
        resp["path"] = path_rad
        resp["units"] = "rad"

    resp["num_waypoints"] = len(resp["path"])
    segs = []
    for a, b in zip(path_rad[:-1], path_rad[1:]):
        segs.append(float(np.linalg.norm(np.array(b) - np.array(a))))
    resp["joint_path_length_rad"] = float(sum(segs))
    try:
        resp.setdefault("advice", {})["recommend_waypoints"] = _recommend_waypoints_from_path(path_rad)
    except Exception:
        pass

    return jsonify(resp), 200

@app.route("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
