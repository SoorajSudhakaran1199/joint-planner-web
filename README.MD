# 🤖 Joint Path Planner (Web UI)

A simple, educational **robot joint-space path planner** with a clean web interface.  
Users can enter start/goal joint angles, define obstacles, and generate a waypoint path for a 6-axis robot.

This project was created to demonstrate:

- ✔️ basic motion-planning concepts  
- ✔️ simple UI for planning experiments  
- ✔️ exporting waypoints as CSV  
- ✔️ a lightweight Python + Flask web app structure

> Future extensions may include collision checking, goal-biased RRT, and path smoothing.

---

## 🚀 Demo Overview

The app lets you:

1️⃣ Enter **start joint values**  
2️⃣ Enter **target joint values**  
3️⃣ (Optional) define obstacles as boxes  
4️⃣ Choose number of waypoints  
5️⃣ View & download the generated joint path

The current baseline planner generates a **straight joint-space interpolation** so the UI always works — you can replace it later with your own planner.

---

## 🛠️ Tech Stack

- **Python 3**
- **Flask** (backend + routing)
- **HTML / CSS / JavaScript** (UI)
- **NumPy** (math utilities)

---

## 📂 Project Structure

```bash
joint-planner-web/
├── app.py              # Flask app + routes + planner
├── requirements.txt    # dependencies
├── Procfile            # deployment support (e.g., Heroku)
├── runtime.txt
├── templates/
│   ├── index.html      # main UI
│   └── results.html    # results page
└── static/             # CSS, images, etc.

Open in your browser: http://127.0.0.1:5001
pip install -r requirements.txt
python app.py