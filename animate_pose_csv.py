import argparse, os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---- MediaPipe landmark names (33) ----
LM = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
    "left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
    "left_heel","right_heel","left_foot_index","right_foot_index"
]
IDX = {name:i for i,name in enumerate(LM)}

# Stick-figure edges (pairs of landmark names)
EDGES = [
    # torso
    ("left_shoulder","right_shoulder"), ("left_hip","right_hip"),
    ("left_shoulder","left_hip"), ("right_shoulder","right_hip"),
    # left arm
    ("left_shoulder","left_elbow"), ("left_elbow","left_wrist"),
    ("left_wrist","left_index"), ("left_wrist","left_thumb"), ("left_wrist","left_pinky"),
    # right arm
    ("right_shoulder","right_elbow"), ("right_elbow","right_wrist"),
    ("right_wrist","right_index"), ("right_wrist","right_thumb"), ("right_wrist","right_pinky"),
    # left leg
    ("left_hip","left_knee"), ("left_knee","left_ankle"), ("left_ankle","left_heel"), ("left_ankle","left_foot_index"),
    # right leg
    ("right_hip","right_knee"), ("right_knee","right_ankle"), ("right_ankle","right_heel"), ("right_ankle","right_foot_index"),
    # head
    ("nose","left_eye"), ("nose","right_eye"), ("left_eye","left_ear"), ("right_eye","right_ear"),
    ("mouth_left","mouth_right"), ("nose","mouth_left"), ("nose","mouth_right")
]

def load_pose_csv(path):
    df = pd.read_csv(path)
    
    cols = {c.lower():c for c in df.columns}
   
    if "x_m" in df.columns and "y_m" in df.columns and "z_m" in df.columns:
        xcol,ycol,zcol = "X_m","Y_m","Z_m"
    else:
        xcol,ycol,zcol = cols.get("x","x"), cols.get("y","y"), cols.get("z","z")
    tcol = cols.get("ts","ts")
    jcol = cols.get("joint","joint")
    if not all([xcol in df, ycol in df, zcol in df, tcol in df, jcol in df]):
        raise ValueError("CSV must contain ts, joint, and x/y/z (or X_m/Y_m/Z_m) columns.")
    
    df = df[[tcol,jcol,xcol,ycol,zcol]].rename(columns={tcol:"ts", jcol:"joint", xcol:"x", ycol:"y", zcol:"z"})
    
    df = df.sort_values("ts")
    
    frames = []
    for ts, g in df.groupby("ts"):
       
        arr = np.full((33,3), np.nan, dtype=np.float32)
        for _, row in g.iterrows():
            if row["joint"] in IDX:
                arr[IDX[row["joint"]]] = [row["x"], row["y"], row["z"]]
        frames.append((ts, arr))
    return frames

def compute_bounds(frames, margin=0.15):
    pts = np.concatenate([a for _,a in frames], axis=0)
    pts = pts[~np.isnan(pts).any(1)]
    if len(pts)==0:
        return (-1,1), (-1,1), (-1,1)
    xmn,xmx = float(np.min(pts[:,0])), float(np.max(pts[:,0]))
    ymn,ymx = float(np.min(pts[:,1])), float(np.max(pts[:,1]))
    zmn,zmx = float(np.min(pts[:,2])), float(np.max(pts[:,2]))

    cx,cy,cz = (xmn+xmx)/2, (ymn+ymx)/2, (zmn+zmx)/2
    r = max(xmx-xmn, ymx-ymn, zmx-zmn)/2
    r *= (1+margin)
    return (cx-r, cx+r), (cy-r, cy+r), (cz-r, cz+r)

def animate(frames, out_path=None, fps=30, step=1, invert_y=False, invert_z=False):
    
    frames_used = frames[::max(1,step)]
    xlim, ylim, zlim = compute_bounds(frames_used)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=15, azim=-70) 

    lines = []
    for _ in EDGES:
        ln, = ax.plot([], [], [], lw=2)
        lines.append(ln)
    scat = ax.scatter([], [], [], s=10)

    def draw_frame(arr):
        
        A = arr.copy()
        if invert_y: A[:,1] *= -1
        if invert_z: A[:,2] *= -1
        
        for i,(a,b) in enumerate(EDGES):
            ia, ib = IDX[a], IDX[b]
            if np.any(np.isnan(A[[ia,ib]])): 
                lines[i].set_data([], [])
                lines[i].set_3d_properties([])
            else:
                xs = [A[ia,0], A[ib,0]]
                ys = [A[ia,1], A[ib,1]]
                zs = [A[ia,2], A[ib,2]]
                lines[i].set_data(xs, ys)
                lines[i].set_3d_properties(zs)
        
        good = ~np.isnan(A).any(1)
        scat._offsets3d = (A[good,0], A[good,1], A[good,2])

    def init():
        draw_frame(np.full((33,3), np.nan))
        return lines + [scat]

    def update(k):
        ts, arr = frames_used[k]
        draw_frame(arr)
        ax.set_title(f"t = {ts:.2f}s   frame {k+1}/{len(frames_used)}")
        return lines + [scat]

    anim = FuncAnimation(fig, update, init_func=init, frames=len(frames_used), interval=1000/fps, blit=False)

    if out_path:
        try:
            writer = FFMpegWriter(fps=fps, bitrate=4000)
            anim.save(out_path, writer=writer)
            print(f"[✓] Saved animation to {out_path}")
        except Exception as e:
            print(f"[!] Could not save MP4 (need ffmpeg?). Showing live window instead. Error: {e}")
            plt.show()
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to pose CSV")
    ap.add_argument("--out", default="", help="Optional output MP4 path")
    ap.add_argument("--fps", type=int, default=30, help="Playback FPS")
    ap.add_argument("--step", type=int, default=1, help="Use every Nth row for speed")
    ap.add_argument("--invert_y", action="store_true", help="Flip Y axis")
    ap.add_argument("--invert_z", action="store_true", help="Flip Z axis")
    args = ap.parse_args()

    frames = load_pose_csv(args.csv)
    if not frames:
        raise SystemExit("No frames found in CSV.")
    out = args.out if args.out else None
    animate(frames, out_path=out, fps=args.fps, step=args.step, invert_y=args.invert_y, invert_z=args.invert_z)

if __name__ == "__main__":
    main()
