"""RoadScan core — ONNX Runtime inference (no torch), geotag, storage, PDF.
Light enough for free 512 MB hosts. Shared by server.py.
"""
from __future__ import annotations
import os, io, sqlite3, uuid, datetime, base64, shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags
import onnxruntime as ort

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "models" / "YOLOv8_Small_RDD.onnx"
DB = ROOT / "roadscan.db"
UPLOADS = ROOT / "uploads"; UPLOADS.mkdir(exist_ok=True)

CRACK_IDS = {0, 1, 2}                         # 3 crack sub-types -> "crack"
COLOR = {0: (60, 130, 232), 1: (63, 69, 214)}        # BGR: crack=orange, pothole=red
NAME = {0: "crack", 1: "pothole"}
CONF, IOU, IMGSZ = 0.25, 0.5, 640

_sess = None
def session():
    global _sess
    if _sess is None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1          # keep memory/CPU modest on free tiers
        _sess = ort.InferenceSession(str(WEIGHTS), sess_options=so,
                                     providers=["CPUExecutionProvider"])
    return _sess


# ---------- ONNX detection (letterbox -> infer -> decode -> NMS -> merge) ----------
def _letterbox(img, size=IMGSZ, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((size, size, 3), color, np.uint8)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(img, (nw, nh))
    return canvas, r, dw, dh

def _nms(boxes, scores, iou_th=IOU):
    idx = scores.argsort()[::-1]; keep = []
    while len(idx):
        i = idx[0]; keep.append(i); rest = idx[1:]
        if len(rest) == 0:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1); inter = w * h
        ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        ar = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (ai + ar - inter + 1e-9)
        idx = rest[iou < iou_th]
    return keep

def detect(img):
    """Return [(x1,y1,x2,y2,score,merged_cls)] in original-image coords (0=crack,1=pothole)."""
    lb, r, dw, dh = _letterbox(img)
    blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None]
    s = session()
    out = s.run(None, {s.get_inputs()[0].name: blob})[0][0].T        # (8400, 8)
    scores_all = out[:, 4:]
    cls = scores_all.argmax(1); conf = scores_all.max(1)
    m = conf >= CONF
    if not m.any():
        return []
    box = out[m, :4]; cls = cls[m]; conf = conf[m]
    cx, cy, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    xy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    xy[:, [0, 2]] = (xy[:, [0, 2]] - dw) / r
    xy[:, [1, 3]] = (xy[:, [1, 3]] - dh) / r
    keep = _nms(xy, conf)                        # class-agnostic (merge overlapping sub-types)
    H, W = img.shape[:2]; res = []
    for i in keep:
        mc = 0 if int(cls[i]) in CRACK_IDS else 1
        res.append((max(0, xy[i, 0]), max(0, xy[i, 1]), min(W, xy[i, 2]), min(H, xy[i, 3]),
                    float(conf[i]), mc))
    return res


# ---------- storage (SQLite, auto-upgrades to Supabase when secrets set) ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
USE_SB = bool(SUPABASE_URL and SUPABASE_KEY)
_sb = None
def _supabase():
    global _sb
    if _sb is None:
        from supabase import create_client
        _sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _sb

def _db():
    c = sqlite3.connect(str(DB))
    c.execute("""CREATE TABLE IF NOT EXISTS reports(
        id TEXT PRIMARY KEY, ts TEXT, lat REAL, lng REAL,
        crack INT, pothole INT, severity TEXT, conf REAL, image TEXT)""")
    return c

def save_report(r: dict):
    if USE_SB:
        try:
            _supabase().table("reports").insert(r).execute(); return
        except Exception as e:
            print("[storage] Supabase insert failed, using SQLite:", e)
    c = _db()
    c.execute("INSERT INTO reports VALUES(?,?,?,?,?,?,?,?,?)",
              (r["id"], r["ts"], r["lat"], r["lng"], r["crack"],
               r["pothole"], r["severity"], r["conf"], r["image"]))
    c.commit(); c.close()

def recent(n=50) -> list[dict]:
    if USE_SB:
        try:
            return _supabase().table("reports").select("*").order(
                "ts", desc=True).limit(n).execute().data
        except Exception as e:
            print("[storage] Supabase read failed, using SQLite:", e)
    c = _db()
    cols = ["id", "ts", "lat", "lng", "crack", "pothole", "severity", "conf", "image"]
    rows = c.execute(f"SELECT {','.join(cols)} FROM reports ORDER BY ts DESC LIMIT ?",
                     (n,)).fetchall()
    c.close()
    return [dict(zip(cols, r)) for r in rows]

def get_report(rid: str):
    if USE_SB:
        try:
            d = _supabase().table("reports").select("*").eq("id", rid).limit(1).execute().data
            return d[0] if d else None
        except Exception as e:
            print("[storage] Supabase get failed:", e)
    c = _db()
    cols = ["id", "ts", "lat", "lng", "crack", "pothole", "severity", "conf", "image"]
    row = c.execute(f"SELECT {','.join(cols)} FROM reports WHERE id=?", (rid,)).fetchone()
    c.close()
    return dict(zip(cols, row)) if row else None

def stats() -> dict:
    rs = recent(1000)
    return {"total": len(rs),
            "cracks": sum(r["crack"] for r in rs),
            "potholes": sum(r["pothole"] for r in rs),
            "hotspots": sum(1 for r in rs if r["severity"] == "High")}


# ---------- seed examples ----------
SEED = [
    ("seed1.jpg", 1, 2, 0.60, 31.5102, 74.3441, 5,    "High"),
    ("seed2.jpg", 2, 0, 0.52, 31.4698, 74.2712, 22,   "Low"),
    ("seed3.jpg", 0, 3, 0.82, 31.5601, 74.3294, 60,   "High"),
    ("seed4.jpg", 1, 2, 0.70, 31.5386, 74.3005, 180,  "High"),
    ("seed5.jpg", 3, 0, 0.58, 31.5305, 74.3489, 1200, "Low"),
]

def seed_if_empty():
    for i, (f, *_) in enumerate(SEED):
        dst = UPLOADS / f"RS-SEED{i+1}.jpg"
        if not dst.exists():
            try:
                shutil.copy(ROOT / "seed" / f, dst)
            except Exception:
                pass
    try:
        if recent(1):
            return
    except Exception:
        return
    now = datetime.datetime.now()
    for i, (f, ck, ph, cf, lat, lng, mins, sev) in enumerate(SEED):
        rid = f"RS-SEED{i+1}"
        ts = (now - datetime.timedelta(minutes=mins)).strftime("%Y-%m-%d %H:%M:%S")
        save_report(dict(id=rid, ts=ts, lat=lat, lng=lng, crack=ck, pothole=ph,
                         severity=sev, conf=cf, image=str(UPLOADS / f"{rid}.jpg")))


# ---------- PDF report ----------
def build_pdf(ids: list[str]) -> bytes:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    pdf = FPDF(format="A4"); pdf.set_auto_page_break(True, margin=15)
    made = 0
    for rid in ids:
        r = get_report(rid)
        if not r:
            continue
        made += 1
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18); pdf.set_text_color(24, 26, 30)
        pdf.cell(0, 10, "RoadScan  -  Road Damage Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10); pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, "Automated crack & pothole detection with geotagging",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(244, 180, 0); pdf.set_line_width(0.8)
        y = pdf.get_y() + 2; pdf.line(10, y, 200, y); pdf.ln(6)
        img = UPLOADS / f"{rid}.jpg"
        if img.exists():
            try:
                pdf.image(str(img), x=15, w=180)
            except Exception as e:
                print("[pdf] image embed failed:", e)
        pdf.ln(5)
        typ = "Pothole" if r["pothole"] > r["crack"] else ("Crack" if r["crack"] else "None detected")
        loc = (f"{float(r['lat']):.5f}, {float(r['lng']):.5f}"
               if r.get("lat") is not None else "Not available")
        rows = [("Report ID", rid), ("Damage type", typ),
                ("Cracks", r["crack"]), ("Potholes", r["pothole"]),
                ("Severity", r["severity"]), ("Confidence", f"{float(r['conf']):.2f}"),
                ("Location (lat, long)", loc), ("Reported", r["ts"])]
        for k, v in rows:
            pdf.set_font("Helvetica", "B", 11); pdf.set_text_color(107, 110, 118)
            pdf.cell(55, 8, str(k))
            pdf.set_font("Helvetica", "", 11); pdf.set_text_color(24, 26, 30)
            pdf.cell(0, 8, str(v), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if made == 0:
        pdf.add_page(); pdf.set_font("Helvetica", "", 12); pdf.cell(0, 10, "No reports found.")
    return bytes(pdf.output())


# ---------- EXIF GPS ----------
def exif_gps(pil_img: Image.Image):
    try:
        ex = pil_img._getexif() or {}
        gps = next((v for k, v in ex.items() if ExifTags.TAGS.get(k) == "GPSInfo"), None)
        if not gps:
            return None
        def dms(x): return float(x[0]) + float(x[1]) / 60 + float(x[2]) / 3600
        lat = dms(gps[2]) * (-1 if gps[1] in ("S", b"S") else 1)
        lng = dms(gps[4]) * (-1 if gps[3] in ("W", b"W") else 1)
        return round(lat, 6), round(lng, 6)
    except Exception:
        return None


# ---------- pipeline ----------
def analyze_bgr(img, pil_for_exif=None, lat_in=None, lng_in=None) -> dict:
    ck = ph = 0; top = 0.0; ttype = 0
    for x1, y1, x2, y2, s, mc in detect(img):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), COLOR[mc], 3)
        cv2.putText(img, f"{NAME[mc]} {s:.2f}", (int(x1), max(18, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR[mc], 2)
        if mc == 0: ck += 1
        else: ph += 1
        if s > top: top, ttype = s, mc

    gps = exif_gps(pil_for_exif) if pil_for_exif is not None else None
    if gps:
        lat, lng, src = gps[0], gps[1], "photo GPS"
    elif lat_in not in (None, "") and lng_in not in (None, ""):
        try:
            lat, lng, src = float(lat_in), float(lng_in), "manual"
        except ValueError:
            lat = lng = None; src = None
    else:
        lat = lng = None; src = None

    sev = "High" if (ph >= 2 or (ttype == 1 and top >= 0.6)) else \
          ("Medium" if top >= 0.6 else ("Low" if (ck or ph) else "None"))
    rid = "RS-" + uuid.uuid4().hex[:6].upper()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    img_path = str(UPLOADS / f"{rid}.jpg")
    cv2.imwrite(img_path, img)
    rec = dict(id=rid, ts=ts, lat=lat, lng=lng, crack=ck, pothole=ph,
               severity=sev, conf=round(top, 3), image=img_path)
    save_report(rec)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    out = dict(rec)
    out["annotated"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    out["source"] = src
    return out
