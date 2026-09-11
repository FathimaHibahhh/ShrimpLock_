import base64
import math
from typing import Dict, Any, Tuple
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="ShrimpLock: Photo Roast Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
)

# Colors in BGR for OpenCV
NEON_GREEN = (126, 255, 84)    # #54ff7e
NEON_RED = (78, 46, 255)       # #ff2e4e
NEON_YELLOW = (40, 215, 255)   # #ffd728
CYAN_ACCENT = (255, 220, 0)


def calculate_angle_with_vertical(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Computes absolute inclination angle in degrees relative to the vertical axis.
    p1 is the top point (e.g. ear or shoulder), p2 is bottom point (shoulder or hip).
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    angle_rad = math.atan2(abs(dx), abs(dy) + 1e-6)
    return math.degrees(angle_rad)


def calculate_tilt_angle(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Computes absolute horizontal tilt angle in degrees between two bilateral landmarks.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(abs(dy), abs(dx) + 1e-6)
    return math.degrees(angle_rad)


@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_posture(image: UploadFile = File(...)) -> Dict[str, Any]:
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await image.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode image.")

    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(img_rgb)

    if not results.pose_landmarks:
        raise HTTPException(
            status_code=422,
            detail="No human detected! Sit in view, adjust lighting, or make sure your upper body is visible."
        )

    landmarks = results.pose_landmarks.landmark

    def get_coords(idx: int) -> Tuple[int, int, float]:
        lm = landmarks[idx]
        return int(lm.x * w), int(lm.y * h), lm.visibility

    # Landmarks extraction
    left_ear, right_ear = get_coords(7), get_coords(8)
    left_shoulder, right_shoulder = get_coords(11), get_coords(12)
    left_hip, right_hip = get_coords(23), get_coords(24)

    # Determine dominant profile or coronal view
    left_vis = (left_ear[2] + left_shoulder[2] + left_hip[2]) / 3.0
    right_vis = (right_ear[2] + right_shoulder[2] + right_hip[2]) / 3.0

    # Shoulder horizontal tilt
    shoulder_tilt = calculate_tilt_angle(
        (left_shoulder[0], left_shoulder[1]),
        (right_shoulder[0], right_shoulder[1])
    )

    # Neck & Torso angles based on dominant side
    if left_vis >= right_vis:
        chosen_ear = (left_ear[0], left_ear[1])
        chosen_shoulder = (left_shoulder[0], left_shoulder[1])
        chosen_hip = (left_hip[0], left_hip[1])
    else:
        chosen_ear = (right_ear[0], right_ear[1])
        chosen_shoulder = (right_shoulder[0], right_shoulder[1])
        chosen_hip = (right_hip[0], right_hip[1])

    neck_angle = calculate_angle_with_vertical(chosen_ear, chosen_shoulder)
    torso_angle = calculate_angle_with_vertical(chosen_shoulder, chosen_hip)

    # Overall posture classification
    # Deep slouch occurs when either shoulder tilt, neck forward drop, or torso lean is severe
    is_shrimp = (shoulder_tilt > 20.0) or (neck_angle > 28.0) or (torso_angle > 25.0)
    is_pisa = (7.0 <= shoulder_tilt <= 20.0) or (15.0 <= neck_angle <= 28.0) or (14.0 <= torso_angle <= 25.0)

    if is_shrimp:
        state = "SHRIMP"
        title = "BOILED PRAWN DETECTED"
        roast_comment = "Faaahhh! Critical anatomy error! Backbone deleted. You are legally classified as a shrimp."
        sound_trigger = "faahhh"
        accent_color = NEON_RED
    elif is_pisa:
        state = "PISA"
        title = "TOWER OF PISA MODE"
        roast_comment = "Warning: Structural integrity failing. Your neck has filed an official complaint with HR."
        sound_trigger = "wood_creak"
        accent_color = NEON_YELLOW
    else:
        state = "HUMAN"
        title = "ACCEPTABLE VERTEBRATE"
        roast_comment = "Look at you, showing off with an intact spine. You may temporarily exist as a mammal."
        sound_trigger = "angelic"
        accent_color = NEON_GREEN

    # Annotate image
    annotated = img_bgr.copy()
    thickness = max(2, int(min(h, w) * 0.005))
    joint_radius = max(4, int(min(h, w) * 0.01))

    # Draw Connections
    connections = [
        ((left_shoulder[0], left_shoulder[1]), (right_shoulder[0], right_shoulder[1])),
        (chosen_ear, chosen_shoulder),
        (chosen_shoulder, chosen_hip),
    ]

    for pt1, pt2 in connections:
        cv2.line(annotated, pt1, pt2, accent_color, thickness, cv2.LINE_AA)

    # Draw vertical reference guide line at shoulder
    cv2.line(
        annotated,
        chosen_shoulder,
        (chosen_shoulder[0], chosen_shoulder[1] - int(min(h, w) * 0.18)),
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    # Draw Joint Points
    joints = [
        (left_ear[0], left_ear[1]),
        (right_ear[0], right_ear[1]),
        (left_shoulder[0], left_shoulder[1]),
        (right_shoulder[0], right_shoulder[1]),
        (left_hip[0], left_hip[1]),
        (right_hip[0], right_hip[1])
    ]

    for x, y in joints:
        cv2.circle(annotated, (x, y), joint_radius, accent_color, -1, cv2.LINE_AA)
        cv2.circle(annotated, (x, y), joint_radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

    # HUD Stamp
    hud_text = f"{state} | Neck: {neck_angle:.1f} deg | Tilt: {shoulder_tilt:.1f} deg"
    font_scale = max(0.5, min(w, h) * 0.0008)
    cv2.putText(
        annotated,
        hud_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        accent_color,
        2,
        cv2.LINE_AA
    )

    # Encode to base64
    _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    base64_image = base64.b64encode(buffer).decode("utf-8")

    return {
        "posture_state": state,
        "shoulder_angle": round(shoulder_tilt, 1),
        "neck_angle": round(neck_angle, 1),
        "torso_angle": round(torso_angle, 1),
        "title": title,
        "roast_comment": roast_comment,
        "sound_trigger": sound_trigger,
        "annotated_image_base64": f"data:image/jpeg;base64,{base64_image}"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)