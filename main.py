import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans
from ultralytics import YOLO
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram
)
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from inference import get_model
import threading
import time
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display # For Jupyter/IPython display

# === CONFIG ===
CONFIG = SoccerPitchConfiguration()
PLAYER_ID = 1
REFEREE_ID = 2
BALL_ID = 0
MAX_CROPS = 100
HUE_TOLERANCE = 20  # Degrees
POSSESSION_DISTANCE_THRESHOLD = 300  # Pixels
OUTLIER_DISTANCE_THRESHOLD = 0.2    # For color outlier detection
UNCLASSIFIED_CLASS_ID = 99          # ID for players with outlier colors
COLOR_TOLERANCE = 30                # For Voronoi pixel analysis
VORONOI_CALC_SKIP_FRAMES = 5        # Calculate Voronoi every 6th frame
REF_TEAM_0_BGR = np.array([255, 191, 0], dtype=np.uint8)   # Blue reference
REF_TEAM_1_BGR = np.array([147, 20, 255], dtype=np.uint8)  # Pink reference
NEUTRAL_COLOR_BGR = np.array([200, 200, 200], dtype=np.uint8)  # Light gray

# === PATHS ===
STITCHED_STREAM_URL = "http://localhost:5000/api/stitched/stream"
OUTPUT_PATH = "C:/Users/tonyi/OneDrive/Documents/compvision/output/pass_accuracy.mp4"

# Base paths for output, will be appended with "_half1", "_half2", "_overall"
DOTPLOT_OUTPUT_BASE = "C:/Users/tonyi/OneDrive/Documents/compvision/output/pass_dotplot"
HEATMAP_OUTPUT_BASE = "C:/Users/tonyi/OneDrive/Documents/compvision/output/pass_heatmap"
VORONOI_HEATMAP_BASE = "C:/Users/tonyi/OneDrive/Documents/compvision/output/voronoi_heatmap"

PLAYER_MODEL_PATH = "C:/Users/tonyi/OneDrive/Documents/compvision/models/best.pt"
FIELD_MODEL_ID = "football-field-detection-f07vi/15"
ROBOFLOW_API_KEY = "zEZIynLb2bpdcVTfLZ"

# === GLOBAL STATE FOR HALF-TIME CONTROL ===
processing_paused = False
current_half = 1
first_half_end_frame_idx = -1 # To mark when first half ended for reporting

# === POSSESSION TRACKING ===
total_active_frames = 0          # Frames during active play (excludes halftime)
team0_possession_frames = 0      # Frames where Team 0 had possession
team1_possession_frames = 0      # Frames where Team 1 had possession
no_possession_frames = 0         # Frames with no possession (ball not detected)

# === GLOBAL STATE FOR HALF-TIME CONTROL ===
team0_possession_frames_half1 = 0
team1_possession_frames_half1 = 0
no_possession_frames_half1 = 0
total_active_frames_half1 = 0

team0_possession_frames_half2 = 0
team1_possession_frames_half2 = 0
no_possession_frames_half2 = 0
total_active_frames_half2 = 0

# === GREEN HUE FILTER FUNCTIONS ===
def get_dominant_grass_hue(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h *= 2
    mask = (s >= 50) & (v >= 50)
    h_masked = h[mask]
    hist, bin_edges = np.histogram(h_masked, bins=180, range=(0, 360))
    return bin_edges[np.argmax(hist)]

def center_weighted_color(crop, h_low, h_high):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h *= 2  # Scale hue from 0-180 to 0-360
    h /= 360  # Normalize hue to 0-1
    s /= 255  # Normalize saturation to 0-1
    v /= 255  # Normalize value to 0-1
    
    # Create mask for green pixels (to exclude)
    green_mask = (
        (h >= h_low / 360) & (h <= h_high / 360) &
        (s >= 0.2) & (s <= 1.0) &
        (v >= 0.2) & (v <= 1.0)
    )  # Added missing closing parenthesis here
    
    # Invert to get mask for non-green pixels (to include)
    mask = ~green_mask

    if not np.any(mask):
        return None

    h_, w_ = mask.shape
    y, x = np.ogrid[:h_, :w_]
    center_y, center_x = h_ // 2, w_ // 2
    sigma = min(h_, w_) / 4
    weights = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
    weights = weights * mask

    total_weight = np.sum(weights)
    if total_weight == 0:
        return None

    h_avg = np.sum(h * weights) / total_weight
    s_avg = np.sum(s * weights) / total_weight
    v_avg = np.sum(v * weights) / total_weight
    return (h_avg, s_avg, v_avg)

# === LOAD MODELS ===
PLAYER_MODEL = YOLO(PLAYER_MODEL_PATH)
FIELD_MODEL = get_model(FIELD_MODEL_ID, api_key=ROBOFLOW_API_KEY)

# === STATE ===
tracker = sv.ByteTrack()
tracker.reset()
crops, crop_tids, crop_colors = [], [], []
first_training_frame = None
static_transformer = None
goalkeeper_tracker_ids = []
goalkeeper_id_to_team = {}
RADAR_WIDTH, RADAR_HEIGHT = None, None
BASE_PITCH_IMAGE = None

# Possession tracking (per-half and overall)
successful_passes_half1 = defaultdict(int)
unsuccessful_passes_half1 = defaultdict(int)
successful_passes_half2 = defaultdict(int)
unsuccessful_passes_half2 = defaultdict(int)
successful_passes_overall = defaultdict(int)
unsuccessful_passes_overall = defaultdict(int)

possession_event_log = [] # This can be used to analyze full match events

# Pass origin positions (per-half and overall)
successful_pass_start_positions_team0_half1 = []
successful_pass_start_positions_team1_half1 = []
successful_pass_start_pixels_team0_half1 = []
successful_pass_start_pixels_team1_half1 = []

successful_pass_start_positions_team0_half2 = []
successful_pass_start_positions_team1_half2 = []
successful_pass_start_pixels_team0_half2 = []
successful_pass_start_pixels_team1_half2 = []

successful_pass_start_positions_team0_overall = []
successful_pass_start_positions_team1_overall = []
successful_pass_start_pixels_team0_overall = []
successful_pass_start_pixels_team1_overall = []

current_possessor_id = None
current_possessor_team = None

# Voronoi state (per-half and overall)
team_0_control_sum_half1 = None
team_1_control_sum_half1 = None
voronoi_frame_count_half1 = 0

team_0_control_sum_half2 = None
team_1_control_sum_half2 = None
voronoi_frame_count_half2 = 0

team_0_control_sum_overall = None
team_1_control_sum_overall = None
voronoi_frame_count_overall = 0

found_team_colors = False
dynamic_team0_bgr = None
dynamic_team1_bgr = None

def reset_half_stats():
    global current_possessor_id, current_possessor_team
    global team0_possession_frames, team1_possession_frames, no_possession_frames, total_active_frames
    
    # Reset possession for the new half, but not the overall counts
    current_possessor_id = None
    current_possessor_team = None
    
    # Reset possession counters for the new half
    team0_possession_frames = 0
    team1_possession_frames = 0
    no_possession_frames = 0
    total_active_frames = 0

def get_possession_percentages():
    # Only count frames where a team had possession (exclude no_possession_frames)
    frames_with_possession = team0_possession_frames + team1_possession_frames
    
    if frames_with_possession == 0:
        return 0.0, 0.0  # Avoid division by zero
    
    team0_percent = (team0_possession_frames / frames_with_possession) * 100
    team1_percent = (team1_possession_frames / frames_with_possession) * 100
    
    return team0_percent, team1_percent  # Now these will always sum to 100%

def detect_goalkeepers(frame, current_half):
    """Detect goalkeepers using current frame positions only"""
    global static_transformer, goalkeeper_tracker_ids, goalkeeper_id_to_team, tracker
    
    # Get player detections
    yolo_result = PLAYER_MODEL.predict(frame, verbose=False)[0]
    dets = sv.Detections.from_ultralytics(yolo_result).with_nms(0.5, class_agnostic=True)
    tracked = tracker.update_with_detections(dets[dets.class_id == PLAYER_ID])
    
    if len(tracked.xyxy) < 2:  # Need at least 2 players
        return
    
    # Transform current positions to pitch coordinates
    current_positions = static_transformer.transform_points(
        tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
    x_coords = current_positions[:, 0]
    
    # Sort by x-coordinate to find goalkeepers
    sorted_idx = np.argsort(x_coords)
    goalkeeper_tracker_ids = [
        tracked.tracker_id[sorted_idx[0]],  # Leftmost
        tracked.tracker_id[sorted_idx[-1]]  # Rightmost
    ]
    
    # Get current team assignments (from classifier)
    team0_players = []
    team1_players = []
    
    for i, tid in enumerate(tracked.tracker_id):
        if tracked.class_id[i] == 0:
            team0_players.append(current_positions[i])
        elif tracked.class_id[i] == 1:
            team1_players.append(current_positions[i])
    
    # Calculate current centroids
    team0_centroid = np.mean(team0_players, axis=0) if team0_players else None
    team1_centroid = np.mean(team1_players, axis=0) if team1_players else None
    
    # Assign goalkeepers to nearest team centroid
    for gid in goalkeeper_tracker_ids:
        gidx = np.where(tracked.tracker_id == gid)[0][0]
        gpos = current_positions[gidx]
        
        if team0_centroid is not None and team1_centroid is not None:
            dist0 = np.linalg.norm(gpos - team0_centroid)
            dist1 = np.linalg.norm(gpos - team1_centroid)
            goalkeeper_id_to_team[gid] = 0 if dist0 < dist1 else 1
        else:
            # Fallback to classifier's team assignment
            goalkeeper_id_to_team[gid] = tracked.class_id[gidx]
    
    print(f"Goalkeepers detected for half {current_half}:")
    for gid in goalkeeper_tracker_ids:
        team = goalkeeper_id_to_team.get(gid, "unknown")
        print(f"  Tracker ID {gid} assigned to team {team}")

# --- REPORTING FUNCTIONS (DEFINED BEFORE THEY ARE CALLED) ---

def generate_and_display_dot_plot(pass_positions, output_path, team_name, dot_color_sv):
    pitch_for_dotplot = draw_pitch(CONFIG)
    if pitch_for_dotplot.shape[2] == 4:
        pitch_for_dotplot = cv2.cvtColor(pitch_for_dotplot, cv2.COLOR_RGBA2BGR)

    if pass_positions:
        pitch_for_dotplot = draw_points_on_pitch(
            CONFIG,
            xy=np.array(pass_positions),
            face_color=dot_color_sv,
            edge_color=sv.Color.BLACK,
            radius=8,
            pitch=pitch_for_dotplot
        )

    cv2.putText(pitch_for_dotplot, f"{team_name} Pass Origin Dot Plot", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(output_path, pitch_for_dotplot)
    print(f"✅ {team_name} Pass origin dot plot saved to: {output_path}")
    

def generate_and_save_heatmap(pass_pixel_positions, output_path, team_name, title_color_rgb, base_pitch_image):
    w = base_pitch_image.shape[1]
    h = base_pitch_image.shape[0]

    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    base_pitch_image_rgb = cv2.cvtColor(base_pitch_image, cv2.COLOR_BGR2RGB)
    ax.imshow(base_pitch_image_rgb, extent=[0, w, 0, h], origin='lower', aspect='auto', zorder=-1)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_xticks([])
    ax.set_yticks([])

    if pass_pixel_positions:
        pass_pixel_positions = np.array(pass_pixel_positions)
        sns.kdeplot(
            x=pass_pixel_positions[:,0],
            y=pass_pixel_positions[:,1],
            fill=True,
            cmap='viridis',
            alpha=0.6,
            levels=10,
            ax=ax,
            zorder=0
        )
    else:
        print(f"⚠️ No pass data for {team_name} - generating empty heatmap")

    ax.set_title(f"{team_name} Successful Pass Origin Heatmap", color=title_color_rgb)
    plt.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"✅ {team_name} Pass origin heatmap saved to: {output_path}")
    

# Function to generate all reports
def generate_all_reports(
    pass_positions_team0,
    pass_positions_team1,
    pass_pixels_team0,
    pass_pixels_team1,
    team0_control_sum_data,
    team1_control_sum_data,
    voronoi_frame_count_data,
    suffix
):
    print(f"\n--- Generating Pass Visualizations for {suffix} ---")
    team0_color_sv = sv.Color.from_hex("00BFFF")
    team1_color_sv = sv.Color.from_hex("FF1493")
    team0_title_color_rgb = (0/255, 191/255, 255/255)
    team1_title_color_rgb = (255/255, 20/255, 147/255)

    generate_and_display_dot_plot(pass_positions_team0, f"{DOTPLOT_OUTPUT_BASE}_team0_{suffix}.png", f"Team 1 ({suffix})", team0_color_sv)
    generate_and_display_dot_plot(pass_positions_team1, f"{DOTPLOT_OUTPUT_BASE}_team1_{suffix}.png", f"Team 2 ({suffix})", team1_color_sv)
    generate_and_save_heatmap(pass_pixels_team0, f"{HEATMAP_OUTPUT_BASE}_team0_{suffix}.png", f"Team 1 ({suffix})", team0_title_color_rgb, BASE_PITCH_IMAGE)
    generate_and_save_heatmap(pass_pixels_team1, f"{HEATMAP_OUTPUT_BASE}_team1_{suffix}.png", f"Team 2 ({suffix})", team1_title_color_rgb, BASE_PITCH_IMAGE)

    print(f"\n--- Generating Voronoi Heatmap for {suffix} ---")
    if voronoi_frame_count_data > 0 and found_team_colors:
        dynamic_team0_bgr_f = dynamic_team0_bgr.astype(np.float32) / 255.0
        dynamic_team1_bgr_f = dynamic_team1_bgr.astype(np.float32) / 255.0
        neutral_color_f = NEUTRAL_COLOR_BGR.astype(np.float32) / 255.0

        total_control = team0_control_sum_data + team1_control_sum_data
        total_control_safe = np.where(total_control == 0, 1, total_control)
        team0_ratio = team0_control_sum_data / total_control_safe
        team1_ratio = team1_control_sum_data / total_control_safe

        INTENSITY_FACTOR = 2.0
        heatmap_image = BASE_PITCH_IMAGE.copy().astype(np.float32) / 255.0

        dom_strength_team0 = np.clip((team0_ratio - 0.5) * INTENSITY_FACTOR, 0, 1)
        dom_strength_team1 = np.clip((team1_ratio - 0.5) * INTENSITY_FACTOR, 0, 1)

        mask_team0_dominates = team0_ratio > team1_ratio
        heatmap_image[mask_team0_dominates] = (
            neutral_color_f * (1 - dom_strength_team0[mask_team0_dominates, np.newaxis]) +
            dynamic_team0_bgr_f * dom_strength_team0[mask_team0_dominates, np.newaxis]
        )

        mask_team1_dominates = team1_ratio > team0_ratio
        heatmap_image[mask_team1_dominates] = (
            neutral_color_f * (1 - dom_strength_team1[mask_team1_dominates, np.newaxis]) +
            dynamic_team1_bgr_f * dom_strength_team1[mask_team1_dominates, np.newaxis]
        )

        mask_even_or_no_control = (team0_ratio == team1_ratio) | (total_control == 0)
        heatmap_image[mask_even_or_no_control] = neutral_color_f

        # Blend the Voronoi overlay with the base pitch image to keep lines visible
        voronoi_overlay = (heatmap_image * 255).astype(np.uint8)
        average_voronoi_image = cv2.addWeighted(BASE_PITCH_IMAGE, 0.3, voronoi_overlay, 0.7, 0)

        output_path = f"{VORONOI_HEATMAP_BASE}_{suffix}.png"
        cv2.imwrite(output_path, average_voronoi_image)
        print(f"✅ Voronoi heatmap saved to: {output_path}")
        
    else:
        print(f"⚠️ Could not generate Voronoi heatmap for {suffix} - insufficient data or team colors not found")

# === UPDATE on_stop_half_time FUNCTION ===
def on_stop_half_time(frame_idx_param):
    global processing_paused, current_half, first_half_end_frame_idx
    global team0_possession_frames_half1, team1_possession_frames_half1
    global no_possession_frames_half1, total_active_frames_half1
    
    if not processing_paused and current_half == 1:
        processing_paused = True
        first_half_end_frame_idx = frame_idx_param
        
        # CAPTURE FIRST HALF STATS
        team0_possession_frames_half1 = team0_possession_frames
        team1_possession_frames_half1 = team1_possession_frames
        no_possession_frames_half1 = no_possession_frames
        total_active_frames_half1 = total_active_frames
        
        print("Half-time! Processing paused. Reports will be generated at the end of the match.")

def on_start_second_half(frame):
    global processing_paused, current_half
    global goalkeeper_tracker_ids, goalkeeper_id_to_team
    
    if processing_paused and current_half == 1:
        # Clear previous goalkeeper assignments
        goalkeeper_tracker_ids = []
        goalkeeper_id_to_team = {}
        
        # Force redetection for second half
        detect_goalkeepers(frame, current_half=2)
        
        processing_paused = False
        current_half = 2
        print("Starting 2nd Half! Processing resumed.")
        print(f"Second half goalkeepers: {goalkeeper_tracker_ids}")
        reset_half_stats()

# === TRAINING LOOP ===
print("--- Training Player Classifier ---")
print("Connecting to stitched stream...")

# Connect to stitched stream
stitched_cap = cv2.VideoCapture(STITCHED_STREAM_URL)
if not stitched_cap.isOpened():
    print("❌ Failed to connect to stitched stream")
    exit()

print("✅ Connected to stitched stream")

# Get frames from stitched stream
frame_count = 0
while frame_count < MAX_CROPS:
    ret, frame = stitched_cap.read()
    if not ret:
        print("❌ Failed to read frame from stitched stream")
        continue
    
    frame_count += 1
    if first_training_frame is None:
        first_training_frame = frame.copy()
        dom_hue = get_dominant_grass_hue(frame)
        h_low = max(0, dom_hue - HUE_TOLERANCE)
        h_high = min(360, dom_hue + HUE_TOLERANCE)

        field_result = FIELD_MODEL.infer(frame, confidence=0.3)[0]
        kpts = sv.KeyPoints.from_inference(field_result)
        mask = kpts.confidence[0] > 0.5
        frame_pts = kpts.xy[0][mask]
        pitch_pts = np.array(CONFIG.vertices)[mask]
        static_transformer = ViewTransformer(source=frame_pts, target=pitch_pts)

        BASE_PITCH_IMAGE = draw_pitch(CONFIG)
        RADAR_HEIGHT, RADAR_WIDTH = BASE_PITCH_IMAGE.shape[:2]

        # Initialize Voronoi accumulation grids for both halves and overall
        team_0_control_sum_half1 = np.zeros((RADAR_HEIGHT, RADAR_WIDTH), dtype=np.float32)
        team_1_control_sum_half1 = np.zeros((RADAR_HEIGHT, RADAR_WIDTH), dtype=np.float32)
        team_0_control_sum_half2 = np.zeros((RADAR_HEIGHT, RADAR_WIDTH), dtype=np.float32)
        team_1_control_sum_half2 = np.zeros((RADAR_HEIGHT, RADAR_WIDTH), dtype=np.float32)
        team_0_control_sum_overall = np.zeros((RADAR_HEIGHT, RADAR_WIDTH), dtype=np.float32)
        team_1_control_sum_overall = np.zeros((RADAR_HEIGHT, RADAR_WIDTH), dtype=np.float32)

        yolo_result = PLAYER_MODEL.predict(frame, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(yolo_result).with_nms(0.5, class_agnostic=True)
        tracked_first = tracker.update_with_detections(dets[dets.class_id == PLAYER_ID])
        if tracked_first.xyxy.shape[0] < 2:
            continue
        positions = static_transformer.transform_points(tracked_first.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        x_coords = positions[:, 0]
        sorted_idx = np.argsort(x_coords)
        goalkeeper_tracker_ids = [
            tracked_first.tracker_id[sorted_idx[0]],
            tracked_first.tracker_id[sorted_idx[-1]]
        ]

    yolo_result = PLAYER_MODEL.predict(frame, verbose=False)[0]
    dets = sv.Detections.from_ultralytics(yolo_result).with_nms(0.5, class_agnostic=True)
    tracked = tracker.update_with_detections(dets[dets.class_id == PLAYER_ID])

    for tid, box in zip(tracked.tracker_id, tracked.xyxy):
        if len(crops) >= MAX_CROPS:
            break
        crop = sv.crop_image(frame, box)
        color = center_weighted_color(crop, h_low, h_high)
        if color is None:
            continue
        crops.append(crop)
        crop_colors.append(color)
        crop_tids.append(tid)
    if len(crops) >= MAX_CROPS:
        break

# === KMEANS TRAINING ===
print("--- Training KMeans Classifier ---")
color_array = np.array(crop_colors, dtype=np.float64)
kmeans = KMeans(n_clusters=2, n_init="auto").fit(color_array)
thread_preds = [None] * len(crops)

def classify_crop(i, crop):
    color = center_weighted_color(crop, h_low, h_high)
    thread_preds[i] = kmeans.predict([color])[0] if color is not None else 0

threads = [threading.Thread(target=classify_crop, args=(i, crop)) for i, crop in enumerate(crops)]
[t.start() for t in threads]
[t.join() for t in threads]

label_map = defaultdict(list)
for tid, label in zip(crop_tids, thread_preds):
    label_map[tid].append(label)
track_id_to_team = {tid: max(set(l), key=l.count) for tid, l in label_map.items()}

# === GOALKEEPER TEAM ASSIGNMENT ===
positions_dict = {tid: pos for tid, pos in zip(tracked_first.tracker_id, positions)}
team_positions = defaultdict(list)
for tid, team in track_id_to_team.items():
    if tid in positions_dict:
        team_positions[team].append(positions_dict[tid])
centroids = {team: np.mean(pts, axis=0) for team, pts in team_positions.items()}
for gid in goalkeeper_tracker_ids:
    gpos = positions_dict.get(gid)
    if gpos is not None:
        dists = {team: np.linalg.norm(gpos - center) for team, center in centroids.items()}
        goalkeeper_id_to_team[gid] = min(dists, key=dists.get)

print("Initial Goalkeeper Assignments:")
for gid in goalkeeper_tracker_ids:
    print(f"  Tracker ID {gid} assigned to team {goalkeeper_id_to_team.get(gid, 'unknown')}")

# === OUTPUT INIT ===
dummy = draw_pitch(CONFIG)
h, w = dummy.shape[:2]
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))

# Get frame dimensions from stitched stream
stitched_width = int(stitched_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
stitched_height = int(stitched_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Stitched stream dimensions: {stitched_width}x{stitched_height}")

# Create OpenCV window (buttons removed due to QT dependency)
WINDOW_NAME = "Soccer Analysis"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# === MAIN PROCESSING LOOP ===
print("--- Processing Stitched Stream Frames ---")
print("Press '1' to pause for Half-time (and generate Half 1 reports).")
print("Press '2' to start 2nd Half (when paused).")
print("Press 'q' to quit.")

frame_idx = 0
while True:
    ret, frame = stitched_cap.read()
    if not ret:
        print("❌ Failed to read frame from stitched stream")
        continue
    
    frame_idx += 1

    if processing_paused:
        paused_radar = BASE_PITCH_IMAGE.copy()
        cv2.putText(paused_radar, "HALF-TIME PAUSED", (RADAR_WIDTH // 2 - 150, RADAR_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(paused_radar, "Press '2' to Start 2nd Half", (RADAR_WIDTH // 2 - 200, RADAR_HEIGHT // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, paused_radar)
        key = cv2.waitKey(1) & 0xFF # Listen for key presses even when paused
        if key == ord('q'):
            break
        elif key == ord('2'): # Press '2' to start 2nd half
            on_start_second_half(frame) # Pass current frame for goalkeeper detection
        continue # Skip processing frames if paused

    yolo_result = PLAYER_MODEL.predict(frame, verbose=False)[0]
    dets = sv.Detections.from_ultralytics(yolo_result).with_nms(0.5, class_agnostic=True)
    ball = dets[dets.class_id == BALL_ID]
    ball.xyxy = sv.pad_boxes(ball.xyxy, px=10)
    nonball = dets[dets.class_id != BALL_ID]
    tracked = tracker.update_with_detections(nonball)

    pls = tracked[tracked.class_id == PLAYER_ID]
    refs = tracked[tracked.class_id == REFEREE_ID]

    # === LIVE CLASSIFICATION ===
    live_colors = [None] * len(pls.xyxy)

    def extract_color(i, box):
        crop = sv.crop_image(frame, box)
        color = center_weighted_color(crop, h_low, h_high)
        live_colors[i] = color if color is not None else (0, 0, 0)

    threads = [threading.Thread(target=extract_color, args=(i, box)) for i, box in enumerate(pls.xyxy)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    valid_live_colors = [lc for lc in live_colors if lc is not None]
    valid_player_indices = [i for i, lc in enumerate(live_colors) if lc is not None]

    if valid_live_colors:
        predicted_classes = kmeans.predict(np.array(valid_live_colors, dtype=np.float64))
        new_player_class_ids = np.zeros(len(pls.class_id), dtype=int)
        for i, pred_class in zip(valid_player_indices, predicted_classes):
            new_player_class_ids[i] = pred_class
        pls.class_id = new_player_class_ids
    else:
        pls.class_id = np.zeros(len(pls.class_id), dtype=int)

    # Override goalkeeper IDs
    gk_mask = np.array([tid in goalkeeper_tracker_ids for tid in pls.tracker_id])
    gks = pls[gk_mask]
    gks.class_id = np.array([goalkeeper_id_to_team.get(tid, 0) for tid in gks.tracker_id])
    players = sv.Detections.merge([pls[~gk_mask], gks])
    refs.class_id -= 1

    ball_xy_original = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    ball_xy = static_transformer.transform_points(ball_xy_original)
    ply_xy_original = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    ply_xy = static_transformer.transform_points(ply_xy_original)
    ref_xy = static_transformer.transform_points(refs.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))

    # === POSSESSION TRACKING ===
    if len(ball_xy) > 0:
        ball_center = ball_xy[0]
        closest_player_id = None
        min_dist = float('inf')

        for i, player_pos in enumerate(ply_xy):
            dist = np.linalg.norm(player_pos - ball_center)
            if dist < min_dist:
                min_dist = dist
                closest_player_id = players.tracker_id[i]

        if closest_player_id is not None and min_dist < POSSESSION_DISTANCE_THRESHOLD:
            closest_player_team = track_id_to_team.get(closest_player_id)
            if closest_player_team is None:
                player_idx = np.where(players.tracker_id == closest_player_id)[0]
                if len(player_idx) > 0:
                    closest_player_team = players.class_id[player_idx[0]]
                else:
                    closest_player_team = 0 # Default to team 0 if no team found

            if current_possessor_id is None:
                current_possessor_id = closest_player_id
                current_possessor_team = closest_player_team
                possession_event_log.append((frame_idx, 'START_POSSESSION', current_possessor_id, current_possessor_team))
            elif closest_player_id != current_possessor_id:
                if current_possessor_team is not None:
                    if closest_player_team == current_possessor_team:
                        # Successful pass
                        if current_half == 1:
                            successful_passes_half1[current_possessor_team] += 1
                        elif current_half == 2:
                            successful_passes_half2[current_possessor_team] += 1
                        successful_passes_overall[current_possessor_team] += 1

                        possession_event_log.append((frame_idx, 'PASS_SUCCESS', current_possessor_id, current_possessor_team, closest_player_id))
                        player_idx_prev_possessor = np.where(players.tracker_id == current_possessor_id)[0]
                        if len(player_idx_prev_possessor) > 0:
                            pos_meter = ply_xy[player_idx_prev_possessor[0]]
                            x_meter = np.clip(pos_meter[0], 0, CONFIG.length)
                            y_meter = np.clip(pos_meter[1], 0, CONFIG.width)
                            x_pixel = (x_meter / CONFIG.length) * RADAR_WIDTH
                            y_pixel = RADAR_HEIGHT - (y_meter / CONFIG.width) * RADAR_HEIGHT

                            if current_possessor_team == 0:
                                if current_half == 1:
                                    successful_pass_start_positions_team0_half1.append([x_meter, y_meter])
                                    successful_pass_start_pixels_team0_half1.append([x_pixel, y_pixel])
                                elif current_half == 2:
                                    successful_pass_start_positions_team0_half2.append([x_meter, y_meter])
                                    successful_pass_start_pixels_team0_half2.append([x_pixel, y_pixel])
                                successful_pass_start_positions_team0_overall.append([x_meter, y_meter])
                                successful_pass_start_pixels_team0_overall.append([x_pixel, y_pixel])
                            else: # current_possessor_team == 1
                                if current_half == 1:
                                    successful_pass_start_positions_team1_half1.append([x_meter, y_meter])
                                    successful_pass_start_pixels_team1_half1.append([x_pixel, y_pixel])
                                elif current_half == 2:
                                    successful_pass_start_positions_team1_half2.append([x_meter, y_meter])
                                    successful_pass_start_pixels_team1_half2.append([x_pixel, y_pixel])
                                successful_pass_start_positions_team1_overall.append([x_meter, y_meter])
                                successful_pass_start_pixels_team1_overall.append([x_pixel, y_pixel])
                    else:
                        # Unsuccessful pass (turnover)
                        if current_half == 1:
                            unsuccessful_passes_half1[current_possessor_team] += 1
                        elif current_half == 2:
                            unsuccessful_passes_half2[current_possessor_team] += 1
                        unsuccessful_passes_overall[current_possessor_team] += 1
                        possession_event_log.append((frame_idx, 'PASS_UNSUCCESS', current_possessor_id, current_possessor_team, closest_player_id, closest_player_team))

                current_possessor_id = closest_player_id
                current_possessor_team = closest_player_team

    # === UPDATE POSSESSION STATS (ONLY DURING ACTIVE PLAY) ===
    if not processing_paused:
        total_active_frames += 1
        
        if current_possessor_team == 0:
            team0_possession_frames += 1
        elif current_possessor_team == 1:
            team1_possession_frames += 1
        else:
            no_possession_frames += 1

    # === VORONOI DIAGRAM ACCUMULATION ===
    if (frame_idx % (VORONOI_CALC_SKIP_FRAMES + 1) == 0):
        team_0_players = ply_xy[players.class_id == 0]
        team_1_players = ply_xy[players.class_id == 1]

        if len(team_0_players) > 0 and len(team_1_players) > 0:
            voronoi_img = BASE_PITCH_IMAGE.copy()
            voronoi_img = draw_pitch_voronoi_diagram(
                config=CONFIG,
                team_1_xy=team_0_players,
                team_2_xy=team_1_players,
                team_1_color=sv.Color.from_hex('00BFFF'),
                team_2_color=sv.Color.from_hex('FF1493'),
                pitch=voronoi_img
            )

            if voronoi_img.shape[2] == 4:
                voronoi_bgr = cv2.cvtColor(voronoi_img, cv2.COLOR_RGBA2BGR)
            else:
                voronoi_bgr = voronoi_img.copy()

            # Dynamic team color discovery (first frame only)
            if not found_team_colors:
                pixels = voronoi_bgr.reshape(-1, 3)
                not_black_mask = np.linalg.norm(pixels - [0,0,0], axis=1) > 20
                not_white_mask = np.linalg.norm(pixels - [255,255,255], axis=1) > 20
                filtered_pixels = pixels[not_black_mask & not_white_mask]

                if len(filtered_pixels) >= 2:
                    kmeans_colors = KMeans(n_clusters=2, n_init="auto", random_state=42).fit(filtered_pixels.astype(np.float32))
                    dist_to_ref0_cluster0 = np.linalg.norm(kmeans_colors.cluster_centers_[0] - REF_TEAM_0_BGR)
                    dist_to_ref0_cluster1 = np.linalg.norm(kmeans_colors.cluster_centers_[1] - REF_TEAM_0_BGR)

                    if dist_to_ref0_cluster0 < dist_to_ref0_cluster1:
                        dynamic_team0_bgr = kmeans_colors.cluster_centers_[0].astype(np.uint8)
                        dynamic_team1_bgr = kmeans_colors.cluster_centers_[1].astype(np.uint8)
                    else:
                        dynamic_team0_bgr = kmeans_colors.cluster_centers_[1].astype(np.uint8)
                        dynamic_team1_bgr = kmeans_colors.cluster_centers_[0].astype(np.uint8)

                    found_team_colors = True

            # Accumulate control if colors are discovered
            if found_team_colors:
                pixels = voronoi_bgr.reshape(-1, 3)
                dist_to_team0_sq = np.sum((pixels - dynamic_team0_bgr)**2, axis=1)
                dist_to_team1_sq = np.sum((pixels - dynamic_team1_bgr)**2, axis=1)

                mask_team0_control = (dist_to_team0_sq < (COLOR_TOLERANCE**2)) & (dist_to_team0_sq < dist_to_team1_sq)
                mask_team1_control = (dist_to_team1_sq < (COLOR_TOLERANCE**2)) & (dist_to_team1_sq < dist_to_team0_sq)

                mask_team0_control = mask_team0_control.reshape(RADAR_HEIGHT, RADAR_WIDTH)
                mask_team1_control = mask_team1_control.reshape(RADAR_HEIGHT, RADAR_WIDTH)

                # Accumulate for current half
                if current_half == 1:
                    team_0_control_sum_half1[mask_team0_control] += 1
                    team_1_control_sum_half1[mask_team1_control] += 1
                    voronoi_frame_count_half1 += 1
                elif current_half == 2:
                    team_0_control_sum_half2[mask_team0_control] += 1
                    team_1_control_sum_half2[mask_team1_control] += 1
                    voronoi_frame_count_half2 += 1

                # Accumulate for overall
                team_0_control_sum_overall[mask_team0_control] += 1
                team_1_control_sum_overall[mask_team1_control] += 1
                voronoi_frame_count_overall += 1

    # === DRAW RADAR OUTPUT ===
    radar = BASE_PITCH_IMAGE.copy() # Start with a fresh pitch image
    radar = draw_points_on_pitch(CONFIG, xy=ball_xy, face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK, radius=10, pitch=radar)
    radar = draw_points_on_pitch(CONFIG, xy=ply_xy[players.class_id == 0], face_color=sv.Color.from_hex("00BFFF"), edge_color=sv.Color.BLACK, radius=16, pitch=radar)
    radar = draw_points_on_pitch(CONFIG, xy=ply_xy[players.class_id == 1], face_color=sv.Color.from_hex("FF1493"), edge_color=sv.Color.BLACK, radius=16, pitch=radar)
    radar = draw_points_on_pitch(CONFIG, xy=ref_xy, face_color=sv.Color.from_hex("FFD700"), edge_color=sv.Color.BLACK, radius=16, pitch=radar)

    team_color_0_bgr = sv.Color.from_hex("00BFFF").as_bgr()
    team_color_1_bgr = sv.Color.from_hex("FF1493").as_bgr()

    # Display current half
    cv2.putText(radar, f"Half: {current_half}", (RADAR_WIDTH - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Get possession percentages
    team0_percent, team1_percent = get_possession_percentages()

    if current_possessor_team is not None:
        team_name = f"Team {current_possessor_team + 1}"
        display_color = team_color_0_bgr if current_possessor_team == 0 else team_color_1_bgr
        cv2.putText(radar, f"Current Possessor: {team_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2, cv2.LINE_AA)

    for team_id in range(2):
        team_name = f"Team {team_id + 1}"
        # Display stats for the current half
        current_successful_passes = successful_passes_half1[team_id] if current_half == 1 else successful_passes_half2[team_id]
        current_unsuccessful_passes = unsuccessful_passes_half1[team_id] if current_half == 1 else unsuccessful_passes_half2[team_id]

        total_pass_attempts = current_successful_passes + current_unsuccessful_passes
        pass_accuracy_text = "Acc: N/A"
        if total_pass_attempts > 0:
            accuracy_percentage = (current_successful_passes / total_pass_attempts) * 100
            pass_accuracy_text = f"Acc: {accuracy_percentage:.1f}%"

        display_color = team_color_0_bgr if team_id == 0 else team_color_1_bgr
        y_offset = 60 + team_id * 90  # Increased spacing for all stats
        
        # Display all stats vertically aligned for each team
        cv2.putText(radar, f"{team_name} Passes: {current_successful_passes}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2, cv2.LINE_AA)
        cv2.putText(radar, pass_accuracy_text, 
                    (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2, cv2.LINE_AA)
        
        # Display possession percentage
        possession_percent = team0_percent if team_id == 0 else team1_percent
        cv2.putText(radar, f"Possession: {possession_percent:.1f}%", 
                    (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2, cv2.LINE_AA)

    # Add instructions for keyboard controls to the radar image
    cv2.putText(radar, "Press '1' for Half-time", (RADAR_WIDTH - 250, RADAR_HEIGHT - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(radar, "Press 'q' to Quit", (RADAR_WIDTH - 200, RADAR_HEIGHT - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    if radar.shape[2] == 4:
        radar = cv2.cvtColor(radar, cv2.COLOR_RGBA2BGR)

    out.write(radar)
    cv2.imshow(WINDOW_NAME, radar)
    key = cv2.waitKey(1) & 0xFF # Listen for key presses during processing
    if key == ord('q'):
        break
    elif key == ord('1') and current_half == 1 and not processing_paused: # Press '1' to stop half-time
        on_stop_half_time(frame_idx) # Pass current_frame_idx

# === AFTER MAIN PROCESSING LOOP (BEFORE REPORTS) ===
out.release()
stitched_cap.release()
cv2.destroyAllWindows()
print("✅ Radar video saved to:", OUTPUT_PATH)

# CAPTURE SECOND HALF STATS (add this right after video processing)
if current_half == 2:
    team0_possession_frames_half2 = team0_possession_frames
    team1_possession_frames_half2 = team1_possession_frames
    no_possession_frames_half2 = no_possession_frames
    total_active_frames_half2 = total_active_frames

# === GENERATE ALL REPORTS AT THE VERY END ===

print("\n--- Generating Half 1 Pass Visualizations ---")
generate_all_reports(
    successful_pass_start_positions_team0_half1,
    successful_pass_start_positions_team1_half1,
    successful_pass_start_pixels_team0_half1,
    successful_pass_start_pixels_team1_half1,
    team_0_control_sum_half1,
    team_1_control_sum_half1,
    voronoi_frame_count_half1,
    "half1"
)

# Check if the second half was processed at all (current_half is 2 or more frames were processed after half-time)
if current_half == 2:
    print("\n--- Generating Second Half Pass Visualizations ---")
    generate_all_reports(
        successful_pass_start_positions_team0_half2,
        successful_pass_start_positions_team1_half2,
        successful_pass_start_pixels_team0_half2,
        successful_pass_start_pixels_team1_half2,
        team_0_control_sum_half2,
        team_1_control_sum_half2,
        voronoi_frame_count_half2,
        "half2"
    )

print("\n--- Generating Overall Match Pass Visualizations ---")
generate_all_reports(
    successful_pass_start_positions_team0_overall,
    successful_pass_start_positions_team1_overall,
    successful_pass_start_pixels_team0_overall,
    successful_pass_start_pixels_team1_overall,
    team_0_control_sum_overall,
    team_1_control_sum_overall,
    voronoi_frame_count_overall,
    "overall"
)

# === FINAL STATS ===
print("\n=== Final Statistics ===")
print(f"Total frames with Voronoi (Overall): {voronoi_frame_count_overall}")

# Helper function to calculate possession percentages
def calculate_possession_percentages(team0_frames, team1_frames):
    total = team0_frames + team1_frames
    if total == 0:
        return 0.0, 0.0
    return (team0_frames/total)*100, (team1_frames/total)*100

# Helper function to print team stats
def print_team_stats(team_id, successful, unsuccessful, possession_pct, half_name=""):
    team_name = f"Team {team_id + 1}"
    suffix = f" ({half_name})" if half_name else ""
    total_passes = successful + unsuccessful
    
    print(f"\n{team_name}{suffix}:")
    print(f"  Successful Passes: {successful}")
    print(f"  Unsuccessful Passes: {unsuccessful}")
    print(f"  Total Pass Attempts: {total_passes}")
    if total_passes > 0:
        print(f"  Pass Accuracy: {successful/total_passes*100:.1f}%")
    else:
        print("  Pass Accuracy: N/A")
    print(f"  Possession: {possession_pct:.1f}%")

# Initialize half-time variables if not already set
if 'team0_possession_frames_half1' not in globals():
    team0_possession_frames_half1 = team0_possession_frames if current_half == 1 else 0
    team1_possession_frames_half1 = team1_possession_frames if current_half == 1 else 0

if 'team0_possession_frames_half2' not in globals():
    team0_possession_frames_half2 = team0_possession_frames if current_half == 2 else 0
    team1_possession_frames_half2 = team1_possession_frames if current_half == 2 else 0

# --- Half 1 Statistics ---
if team0_possession_frames_half1 > 0 or team1_possession_frames_half1 > 0:
    print("\n--- Half 1 Statistics ---")
    team0_pct_h1, team1_pct_h1 = calculate_possession_percentages(
        team0_possession_frames_half1, 
        team1_possession_frames_half1
    )
    
    for team_id in range(2):
        successful = successful_passes_half1[team_id]
        unsuccessful = unsuccessful_passes_half1[team_id]
        possession_pct = team0_pct_h1 if team_id == 0 else team1_pct_h1
        print_team_stats(team_id, successful, unsuccessful, possession_pct, "Half 1")

# --- Half 2 Statistics ---
if current_half == 2 and (team0_possession_frames_half2 > 0 or team1_possession_frames_half2 > 0):
    print("\n--- Half 2 Statistics ---")
    team0_pct_h2, team1_pct_h2 = calculate_possession_percentages(
        team0_possession_frames_half2,
        team1_possession_frames_half2
    )
    
    for team_id in range(2):
        successful = successful_passes_half2[team_id]
        unsuccessful = unsuccessful_passes_half2[team_id]
        possession_pct = team0_pct_h2 if team_id == 0 else team1_pct_h2
        print_team_stats(team_id, successful, unsuccessful, possession_pct, "Half 2")

# --- Overall Statistics ---
print("\n--- Overall Match Statistics ---")
team0_total_frames = team0_possession_frames_half1 + team0_possession_frames_half2
team1_total_frames = team1_possession_frames_half1 + team1_possession_frames_half2
team0_pct_ov, team1_pct_ov = calculate_possession_percentages(team0_total_frames, team1_total_frames)

for team_id in range(2):
    successful = successful_passes_overall[team_id]
    unsuccessful = unsuccessful_passes_overall[team_id]
    possession_pct = team0_pct_ov if team_id == 0 else team1_pct_ov
    print_team_stats(team_id, successful, unsuccessful, possession_pct)

# Additional summary
total_match_frames = total_active_frames_half1 + total_active_frames_half2 if current_half == 2 else total_active_frames_half1
print(f"\nTotal active frames: {total_match_frames}")
print(f"Team 1 possession: {team0_total_frames/total_match_frames*100:.1f}%")
print(f"Team 2 possession: {team1_total_frames/total_match_frames*100:.1f}%")