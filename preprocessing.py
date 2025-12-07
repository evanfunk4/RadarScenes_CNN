import os
import json
import math
from typing import List, Tuple, Optional
import time # Import time module for profiling

import numpy as np
from radar_scenes.sequence import Sequence


from config import BEVConfig




# ------------- BEV helpers -------------

def compute_bev_shape(cfg: BEVConfig) -> Tuple[int, int]:
   """
   Return (H, W) for BEV grid.
   H along y (left-right), W along x (forward-back).
   """
   width_m = cfg.x_max - cfg.x_min
   height_m = cfg.y_max - cfg.y_min
   W = int(math.ceil(width_m / cfg.resolution))
   H = int(math.ceil(height_m / cfg.resolution))
   return H, W



def num_bev_channels(cfg: BEVConfig) -> int:
   c = 0
   if cfg.use_counts:
       c += 1
   if cfg.use_log_counts:
       c += 1
   if cfg.use_mean_rcs:
       c += 1
   if cfg.use_mean_abs_vr:
       c += 1
   if cfg.use_tslh:
       c += 1
   return c



def pointcloud_to_indices(
   x_cc: np.ndarray,
   y_cc: np.ndarray,
   cfg: BEVConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
   """
   Map car-centric coordinates (x_cc, y_cc) to BEV grid indices.


   Returns:
       ix: x indices (W-dimension)
       iy: y indices (H-dimension)
       valid: boolean mask for points inside the grid
   """
   x_rel = x_cc - cfg.x_min
   y_rel = y_cc - cfg.y_min


   ix = np.floor(x_rel / cfg.resolution).astype(np.int32)
   iy = np.floor(y_rel / cfg.resolution).astype(np.int32)


   H, W = compute_bev_shape(cfg)
   valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
   return ix, iy, valid



def build_bev_and_labels_for_scene(
   scene,
   cfg: BEVConfig,
   tslh_grid_prev: Optional[np.ndarray],
   dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
   """
   Build BEV feature tensor + per-cell semantic labels + updated TSLH grid
   for one RadarScenes scene.


   Returns:
       bev: (H, W, C) float32
       labels: (H, W) int16 (class index or -1 for ignore)
       tslh_grid: (H, W) float32 (time since last hit, seconds)
   """
   H, W = compute_bev_shape(cfg)
   C = num_bev_channels(cfg)


   rd = scene.radar_data
   x_cc = rd["x_cc"]
   y_cc = rd["y_cc"]
   rcs = rd["rcs"]
   vr = rd["vr_compensated"]
   label_ids = rd["label_id"]   # int16


   ix, iy, valid = pointcloud_to_indices(x_cc, y_cc, cfg)


   ix = ix[valid]
   iy = iy[valid]
   rcs = rcs[valid]
   vr = vr[valid]
   label_ids = label_ids[valid]


   # Accumulators
   counts = np.zeros((H, W), dtype=np.float32)
   sum_rcs = np.zeros((H, W), dtype=np.float32)
   sum_abs_vr = np.zeros((H, W), dtype=np.float32)


   label_hist = np.zeros((H, W, cfg.num_classes), dtype=np.int32)


   if ix.size > 0:
       np.add.at(counts, (iy, ix), 1.0)
       np.add.at(sum_rcs, (iy, ix), rcs)
       np.add.at(sum_abs_vr, (iy, ix), np.abs(vr))


       valid_labels = (label_ids >= 0) & (label_ids < cfg.num_classes)
       ix_l = ix[valid_labels]
       iy_l = iy[valid_labels]
       lbl_l = label_ids[valid_labels]
       np.add.at(label_hist, (iy_l, ix_l, lbl_l), 1)


   eps = 1e-6
   mean_rcs = np.where(counts > 0, sum_rcs / (counts + eps), 0.0)
   mean_abs_vr = np.where(counts > 0, sum_abs_vr / (counts + eps), 0.0)
   log_counts = np.log1p(counts)


   # Time since last hit (TSLH)
   if tslh_grid_prev is None:
       tslh_grid = np.full((H, W), fill_value=cfg.max_tslh_s, dtype=np.float32)
   else:
       tslh_grid = (tslh_grid_prev + dt).astype(np.float32)
       tslh_grid = np.minimum(tslh_grid, cfg.max_tslh_s)
   tslh_grid[counts > 0] = 0.0


   channels = []
   if cfg.use_counts:
       channels.append(counts)
   if cfg.use_log_counts:
       channels.append(log_counts)
   if cfg.use_mean_rcs:
       channels.append(mean_rcs)
   if cfg.use_mean_abs_vr:
       channels.append(mean_abs_vr)
   if cfg.use_tslh:
       channels.append(tslh_grid)


   bev = np.stack(channels, axis=-1).astype(np.float32)


   # Per-cell majority label
   labels = np.full((H, W), fill_value=-1, dtype=np.int16)
   cell_point_counts = label_hist.sum(axis=-1)
   mask_cells = cell_point_counts >= cfg.min_points_per_cell
   if np.any(mask_cells):
       majority_labels = label_hist.argmax(axis=-1).astype(np.int16)
       labels[mask_cells] = majority_labels[mask_cells]


   return bev, labels, tslh_grid



def preprocess_sequence(
   scenes_json_path: str,
   cfg: BEVConfig,
   out_dir: str,
) -> str:
   """
   Preprocess a single RadarScenes sequence and save as NPZ.


   NPZ contains:
       bevs: (T, H, W, C)
       labels: (T, H, W)
       timestamps: (T,)
       scene_ids: (T,)
       cfg: json-serialized BEVConfig dict


   Returns:
       path to npz file
   """
   start_time_sequence = time.time()
   os.makedirs(out_dir, exist_ok=True)


   seq_load_start = time.time()
   seq = Sequence.from_json(scenes_json_path)
   seq_load_end = time.time()
   print(f"  [preprocess_sequence] Time to load sequence {os.path.basename(scenes_json_path)}: {seq_load_end - seq_load_start:.4f} seconds")

   seq_name = os.path.basename(os.path.dirname(scenes_json_path))


   bevs: List[np.ndarray] = []
   labels_list: List[np.ndarray] = []
   timestamps: List[float] = []
   scene_ids: List[int] = []


   tslh_grid = None
   prev_ts: Optional[float] = None

   scene_processing_times = []
   try:
       for idx, scene in enumerate(seq.scenes()):
           # Check for preprocessing scene limit from BEVConfig
           if hasattr(cfg, 'max_scenes_per_sequence_for_preprocessing') and cfg.max_scenes_per_sequence_for_preprocessing is not None and idx >= cfg.max_scenes_per_sequence_for_preprocessing:
               print(f"    [preprocess_sequence] Limiting sequence {seq_name} to {cfg.max_scenes_per_sequence_for_preprocessing} scenes.")
               break

           print(f"    [preprocess_sequence] Processing scene {idx+1} of sequence {seq_name}...")
           scene_start_time = time.time()
           ts = float(scene.timestamp) * 1e-6  # microseconds \u2192 seconds
           dt = 0.0 if prev_ts is None else max(0.0, ts - prev_ts)
           prev_ts = ts


           bev, lbl, tslh_grid = build_bev_and_labels_for_scene(
               scene, cfg, tslh_grid, dt
           )


           bevs.append(bev)
           labels_list.append(lbl)
           timestamps.append(ts)
           scene_ids.append(idx)
           scene_processing_times.append(time.time() - scene_start_time)
           print(f"    [preprocess_sequence] Finished scene {idx+1} in {scene_processing_times[-1]:.4f} seconds.")

   except Exception as e:
       print(f"[ERROR] An error occurred during scene processing for sequence {seq_name}: {e}")
       # Re-raise to ensure the main script knows it failed
       raise

   if scene_processing_times:
       print(f"  [preprocess_sequence] Avg time per scene: {np.mean(scene_processing_times):.4f} seconds (Total {len(scene_processing_times)} scenes)")
       print(f"  [preprocess_sequence] Max time per scene: {np.max(scene_processing_times):.4f} seconds")

   stack_start_time = time.time()
   bevs_arr = np.stack(bevs, axis=0)         # (T, H, W, C)
   labels_arr = np.stack(labels_list, axis=0)  # (T, H, W)
   stack_end_time = time.time()
   print(f"  [preprocess_sequence] Time to stack arrays: {stack_end_time - stack_start_time:.4f} seconds")

   save_start_time = time.time()
   out_path = os.path.join(out_dir, f"{seq_name}_bev_seg.npz")
   np.savez_compressed(
       out_path,
       bevs=bevs_arr,
       labels=labels_arr,
       timestamps=np.array(timestamps, dtype=np.float64),
       scene_ids=np.array(scene_ids, dtype=np.int32),
       cfg=json.dumps(cfg.__dict__),
   )
   save_end_time = time.time()
   print(f"  [preprocess_sequence] Time to save {out_path}: {save_end_time - save_start_time:.4f} seconds")
   print(f"Saved preprocessed sequence {seq_name} to {out_path}")

   print(f"[preprocess_sequence] Total time for sequence {seq_name}: {time.time() - start_time_sequence:.4f} seconds")
   return out_path



def preprocess_all_sequences(
   dataset_root: str,
   out_dir: str,
   cfg: BEVConfig,
) -> List[str]:
   """
   Preprocess all sequences under dataset_root/data/sequence_*.


   Skips sequences already preprocessed.


   Returns:
       list of npz paths
   """
   start_time_all_sequences = time.time()
   data_root = os.path.join(dataset_root, "RadarScenes", "data") # Corrected path to data directory
   os.makedirs(out_dir, exist_ok=True)


   seq_dirs = sorted(
       d for d in os.listdir(data_root)
       if d.startswith("sequence_")
   )


   npz_paths: List[str] = []


   # Process all sequences found (removed debugging limit here)
   for i, seq_dir in enumerate(seq_dirs):
       print(f"[preprocess_all_sequences] Processing sequence {i+1}/{len(seq_dirs)}: {seq_dir}")
       scenes_json = os.path.join(data_root, seq_dir, "scenes.json")
       if not os.path.isfile(scenes_json):
           print(f"[preprocess_all_sequences] Skipping {seq_dir}, scenes.json not found.")
           continue


       out_path = os.path.join(out_dir, f"{seq_dir}_bev_seg.npz")
       if os.path.isfile(out_path):
           print(f"[preprocess_all_sequences] Skipping {seq_dir}, already exists.")
           npz_paths.append(out_path)
           continue


       try:
           p = preprocess_sequence(scenes_json, cfg, out_dir)
           npz_paths.append(p)
       except Exception as e:
           print(f"[ERROR] Preprocessing failed for sequence {seq_dir}: {e}")
           print("Continuing with next sequence...")

   print(f"[preprocess_all_sequences] Total time for all sequences: {time.time() - start_time_all_sequences:.4f} seconds")
   return npz_paths

if __name__ == "__main__":
    RADARSCENES_ROOT = "."

    WORK_DIR = "./experiments"
    os.makedirs(WORK_DIR, exist_ok=True)

    # Use the same BEVConfig as in train.py
    bev_cfg = BEVConfig(
        x_min=-20.0,
        x_max=80.0,
        y_min=-40.0,
        y_max=40.0,
        resolution=0.25,
        history=3,
        future=0,
        max_tslh_s=1.0,
        min_points_per_cell=1,
        num_classes=12,
        # plus any other flags (use_counts, use_log_counts, etc.) your BEVConfig has
    )

    # Explicitly set max_scenes_per_sequence_for_preprocessing to None for full run
    setattr(bev_cfg, 'max_scenes_per_sequence_for_preprocessing', None)

    out_dir = os.path.join(WORK_DIR, "preprocessed")
    npz_paths = preprocess_all_sequences(
        dataset_root=RADARSCENES_ROOT,
        out_dir=out_dir,
        cfg=bev_cfg,
    )
    print(f"Done. Wrote {len(npz_paths)} NPZ files into {out_dir}")
