from typing import List, Optional
from dataclasses import dataclass

import torch


@dataclass
class BEVConfig:
   # BEV extents (car-centric)
   x_min: float = -20.0   # behind car (forward axis)
   x_max: float = 80.0    # in front of car
   y_min: float = -40.0   # left
   y_max: float = 40.0    # right
   resolution: float = 0.25  # meters per cell


   # Which feature channels to include
   use_counts: bool = True
   use_log_counts: bool = True
   use_mean_rcs: bool = True
   use_mean_abs_vr: bool = True
   use_tslh: bool = True  # time since last hit


   # Temporal stacking
   history: int = 3   # number of past frames
   future: int = 0    # usually 0 for causal model
   max_tslh_s: float = 1.0  # clamp TSLH


   # Labeling
   min_points_per_cell: int = 1
   num_classes: int = 12  # RadarScenes label_id range 0..11

   # Added for debugging preprocessing memory issues
   max_scenes_per_sequence_for_preprocessing: Optional[int] = None


@dataclass
class TrainConfig:
   dataset_root: str
   work_dir: str
   bev: BEVConfig


   batch_size: int = 8
   num_workers: int = 0
   num_epochs: int = 30
   learning_rate: float = 1e-3
   weight_decay: float = 1e-4


   pin_memory = False
   persistent_workers = False


   def _detect_device() -> str: # deprecated
      """Safely detect an available device. Some torch builds may not expose
      the same MPS helpers (or may raise on access). This helper avoids
      AttributeError when importing the module on systems without MPS.
      """
      try:
         # Prefer MPS (Apple Silicon) when available
         mps_available = False
         if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
            mps_available = torch.mps.is_available()
         elif hasattr(torch, "has_mps"):
            mps_available = bool(getattr(torch, "has_mps"))
         elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available"):
            mps_available = torch.backends.mps.is_available()

         if mps_available:
            return "mps"
         if torch.cuda.is_available():
            return "cuda"
      except Exception:
         # Any error while probing devices -> fall back to cpu
         pass
      return "cpu"

   device: str = "cuda" if torch.cuda.is_available() else "cpu"


   # Which sequences to hold out for test
   test_sequences: Optional[List[str]] = None
