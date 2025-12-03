import os
import glob


import random
from typing import List, Tuple, Optional


import numpy as np
import torch
from torch.utils.data import Dataset


from config import BEVConfig
from preprocessing import num_bev_channels


def collect_npz_files_from_dir(directory: str) -> List[str]:
   """
   Recursively collect all .npz files from a directory.
   Returns a sorted list of file paths.
   """
   pattern = os.path.join(directory, '**', '*.npz')
   return sorted(glob.glob(pattern, recursive=True))




class RadarScenesBEVSegDataset(Dataset):
   """
   Loads preprocessed BEV + labels from NPZ files and returns temporal stacks:


       X: (C_total, H, W)
       y: (H, W) with ignore_index = -1


   Temporal window indices: [t-history .. t+future].
   """


   def __init__(
       self,
       npz_paths: List[str],
       bev_cfg: BEVConfig,
       split: str,
       test_sequences: Optional[List[str]] = None,
       augment: bool = True,
   ) -> None:
       super().__init__()
       self.bev_cfg = bev_cfg
       self.augment = augment
       self.split = split
       self.test_sequences = set(test_sequences or [])


       self.samples: List[Tuple[str, int]] = []  # (npz_path, center_idx)


       for path in npz_paths:
           seq_name = os.path.basename(path).split("_bev_seg.npz")[0]
           is_test_seq = seq_name in self.test_sequences


           if split == "train" and is_test_seq:
               continue
           if split == "val" and is_test_seq:
               continue
           if split == "test" and not is_test_seq:
               continue


           try:
               data = np.load(path, mmap_mode="r")
               T = data["bevs"].shape[0]
           except Exception as e:
               print(f"[ERROR] Could not load {path}: {e}")
               continue


           h = bev_cfg.history
           f = bev_cfg.future
           for center_idx in range(h, T - f):
               self.samples.append((path, center_idx))


       print(f"[{split}] dataset samples: {len(self.samples)}")


   def __len__(self) -> int:
       return len(self.samples)


   def __getitem__(self, idx: int):
       path, center = self.samples[idx]
       data = np.load(path)


       bevs = data["bevs"]      # (T, H, W, C)
       labels = data["labels"]  # (T, H, W)
       T, H, W, C = bevs.shape


       h = self.bev_cfg.history
       f = self.bev_cfg.future


       t_start = center - h
       t_end = center + f + 1    # exclusive


       bev_stack = bevs[t_start:t_end]        # (T_window, H, W, C)
       label_center = labels[center]          # (H, W)


       # (T, H, W, C) → (C_total, H, W)
       bev_stack = np.transpose(bev_stack, (0, 3, 1, 2))  # (T, C, H, W)
       bev_stack = bev_stack.reshape(-1, H, W)            # (C*T, H, W)


       x = torch.from_numpy(bev_stack.astype(np.float32))
       y = torch.from_numpy(label_center.astype(np.int64))


       # simple random flip augmentation
       if self.augment and self.split == "train":
           if random.random() < 0.5:
               x = torch.flip(x, dims=[2])  # flip in W
               y = torch.flip(y, dims=[1])
           if random.random() < 0.5:
               x = torch.flip(x, dims=[1])  # flip in H
               y = torch.flip(y, dims=[0])


       return x, y




def infer_in_channels(cfg: BEVConfig) -> int:
   """
   Compute number of input channels to the CNN based on BEVConfig and temporal window.
   """
   per_frame = num_bev_channels(cfg)
   T_window = cfg.history + cfg.future + 1
   return per_frame * T_window
