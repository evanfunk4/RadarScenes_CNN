import os
from typing import List


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


from config import BEVConfig, TrainConfig
from preprocessing import compute_bev_shape
from dataset import RadarScenesBEVSegDataset, infer_in_channels, collect_npz_files_from_dir
from models import UNetBEV


from torchinfo import summary
from datetime import datetime




def split_sequences_by_name(npz_paths: List[str], num_test: int = 3):
   """
   Choose test sequences, return (test_sequence_names).
   """
   seq_names = sorted(
       set(os.path.basename(p).split("_bev_seg.npz")[0] for p in npz_paths)
   )
   if num_test >= len(seq_names):
       return seq_names
   return seq_names[:num_test]




def train_model(cfg: TrainConfig) -> None:
   # 1) Collect all .npz files from the preprocessed directory
   preproc_dir = os.path.join(cfg.work_dir, "preprocessed")
   npz_paths = collect_npz_files_from_dir(preproc_dir)
   if not npz_paths:
       raise RuntimeError(f"No NPZ files found in {preproc_dir}.")


   # 2) Decide which sequences to hold out for test
   if cfg.test_sequences is None:
       cfg.test_sequences = split_sequences_by_name(npz_paths, num_test=3)
   print(f"Test sequences: {cfg.test_sequences}")


   # 3) Make datasets
   train_dataset = RadarScenesBEVSegDataset(
       npz_paths=npz_paths,
       bev_cfg=cfg.bev,
       split="train",
       test_sequences=cfg.test_sequences,
       augment=True,
   )
   val_dataset = RadarScenesBEVSegDataset(
       npz_paths=npz_paths,
       bev_cfg=cfg.bev,
       split="val",
       test_sequences=cfg.test_sequences,
       augment=False,
   )
   test_dataset = RadarScenesBEVSegDataset(
       npz_paths=npz_paths,
       bev_cfg=cfg.bev,
       split="test",
       test_sequences=cfg.test_sequences,
       augment=False,
   )


   train_loader = DataLoader(
       train_dataset,
       batch_size=cfg.batch_size,
       shuffle=True,
       num_workers=cfg.num_workers,
       pin_memory=cfg.pin_memory,
       persistent_workers=cfg.persistent_workers
   )
   val_loader = DataLoader(
       val_dataset,
       batch_size=cfg.batch_size,
       shuffle=False,
       num_workers=cfg.num_workers,
       pin_memory=cfg.pin_memory,
       persistent_workers=cfg.persistent_workers
   )
   test_loader = DataLoader(
       test_dataset,
       batch_size=cfg.batch_size,
       shuffle=False,
       num_workers=cfg.num_workers,
       pin_memory=cfg.pin_memory,
       persistent_workers=cfg.persistent_workers
   )


   print(f"Training batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")


   # 4) Model / loss / optimizer
   in_channels = infer_in_channels(cfg.bev)
   model = UNetBEV(in_channels=in_channels, num_classes=cfg.bev.num_classes)
   model.to(cfg.device)


   H, W = compute_bev_shape(cfg.bev)
   print(summary(
   model,
   input_size=(1, in_channels, H, W),  # batch size 1
   col_names=("input_size", "output_size", "num_params", "kernel_size"),
   depth=4
   ))




   criterion = nn.CrossEntropyLoss(ignore_index=-1)
   optimizer = torch.optim.AdamW(
       model.parameters(),
       lr=cfg.learning_rate,
       weight_decay=cfg.weight_decay,
   )


   use_amp = cfg.device == "cuda"
   if use_amp:
       scalar = GradScaler(enabled=True)


   best_val_loss = float("inf")
   best_model_path = os.path.join(cfg.work_dir, "best_model.pth")


   # 5) Training loop
   for epoch in range(cfg.num_epochs):


       print(f"\nEpoch {epoch+1}/{cfg.num_epochs} started...")


       model.train()
       total_loss = 0.0
       n_batches = 0


       for x, y in train_loader:


           x = x.to(cfg.device)
           y = y.to(cfg.device)




           optimizer.zero_grad(set_to_none=True)


           if use_amp:
               with autocast():
                   logits = model(x)
                   loss = criterion(logits, y)
               scalar.scale(loss).backward()
               scalar.step(optimizer)
               scalar.update()
           else:
               logits = model(x)
               loss = criterion(logits, y)
               loss.backward()
               optimizer.step()


           total_loss += loss.item()
           n_batches += 1


           if n_batches % 200 == 0:  # heartbeat interval
               ts = datetime.now().strftime('%H:%M:%S')


               # compute quick validation accuracy on 1 mini-batch
               model.eval()
               with torch.no_grad():
                   try:
                       x_val, y_val = next(val_iter)
                   except:
                       val_iter = iter(val_loader)
                       x_val, y_val = next(val_iter)


                   x_val = x_val.to(cfg.device)
                   y_val = y_val.to(cfg.device)
                   val_logits = model(x_val)
                   val_pred = torch.argmax(val_logits, dim=1)


                   # ignore label = -1 in accuracy
                   mask = (y_val != -1)
                   if mask.any():
                       acc = (val_pred[mask] == y_val[mask]).float().mean().item()
                   else:
                       acc = float('nan')


               model.train()


               print(
                   f"{ts}  [epoch {epoch+1}] batch {n_batches}/{len(train_loader)}  "
                   f"loss: {loss.item():.4f}  val_acc: {acc:.3f}"
               )




       train_loss = total_loss / max(1, n_batches)


       # Validation
       model.eval()
       val_loss = 0.0
       val_batches = 0
       with torch.no_grad():
           for x, y in val_loader:
               x = x.to(cfg.device)
               y = y.to(cfg.device)
               logits = model(x)
               loss = criterion(logits, y)
               val_loss += loss.item()
               val_batches += 1


       val_loss = val_loss / max(1, val_batches)
       print(f"Epoch {epoch+1}/{cfg.num_epochs} "
             f"| train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")


       if val_loss < best_val_loss:
           best_val_loss = val_loss
           torch.save(
               {"model_state_dict": model.state_dict(), "config": cfg},
               best_model_path,
           )
           print(f"  -> New best model saved to {best_model_path}")


   # 6) Test
   print("Evaluating best model on test set...")
   checkpoint = torch.load(best_model_path, map_location=cfg.device)
   model.load_state_dict(checkpoint["model_state_dict"])
   model.eval()


   test_loss = 0.0
   test_batches = 0
   with torch.no_grad():
       for x, y in test_loader:
           x = x.to(cfg.device)
           y = y.to(cfg.device)
           logits = model(x)
           loss = criterion(logits, y)
           test_loss += loss.item()
           test_batches += 1


   test_loss = test_loss / max(1, test_batches)
   print(f"Test loss (cross-entropy): {test_loss:.4f}")




if __name__ == "__main__":

   RADARSCENES_ROOT = "./RadarScenes"
   WORK_DIR = "./experiments"


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
   )


   train_cfg = TrainConfig(
       dataset_root=RADARSCENES_ROOT,
       work_dir=WORK_DIR,
       bev=bev_cfg,
       batch_size=8,
       num_workers=0,
       num_epochs=30,
       learning_rate=1e-3,
       weight_decay=1e-4,
       device="cuda" if torch.cuda.is_available() else "cpu",
       test_sequences=None,  # let script pick some
   )


   train_model(train_cfg)
