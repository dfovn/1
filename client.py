import os
import time
import torch
import copy
import cv2
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from plate_recognition.plate_rec import image_processing, get_plate_result
from utils import PlateDataset
from detect_rec_plate import load_model
from plate_recognition.plate_rec import init_model

def collate_fn(batch):
    return batch

class FedClient:
    def __init__(self, client_id, data_root, server):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        self.server = server

        self.local_detector = load_model("weights/yolov8s.pt", self.device)
        self.local_recognizer = init_model(self.device, "weights/plate_rec_color.pth", is_color=True)

        self.dataset = PlateDataset(data_root, client_id=client_id, mode='train')
        self.loader = DataLoader(self.dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_fn)

    def local_train(self, epochs=5):
        global_params = self.server.get_global_models()
        self.local_recognizer.load_state_dict(global_params["recognizer"])

        optimizer = torch.optim.Adam(self.local_recognizer.parameters(), lr=1e-4)
        criterion_plate = nn.CTCLoss(blank=0, zero_infinity=True)
        criterion_color = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.local_recognizer.train()
            epoch_loss = 0.0
            valid_batches = 0
            for batch in self.loader:
                imgs, targets, colors = [], [], []
                for item in batch:
                    img = item["image"]
                    x1, y1, x2, y2 = item["bbox"]
                    roi = img[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    if len(roi.shape) == 2:
                        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    elif roi.shape[2] == 4:
                        roi = roi[:, :, :3]
                    processed = image_processing(roi, self.device)
                    imgs.append(processed)
                    targets.append(self._str_to_indices(item["plate_number"]))
                    colors.append(self._color_to_idx(item["plate_color"]))
                if not imgs:
                    continue
                imgs = torch.cat(imgs, dim=0)
                input_lengths = torch.full((len(imgs),), imgs.shape[-1] // 8, dtype=torch.long, device=self.device)  # 推断长度
                target_concat = torch.cat(targets)
                target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long, device=self.device)
                color_targets = torch.tensor(colors, dtype=torch.long, device=self.device)
                logits, color_logits = self.local_recognizer(imgs)
                if logits.dim() == 3:
                    logits = logits.permute(1, 0, 2)
                try:
                    loss_plate = criterion_plate(logits.log_softmax(2), target_concat, input_lengths, target_lengths)
                    loss_color = criterion_color(color_logits, color_targets)
                    loss = loss_plate + loss_color
                except Exception as e:
                    print(f"[ERROR] Loss computation error: {e}")
                    continue
                if loss.item() < 0:
                    print(f"[WARN] Negative loss detected! Skipping batch. Details:")
                    print(f"imgs.shape: {imgs.shape}, targets: {targets}, color_targets: {colors}")
                    print(f"logits.shape: {logits.shape}, input_lengths: {input_lengths}, target_lengths: {target_lengths}")
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                valid_batches += 1
            print(f"Client {self.client_id} Epoch {epoch+1} Loss: {epoch_loss/(valid_batches or 1):.4f}")
        return {
            "recognizer": copy.deepcopy(self.local_recognizer.state_dict()),
            "num_samples": len(self.dataset)
        }

    def _str_to_indices(self, s):
        plateName = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
        idxs = [plateName.index(ch) if ch in plateName else 0 for ch in s]
        return torch.tensor(idxs, dtype=torch.long, device=self.device)

    def _color_to_idx(self, color):
        color_map = {'黑色': 0, '蓝色': 1, '绿色': 2, '白色': 3, '黄色': 4}
        return color_map.get(color, 0)