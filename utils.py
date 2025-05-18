import os
import cv2
import numpy as np
from torch.utils.data import Dataset

def cv_imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img

class PlateDataset(Dataset):
    def __init__(self, data_root, client_id=None, mode='train'):
        self.samples = []
        if client_id is not None:
            # 目录名为 client_0、client_1 等
            root = os.path.join(data_root, f"client_{client_id}")
        else:
            root = os.path.join(data_root, mode)
        self._load_samples(root)

    def _load_samples(self, root):
        images_dir = os.path.join(root, "images")
        labels_dir = os.path.join(root, "labels")
        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            print(f"[WARN] {images_dir} or {labels_dir} not found!")
            return
        for fname in os.listdir(images_dir):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(images_dir, fname)
            label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")
            if not os.path.exists(label_path):
                continue
            img = cv_imread(img_path)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines()]
                    if len(lines) < 3:
                        continue
                    _, x_c, y_c, w, h = map(float, lines[0].split())
                    x_c, y_c, w, h = x_c * w_img, y_c * h_img, w * w_img, h * h_img
                    x1 = int(x_c - w / 2)
                    y1 = int(y_c - h / 2)
                    x2 = int(x_c + w / 2)
                    y2 = int(y_c + h / 2)
                    bbox = [max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)]
                    plate_number = lines[1]
                    plate_color = lines[2]
                    self.samples.append({
                        'image_path': img_path,
                        'image': img,
                        'bbox': bbox,
                        'plate_number': plate_number,
                        'plate_color': plate_color
                    })
            except Exception as e:
                print(f"[ERROR] Parse {label_path}: {e}")

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)