import os
import cv2
import torch
from torch.utils.data import Dataset

class PlateDataset(Dataset):
    def __init__(self, root, client_id=None, mode="train"):
        self.root = root
        self.mode = mode
        self.img_paths = []
        self.label_paths = []

        # 客户端
        if client_id is not None:
            data_folder = os.path.join(root, f"client_{client_id}")
        else:
            data_folder = os.path.join(root, mode)
        images_dir = os.path.join(data_folder, "images")
        labels_dir = os.path.join(data_folder, "labels")

        for fname in sorted(os.listdir(labels_dir)):
            if fname.endswith('.txt'):
                base = fname[:-4]
                img_path = os.path.join(images_dir, base + ".jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(images_dir, base + ".png")
                if not os.path.exists(img_path):
                    continue
                self.img_paths.append(img_path)
                self.label_paths.append(os.path.join(labels_dir, fname))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]
        img = cv2.imread(img_path)
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
            class_info = lines[0].split()
            x, y, w, h = map(float, class_info[1:5]) # YOLO格式
            plate_number = lines[1]
            plate_color = lines[2]
        H, W = img.shape[:2]
        x1 = int((x - w / 2) * W)
        y1 = int((y - h / 2) * H)
        x2 = int((x + w / 2) * W)
        y2 = int((y + h / 2) * H)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        return {
            "image": img,
            "bbox": (x1, y1, x2, y2),
            "plate_number": plate_number,
            "plate_color": plate_color,
            "img_path": img_path
        }