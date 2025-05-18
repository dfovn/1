# utils.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from plate_recognition.plate_rec import plateName

class PlateDataset(Dataset):
    def __init__(self, base_path, client_id=None, mode='train'):
        self.metadata = []
        self.mode = mode

        # 构建路径
        path = os.path.join(base_path,
                            'train' if mode == 'train' else 'test',
                            f'client_{client_id}' if client_id is not None else '')

        if not os.path.exists(path):
            raise FileNotFoundError(f"路径不存在: {path}")

        img_dir = os.path.join(path, 'images')
        label_dir = os.path.join(path, 'labels')

        # 预加载元数据
        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            base_name = os.path.splitext(img_name)[0]
            label_path = os.path.join(label_dir, f"{base_name}.txt")

            if not os.path.isfile(label_path):
                continue

            try:
                # 预解析标签文件
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]

                if len(lines) < 3:
                    continue

                # 解析边界框
                # 增强bbox验证
                bbox = list(map(float, lines[0].split()[1:5]))
                if (bbox[2] < 0.01 or bbox[3] < 0.01 or  # 过滤过小bbox
                        any(not (0.0 <= x <= 1.0) for x in bbox)):
                    continue
                bbox = [max(0.0, min(1.0, x)) for x in bbox]

                # 解析车牌信息
                plate_number = lines[1].strip().upper()
                plate_color = lines[2].strip()

                # 缓存元数据
                self.metadata.append({
                    'img_path': os.path.join(img_dir, img_name),
                    'label_path': label_path,
                    'bbox': bbox,
                    'plate': (plate_number, plate_color)
                })
            except Exception as e:
                print(f"预加载错误 {img_name}: {str(e)}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        try:
            meta = self.metadata[idx]
            # 使用半精度加载
            img = cv2.imdecode(np.fromfile(meta['img_path'], dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2)  # 压缩加载
            if img is None:
                return None

            h, w = img.shape[:2]
            bbox = meta['bbox']

            # 转换为半精度浮点
            img = img.astype(np.float16)  # 关键修改点
            img = (img / 255.0 - 0.588) / 0.193  # 使用硬编码的mean/std

            # 转换坐标
            x_center, y_center = bbox[0] * w, bbox[1] * h
            box_w = max(10, bbox[2] * w)
            box_h = max(10, bbox[3] * h)

            x1 = int(max(0, x_center - box_w / 2))
            y1 = int(max(0, y_center - box_h / 2))
            x2 = int(min(w, x_center + box_w / 2))
            y2 = int(min(h, y_center + box_h / 2))

            if x2 <= x1 or y2 <= y1:
                return None

            # 添加通道检查和内存连续性保证
            if img.shape[2] != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # 确保内存连续
            img = np.ascontiguousarray(img)

            # 添加尺寸二次验证
            if (x2 - x1) < 16 or (y2 - y1) < 8:
                return None

            return {
                'image': img.astype(np.float32),
                'bbox': np.array([x1, y1, x2, y2], dtype=np.float32),
                'plate_number': meta['plate'][0],
                'plate_color': meta['plate'][1],
                'image_path': meta['img_path']
            }
        except Exception as e:
            print(f"数据加载错误 {idx}: {str(e)}")
            return None

    @staticmethod
    def collate_fn(batch):
        """自定义collate函数，处理无效样本"""
        # 过滤None值
        valid_batch = [b for b in batch if b is not None]
        if not valid_batch:
            return None

        # 组织数据
        collated = {
            'image': torch.stack([torch.from_numpy(b['image']) for b in valid_batch]),
            'bbox': torch.stack([torch.from_numpy(b['bbox']) for b in valid_batch]),
            'plate_number': [b['plate_number'] for b in valid_batch],
            'plate_color': [b['plate_color'] for b in valid_batch],
            'image_path': [b['image_path'] for b in valid_batch]
        }
        return collated

    def validate_dataset(self):
        """数据集完整性检查"""
        error_count = 0
        for idx in tqdm(range(len(self)), desc="验证数据集"):
            try:
                item = self.__getitem__(idx)
                if item is None:
                    error_count += 1
                else:
                    # 检查图像有效性
                    assert item['image'].shape[2] == 3, "通道数错误"
                    assert item['bbox'].shape == (4,), "边界框格式错误"
            except:
                error_count += 1
        print(f"数据集验证完成，总样本数: {len(self)}，无效样本数: {error_count}")