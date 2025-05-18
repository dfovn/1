import os
import time

import torch
import copy
import cv2
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from plate_recognition.plate_rec import image_processing, plateName, get_plate_result
from utils import PlateDataset
import traceback
from detect_rec_plate import load_model
from plate_recognition.plate_rec import init_model

# Debug配置
DEBUG_MODE = True
DEBUG_IMAGE_DIR = "./debug_images"
os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)

def collate_fn(batch):
    return batch


class FedClient:
    def __init__(self, client_id, data_root, server):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        self.server = server
        self.error_count = 0  # 错误计数器

        # 初始化本地模型
        self.local_detector = load_model("weights/yolov8s.pt", self.device)
        self.local_recognizer = init_model(self.device, "weights/plate_rec_color.pth", is_color=True)

        # 数据加载
        self.dataset = PlateDataset(data_root, client_id=client_id, mode='train')
        self.loader = DataLoader(self.dataset, batch_size=8, shuffle=True, num_workers=0,
                                 collate_fn=collate_fn)  # 保持原始数据格式

    def _log_debug_image(self, img, prefix=""):
        """保存调试图像"""
        if DEBUG_MODE:
            filename = f"{DEBUG_IMAGE_DIR}/client{self.client_id}_{prefix}_{time.strftime('%Y%m%d%H%M%S')}.jpg"
            cv2.imwrite(filename, img)
            return filename
        return ""

    def local_train(self, epochs=1):
        global_params = self.server.get_global_models()
        self.local_detector.load_state_dict(global_params["detector"])
        self.local_recognizer.load_state_dict(global_params["recognizer"])

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.local_recognizer.parameters()),
            lr=1e-4
        )

        for epoch in range(epochs):
            self.local_recognizer.train()
            total_loss = 0.0
            batch_counter = 0

            for batch in self.loader:
                batch_counter += 1
                optimizer.zero_grad()
                batch_loss = 0
                valid_samples = 0  # 有效样本计数器

                for item in batch:  # 修改为遍历原始数据项
                    try:
                        # =============== 调试区域 =============== #
                        # 原始图像检查
                        orig_img = item['image']
                        if not isinstance(orig_img, np.ndarray):
                            print(f"[ERROR] Invalid image type: {type(orig_img)}")
                            continue

                        # 边界框处理
                        x1, y1, x2, y2 = item['bbox']
                        print(f"Original bbox: {x1},{y1} -> {x2},{y2}")

                        # 转换为整数并限制范围
                        h, w = orig_img.shape[:2]
                        x1 = max(0, int(x1))
                        y1 = max(0, int(y1))
                        x2 = min(w, int(x2))
                        y2 = min(h, int(y2))

                        # 有效性检查
                        if x2 <= x1 or y2 <= y1:
                            print(f"[WARN] Invalid bbox: {x1},{y1}->{x2},{y2} in image {item.get('image_path', '')}")
                            continue

                        # 裁剪ROI
                        roi = orig_img[y1:y2, x1:x2]
                        print(f"ROI shape: {roi.shape}, dtype: {roi.dtype}")

                        # 空区域检查
                        if roi.size == 0:
                            print(f"[WARN] Empty ROI in image {item.get('image_path', '')}")
                            continue

                        # 通道检查
                        if len(roi.shape) == 2:
                            print(f"[DEBUG] Convert grayscale to BGR")
                            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                        elif roi.shape[2] == 4:
                            print(f"[DEBUG] Remove alpha channel")
                            roi = roi[:, :, :3]

                        # 保存调试图像
                        debug_path = self._log_debug_image(roi, "raw_roi")

                        # =============== 预处理 =============== #
                        # 使用原有预处理流程
                        processed_img = image_processing(roi, self.device)

                        # =============== 模型推理 =============== #
                        # 使用原有识别逻辑
                        plate_info = get_plate_result(
                            roi,  # 原始图像
                            self.device,
                            self.local_recognizer,
                            is_color=True
                        )
                        print(f"Recognition result: {plate_info}")

                        # =============== 损失计算 =============== #
                        # 获取真实标签
                        target_number = item['plate_number']
                        target_color = item['plate_color']

                        # 获取模型输出（需根据实际模型调整）
                        plate_logits, color_logits = self.local_recognizer(processed_img)

                        # 计算损失
                        plate_loss = self._calc_plate_loss(plate_logits, target_number)
                        color_loss = self._calc_color_loss(color_logits, target_color)

                        current_loss = plate_loss + color_loss
                        batch_loss += current_loss
                        valid_samples += 1

                    except Exception as e:
                        self.error_count += 1
                        print(f"\n[ERROR] Client {self.client_id} Error (count={self.error_count}):")
                        traceback.print_exc()
                        print(f"Problematic item: {item.get('image_path', 'unknown')}")
                        print(f"ROI shape: {roi.shape if 'roi' in locals() else 'N/A'}")
                        continue

                # 反向传播
                if valid_samples > 0:
                    batch_loss /= valid_samples
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item()
                    print(f"Batch {batch_counter} processed, valid samples: {valid_samples}")
                else:
                    print(f"Batch {batch_counter} skipped (no valid samples)")

            print(f"Client {self.client_id} Epoch {epoch + 1} Loss: {total_loss / len(self.loader):.4f}")

        return {
            "detector": copy.deepcopy(self.local_detector.state_dict()),
            "recognizer": copy.deepcopy(self.local_recognizer.state_dict())
        }

    def _calc_plate_loss(self, pred_logits, target_str):
        """基于模型输出的字符概率计算损失"""
        # 将 target_str 转换为索引序列（需实现转换逻辑）
        target_indices = self._str_to_indices(target_str)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred_logits, target_indices)
        return loss

    def _calc_color_loss(self, pred_color_logits, target_color):
        """基于颜色分类输出计算损失"""
        color_map = {'黑色': 0, '蓝色': 1, '绿色': 2, '白色': 3, '黄色': 4}
        target_idx = torch.tensor(color_map[target_color], device=self.device)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred_color_logits, target_idx)
        return loss


# =============== 调试辅助函数 ===============#
def print_data_sample(dataset, index=0):
    """打印数据样本调试信息"""
    sample = dataset[index]
    print("\n=== 数据样本调试 ===")
    print(f"Image path: {sample.get('image_path', 'unknown')}")
    print(f"Original bbox: {sample['bbox']}")
    print(f"Plate number: {sample['plate_number']}")
    print(f"Plate color: {sample['plate_color']}")

    img = sample['image']
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")

    # 显示ROI区域
    x1, y1, x2, y2 = sample['bbox']
    roi = img[y1:y2, x1:x2]
    cv2.imshow("Sample ROI", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 单元测试
    test_client = FedClient(0, "processed", None)
    print_data_sample(test_client.dataset)