# =============== fed_client.py ===============#
import os
import sys
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
from plate_recognition.plate_rec import plateName

# Debug配置
DEBUG_MODE = True
DEBUG_IMAGE_DIR = "./debug_images"
os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)

def custom_collate(batch):
    """增强型数据组装函数"""
    # 过滤无效样本
    valid_batch = [b for b in batch if (
            isinstance(b, dict) and
            'image' in b and
            'bbox' in b and
            'plate_number' in b and
            'plate_color' in b
    )]

    if len(valid_batch) == 0:
        return None

    # 组织数据 - 确保 we're working with tensors
    try:
        return {
            'image': torch.stack([torch.as_tensor(b['image']) for b in valid_batch]),
            'bbox': torch.stack([torch.as_tensor(b['bbox']) for b in valid_batch]),
            'plate_number': [b['plate_number'] for b in valid_batch],
            'plate_color': [b['plate_color'] for b in valid_batch],
            'image_path': [b.get('image_path', 'unknown') for b in valid_batch]
        }
    except Exception as e:
        print(f"Error in collate: {str(e)}")
        return None



class FedClient:
    def __init__(self, client_id, data_root, server):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        self.server = server
        self.error_count = 0
        self.ctc_loss = nn.CTCLoss(blank=0)  # 新增CTCLoss

        # 延迟初始化模型
        self.local_detector = None
        self.local_recognizer = None

        # 数据加载器
        self.dataset = PlateDataset(data_root, client_id=client_id, mode='train')
        self.loader = DataLoader(
            self.dataset,
            batch_size=2,
            collate_fn=custom_collate,
            num_workers=0,
            sampler=torch.utils.data.sampler.SequentialSampler(  # 改为顺序采样
                [i for i in range(len(self.dataset))
                 if (self.dataset[i] is not None and
                     'plate_number' in self.dataset[i])]
            ))

    def _log_debug_image(self, img, prefix=""):
        """保存调试图像"""
        if DEBUG_MODE:
            filename = f"{DEBUG_IMAGE_DIR}/client{self.client_id}_{prefix}_{time.strftime('%Y%m%d%H%M%S')}.jpg"
            cv2.imwrite(filename, img)
            return filename
        return ""

    # client.py
    def local_train(self, epochs=1):
        # 延迟加载模型
        if self.local_detector is None:
            self.local_detector = load_model("weights/yolov8s.pt", self.device)
        if self.local_recognizer is None:
            self.local_recognizer = init_model(self.device, "weights/plate_rec_color.pth", True)

        # 显存优化初始化
        torch.cuda.empty_cache()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        # 加载全局参数
        global_params = self.server.get_global_models()
        try:
            self.local_detector.load_state_dict(global_params["detector"])
            self.local_recognizer.load_state_dict(global_params["recognizer"])
        except RuntimeError as e:
            print(f"参数加载失败: {str(e)}")
            return None

        # 初始化优化器和损失函数
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.local_recognizer.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )
        ctc_loss = nn.CTCLoss(blank=0)  # blank对应空白标签
        color_criterion = nn.CrossEntropyLoss()

        try:
            for epoch in range(epochs):
                self.local_recognizer.train()
                total_loss = 0.0
                valid_batches = 0
                optimizer.zero_grad()

                for batch_idx, batch in enumerate(self.loader):
                    # 跳过无效批次
                    if batch is None or not self._validate_batch(batch) or len(batch['image']) == 0:
                        continue

                    try:
                        images = batch['image'].numpy().astype(np.uint8)
                        bboxes = batch['bbox'].numpy().astype(int)
                        plate_numbers = batch['plate_number']
                        plate_colors = batch['plate_color']
                    except Exception as e:
                        self._handle_error(e, batch, 0)
                        continue

                    batch_loss = 0
                    valid_samples = 0

                    for idx in range(len(images)):
                        try:
                            # ROI提取
                            roi = self._safe_extract_roi(images[idx], bboxes[idx])
                            if roi is None:
                                continue

                            # 预处理
                            with torch.cuda.amp.autocast():
                                processed_img = image_processing(roi, self.device)
                                if processed_img.dim() == 3:
                                    processed_img = processed_img.unsqueeze(0)

                                # 前向传播
                                plate_logits, color_logits = self.local_recognizer(processed_img)

                                # 调试输出
                                print(f"[Pre] Plate logits shape: {plate_logits.shape}")

                                # 调整CTCLoss需要的维度
                                if plate_logits.dim() == 2:  # [T,C]
                                    plate_logits = plate_logits.unsqueeze(1)  # [T,1,C]
                                plate_logits = plate_logits.permute(1, 0, 2)  # [N,T,C]

                                # 强制序列长度为8
                                T = plate_logits.size(1)
                                if T < 8:
                                    plate_logits = torch.nn.functional.pad(plate_logits, (0, 0, 0, 8 - T, 0, 0))
                                elif T > 8:
                                    plate_logits = plate_logits[:, :8, :]

                                # 转换为log_softmax
                                plate_logits = torch.log_softmax(plate_logits, dim=2)
                                print(f"[Post] Plate logits shape: {plate_logits.shape}")

                            # 生成目标序列
                            target_indices = self._str_to_indices(plate_numbers[idx]).unsqueeze(0)  # [1,8]
                            print(f"Target indices shape: {target_indices.shape}")

                            # 计算CTCLoss
                            input_lengths = torch.full((1,), 8, dtype=torch.long, device=self.device)
                            target_lengths = torch.full((1,), 8, dtype=torch.long, device=self.device)

                            plate_loss = ctc_loss(
                                plate_logits,  # [N,T,C] = [1,8,78]
                                target_indices,  # [N,S] = [1,8]
                                input_lengths,
                                target_lengths
                            )

                            # 颜色分类损失
                            color_loss = self._calc_color_loss(color_logits, plate_colors[idx])

                            # 总损失
                            total_loss = plate_loss + 0.5 * color_loss
                            batch_loss += total_loss.item()
                            valid_samples += 1

                            # 反向传播
                            (total_loss / 4).backward()  # 假设梯度累积步数为4
                            torch.nn.utils.clip_grad_norm_(self.local_recognizer.parameters(), max_norm=1.0)

                            # 及时释放内存
                            del processed_img, plate_logits, color_logits
                            torch.cuda.empty_cache()

                        except Exception as e:
                            self._handle_error(e, batch, idx)
                            continue

                    if valid_samples > 0:
                        # 参数更新
                        if (batch_idx + 1) % 4 == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                        # 统计损失
                        avg_batch_loss = batch_loss / valid_samples
                        total_loss += avg_batch_loss
                        valid_batches += 1

                    # 定期显存清理
                    if batch_idx % 2 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if valid_batches > 0:
                    avg_epoch_loss = total_loss / valid_batches
                    print(f"Client {self.client_id} Epoch {epoch + 1} | Loss: {avg_epoch_loss:.4f}")

        except RuntimeError as e:
            print(f"训练异常: {str(e)}")
            traceback.print_exc()
            return None

        finally:
            # 资源清理
            self.local_detector.to('cpu')
            self.local_recognizer.to('cpu')
            del optimizer, ctc_loss, color_criterion
            torch.cuda.empty_cache()

        return {
            "detector": copy.deepcopy(self.local_detector.state_dict()),
            "recognizer": copy.deepcopy(self.local_recognizer.state_dict()),
            "sample_num": len(self.dataset)
        }

    # ---------- 辅助方法 ---------- #
    def initialize_models(self):
        """显式初始化模型方法"""
        if self.local_detector is None:
            print(f"Client {self.client_id} 初始化检测模型...")
            self.local_detector = load_model("weights/yolov8s.pt", self.device)
        if self.local_recognizer is None:
            print(f"Client {self.client_id} 初始化识别模型...")
            self.local_recognizer = init_model(self.device, "weights/plate_rec_color.pth", is_color=True)

    def _validate_model_output(self, plate_logits):
        # 统一处理为3D张量 [batch_size=1, seq_len=8, num_classes=78]
        if plate_logits.dim() == 2:
            plate_logits = plate_logits.unsqueeze(0)

        seq_len = plate_logits.size(1)
        if seq_len != 8:
            if seq_len < 8:
                plate_logits = torch.nn.functional.pad(plate_logits, (0, 0, 0, 8 - seq_len, 0, 0))
            else:
                plate_logits = plate_logits[:, :8, :]

        return plate_logits

    # =============== 修改后的_safe_extract_roi方法 =============== #
    def _safe_extract_roi(self, img_np, bbox):
        """优化内存对齐和边界处理"""
        h, w = img_np.shape[:2]

        # 边界保护
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        # 动态对齐策略
        align_granularity = 64  # 64字节对齐
        bytes_per_pixel = 3
        stride = (x2 - x1) * bytes_per_pixel

        # 计算需要补齐的字节数
        padding = (align_granularity - (stride % align_granularity)) % align_granularity
        new_width = (stride + padding) // bytes_per_pixel

        # 调整右边界
        x2 = min(x1 + new_width, w)
        if x2 - x1 < 32 or y2 - y1 < 16:
            return None

        try:
            # 内存对齐提取
            roi = img_np[y1:y2, x1:x2]
            if roi.size == 0:
                return None

            # 内存布局优化
            if not roi.flags['C_CONTIGUOUS']:
                roi = np.ascontiguousarray(roi)

            # 尺寸标准化
            roi = cv2.resize(roi, (168, 48), interpolation=cv2.INTER_AREA)

            # 数据类型转换
            return roi.astype(np.float32) / 255.0  # 添加归一化

        except Exception as e:
            print(f"ROI提取失败: {str(e)}")
            return None

    def _validate_batch(self, batch):
        """验证批次数据结构"""
        required_keys = ['image', 'bbox', 'plate_number', 'plate_color']
        return all(key in batch for key in required_keys) and \
            len(batch['image']) == len(batch['bbox']) == \
            len(batch['plate_number']) == len(batch['plate_color'])

    def _handle_error(self, error, batch, idx):
        """增强的错误处理方法"""
        self.error_count += 1
        error_info = f"""
        [ERROR] Client {self.client_id} Error (count={self.error_count}):
        - Image: {batch.get('image_path', ['unknown'])[idx]}
        - Bbox: {batch['bbox'][idx].numpy() if 'bbox' in batch else 'N/A'}
        - Error: {str(error)}
        """
        print(error_info)
        self._log_debug_image(batch['image'][idx].numpy(), f"error_{self.error_count}")


    def _calc_color_loss(self, pred_color_logits, target_color):
        """基于颜色分类输出计算损失"""
        color_map = {'黑色': 0, '蓝色': 1, '绿色': 2, '白色': 3, '黄色': 4}
        target_idx = torch.tensor(color_map[target_color], device=self.device)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred_color_logits, target_idx)
        return loss

    def _str_to_indices(self, target_str):
        """生成符合CTCLoss要求的target tensor"""
        max_length = 8
        indices = []
        for char in target_str[:max_length]:
            indices.append(plateName.index(char) if char in plateName else 0)
        indices += [0] * (max_length - len(indices))
        return torch.tensor(indices, device=self.device, dtype=torch.long)  # 保持为1D [8]

    # 在计算CTCLoss前调整target格式
    target_indices = self._str_to_indices(plate_numbers[idx])
    target_indices = target_indices.view(1, -1)  # 显式转为2D [1,8]
    
    def _validate_batch(self, batch):
        """增强校验逻辑"""
        required_keys = ['image', 'bbox', 'plate_number', 'plate_color']
        if not all(key in batch for key in required_keys):
            return False
        if len(batch['plate_number']) == 0:
            return False
        return True


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