import os
import time
import torch
import copy
import cv2
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from plate_recognition.plate_rec import image_processing, plateName, init_model
from utils import PlateDataset
import traceback
from detect_rec_plate import load_model

DEBUG_MODE = True
DEBUG_IMAGE_DIR = "./debug_images"
os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)

def custom_collate(batch):
    valid_batch = [b for b in batch if (
        isinstance(b, dict) and
        'image' in b and
        'bbox' in b and
        'plate_number' in b and
        'plate_color' in b
    )]
    if len(valid_batch) == 0:
        return None
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
        self.ctc_loss = nn.CTCLoss(blank=0)
        self.local_detector = None
        self.local_recognizer = None
        self.dataset = PlateDataset(data_root, client_id=client_id, mode='train')
        self.loader = DataLoader(
            self.dataset,
            batch_size=2,
            collate_fn=custom_collate,
            num_workers=0,
            sampler=torch.utils.data.sampler.SequentialSampler(
                [i for i in range(len(self.dataset))
                 if (self.dataset[i] is not None and 'plate_number' in self.dataset[i])]
            )
        )

    def _log_debug_image(self, img, prefix=""):
        if DEBUG_MODE:
            filename = f"{DEBUG_IMAGE_DIR}/client{self.client_id}_{prefix}_{time.strftime('%Y%m%d%H%M%S')}_{np.random.randint(10000)}.jpg"
            cv2.imwrite(filename, img)
            return filename
        return ""

    def local_train(self, epochs=1):
        try:
            if self.local_detector is None:
                self.local_detector = load_model("weights/yolov8s.pt", self.device)
            if self.local_recognizer is None:
                self.local_recognizer = init_model(self.device, "weights/plate_rec_color.pth", True)

            torch.cuda.empty_cache()
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            global_params = self.server.get_global_models()
            detector_keys = set(self.local_detector.state_dict().keys())
            received_det_keys = set(global_params["detector"].keys())
            if detector_keys != received_det_keys:
                print(f"检测器参数不匹配: 缺失 {detector_keys - received_det_keys} 多余 {received_det_keys - detector_keys}")
                return None
            for k, v in global_params["detector"].items():
                if v.shape != self.local_detector.state_dict()[k].shape:
                    raise RuntimeError(f"检测器参数形状不匹配: {k} {v.shape} vs {self.local_detector.state_dict()[k].shape}")
            self.local_detector.load_state_dict(global_params["detector"])
            self.local_recognizer.load_state_dict(global_params["recognizer"])

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.local_recognizer.parameters()),
                lr=1e-4,
                weight_decay=1e-5
            )
            grad_accum_steps = 4

            for epoch in range(epochs):
                self.local_recognizer.train()
                total_loss = 0.0
                valid_batches = 0
                optimizer.zero_grad()
                for batch_idx, batch in enumerate(self.loader):
                    try:
                        if batch is None or not self._validate_batch(batch) or len(batch['image']) == 0:
                            continue
                        images = batch['image'].numpy().astype(np.uint8)
                        bboxes = batch['bbox'].numpy().astype(int)
                        plate_numbers = batch['plate_number']
                        plate_colors = batch['plate_color']
                        batch_loss = 0
                        valid_samples = 0
                        for idx in range(len(images)):
                            try:
                                roi = self._safe_extract_roi(images[idx], bboxes[idx])
                                if roi is None:
                                    continue
                                processed_img = image_processing(roi, self.device)
                                if processed_img.dim() == 3:
                                    processed_img = processed_img.unsqueeze(0)
                                if processed_img.shape[2:] != (48, 168):
                                    continue
                                plate_logits, color_logits = self.local_recognizer(processed_img)
                                plate_logits = plate_logits.permute(1, 0, 2)
                                if plate_logits.size(0) < 8:
                                    plate_logits = nn.functional.pad(plate_logits, (0, 0, 0, 0, 0, 8 - plate_logits.size(0)))
                                elif plate_logits.size(0) > 8:
                                    plate_logits = plate_logits[:8]
                                target_indices = self._str_to_indices(plate_numbers[idx])
                                if target_indices is None:
                                    continue
                                target_indices = target_indices.unsqueeze(0)
                                input_lengths = torch.full((1,), 8, dtype=torch.long, device=self.device)
                                target_lengths = torch.full((1,), 8, dtype=torch.long, device=self.device)
                                # 在损失计算后加断言
                                assert not torch.isnan(plate_logits).any(), "plate_logits contains nan"
                                assert not torch.isnan(target_indices).any(), "target_indices contains nan"
                                assert not torch.isinf(plate_logits).any(), "plate_logits contains inf"
                                assert not torch.isinf(target_indices).any(), "target_indices contains inf"
                                plate_loss = self.ctc_loss(
                                    plate_logits.log_softmax(2),
                                    target_indices,
                                    input_lengths,
                                    target_lengths
                                )
                                if torch.isnan(plate_loss):
                                    print("[ERROR] plate_loss is nan! Skipping sample.")
                                    continue
                                (plate_loss / grad_accum_steps).backward()
                                batch_loss += plate_loss.item()
                                valid_samples += 1
                                del processed_img, plate_logits, color_logits
                                torch.cuda.empty_cache()
                            except Exception as e:
                                self._handle_error(e, batch, idx)
                                continue
                        if valid_samples > 0:
                            torch.nn.utils.clip_grad_norm_(self.local_recognizer.parameters(), max_norm=1.0)
                            if (batch_idx + 1) % grad_accum_steps == 0:
                                optimizer.step()
                                optimizer.zero_grad()
                            avg_batch_loss = batch_loss / valid_samples
                            total_loss += avg_batch_loss
                            valid_batches += 1
                        if batch_idx % 2 == 0:
                            torch.cuda.empty_cache()
                    except Exception as batch_e:
                        print(f"[Batch Error] client {self.client_id}, epoch {epoch} batch {batch_idx}")
                        traceback.print_exc()
                        continue
                if valid_batches > 0:
                    avg_epoch_loss = total_loss / valid_batches
                    print(f"Client {self.client_id} Epoch {epoch + 1} | Loss: {avg_epoch_loss:.4f} | Samples: {valid_samples}")
        except Exception as e:
            print(f"[Train Crash] client {self.client_id}, outer try-catch")
            traceback.print_exc()
            return None
        finally:
            if hasattr(self, "local_detector") and self.local_detector is not None:
                self.local_detector.to('cpu')
            if hasattr(self, "local_recognizer") and self.local_recognizer is not None:
                self.local_recognizer.to('cpu')
            torch.cuda.empty_cache()

        return {
            "detector": copy.deepcopy(self.local_detector.state_dict()),
            "recognizer": copy.deepcopy(self.local_recognizer.state_dict()),
            "sample_num": len(self.dataset)
        }

    def initialize_models(self):
        if self.local_detector is None:
            print(f"Client {self.client_id} 初始化检测模型...")
            self.local_detector = load_model("weights/yolov8s.pt", self.device)
        if self.local_recognizer is None:
            print(f"Client {self.client_id} 初始化识别模型...")
            self.local_recognizer = init_model(self.device, "weights/plate_rec_color.pth", is_color=True)

    def _safe_extract_roi(self, img_np, bbox):
        h, w = img_np.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        align_granularity = 64
        bytes_per_pixel = 3
        stride = (x2 - x1) * bytes_per_pixel
        padding = (align_granularity - (stride % align_granularity)) % align_granularity
        new_width = (stride + padding) // bytes_per_pixel
        x2 = min(x1 + new_width, w)
        if x2 - x1 < 32 or y2 - y1 < 16:
            return None
        try:
            roi = img_np[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            if not roi.flags['C_CONTIGUOUS']:
                roi = np.ascontiguousarray(roi)
            roi = cv2.resize(roi, (168, 48), interpolation=cv2.INTER_AREA)
            return roi.astype(np.float32) / 255.0
        except Exception as e:
            print(f"ROI提取失败: {str(e)}")
            return None

    def _validate_batch(self, batch):
        required_keys = ['image', 'bbox', 'plate_number', 'plate_color']
        return all(key in batch for key in required_keys) and \
            len(batch['image']) == len(batch['bbox']) == \
            len(batch['plate_number']) == len(batch['plate_color']) and \
            len(batch['plate_number']) > 0

    def _handle_error(self, error, batch, idx):
        self.error_count += 1
        error_info = f"""
        [ERROR] Client {self.client_id} Error (count={self.error_count}):
        - Image: {batch.get('image_path', ['unknown'])[idx]}
        - Bbox: {batch['bbox'][idx].numpy() if 'bbox' in batch else 'N/A'}
        - Error: {str(error)}
        """
        print(error_info)
        img = batch['image'][idx].numpy() if hasattr(batch['image'][idx], 'numpy') else batch['image'][idx]
        self._log_debug_image(img, f"error_{self.error_count}")

    def _str_to_indices(self, target_str):
        max_length = 8
        indices = []
        valid_chars = 0
        for char in target_str[:max_length]:
            if char in plateName:
                indices.append(plateName.index(char))
                valid_chars += 1
            else:
                indices.append(0)
        indices += [0] * (max_length - len(indices))
        if valid_chars == 0:
            return None
        return torch.tensor(indices, dtype=torch.long, device=self.device)

def print_data_sample(dataset, index=0):
    sample = dataset[index]
    print("\n=== 数据样本调试 ===")
    print(f"Image path: {sample.get('image_path', 'unknown')}")
    print(f"Original bbox: {sample['bbox']}")
    print(f"Plate number: {sample['plate_number']}")
    print(f"Plate color: {sample['plate_color']}")
    img = sample['image']
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    x1, y1, x2, y2 = sample['bbox']
    roi = img[y1:y2, x1:x2]
    cv2.imshow("Sample ROI", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_client = FedClient(0, "processed", None)
    print_data_sample(test_client.dataset)