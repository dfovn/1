import copy
import torch
from tqdm import tqdm
from detect_rec_plate import load_model, det_rec_plate
from plate_recognition.plate_rec import init_model


class FedServer:
    _preloaded = False  # 类变量用于标记是否已预加载
    _detector = None
    _recognizer = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self._preloaded:
            print(f"Server using device: {self.device}")
            self._preload_models()
            FedServer._preloaded = True

        # 共享预加载的模型
        self.global_detector = copy.deepcopy(FedServer._detector)
        self.global_recognizer = copy.deepcopy(FedServer._recognizer)
        # 确保模型参数类型统一
        self._enforce_parameter_type()

    def _enforce_parameter_type(self):
        """强制模型参数为float32类型"""
        for param in self.global_detector.parameters():
            param.data = param.data.float()
        for param in self.global_recognizer.parameters():
            param.data = param.data.float()

    def _preload_models(self):
        """改进的模型预加载方法"""
        # YOLOv8检测模型
        FedServer._detector = load_model("weights/yolov8s.pt", self.device)
        FedServer._detector = FedServer._detector.to('cpu').float()

        # 车牌识别模型
        FedServer._recognizer = init_model(self.device, "weights/plate_rec_color.pth", True)
        FedServer._recognizer = FedServer._recognizer.to('cpu').float()

        # 确保参数连续性
        for p in FedServer._detector.parameters():
            p.data = p.data.cpu().float().contiguous()
        for p in FedServer._recognizer.parameters():
            p.data = p.data.cpu().float().contiguous()

        FedServer._detector.eval()
        FedServer._recognizer.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def aggregate(self, client_updates):
        """改进的模型聚合方法"""
        valid_updates = [u for u in client_updates if all(k in u for k in ['detector', 'recognizer', 'sample_num'])]
        if not valid_updates:
            print("No valid updates, skipping aggregation")
            return

        # 聚合检测模型
        detector_state = copy.deepcopy(self.global_detector.state_dict())
        for key in detector_state:
            if 'model.24' in key:
                try:
                    stacked = torch.stack([u['detector'][key].float() for u in valid_updates])
                    detector_state[key] = torch.mean(stacked, dim=0).contiguous()
                except RuntimeError as e:
                    print(f"聚合失败 {key}: {str(e)}")
                    continue
        self.global_detector.load_state_dict(detector_state)

        # 聚合识别模型
        recognizer_state = copy.deepcopy(self.global_recognizer.state_dict())
        total_samples = sum(u['sample_num'] for u in valid_updates)

        for key in recognizer_state:
            if key not in valid_updates[0]['recognizer']:
                continue
            try:
                weighted_sum = torch.stack([
                    u['recognizer'][key].float() * u['sample_num'] for u in valid_updates
                ]).sum(dim=0)
                recognizer_state[key] = (weighted_sum / total_samples).contiguous()
            except Exception as e:
                print(f"参数 {key} 聚合失败: {str(e)}")
                continue

        self.global_recognizer.load_state_dict(recognizer_state)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def get_global_models(self):
        """获取全局模型（添加内存优化）"""
        # 确保返回的state_dict是CPU上的
        detector_state = copy.deepcopy(self.global_detector.cpu().state_dict())
        recognizer_state = copy.deepcopy(self.global_recognizer.cpu().state_dict())

        # 恢复模型到原始设备
        self.global_detector.to(self.device)
        self.global_recognizer.to(self.device)

        return {
            "detector": detector_state,
            "recognizer": recognizer_state
        }

    def get_global_models(self):
        return {
            "detector": copy.deepcopy(self.global_detector.state_dict()),
            "recognizer": copy.deepcopy(self.global_recognizer.state_dict())
        }

    def evaluate(self, dataset):
        """使用原有检测识别流程进行评估（添加内存优化）"""
        self.global_detector.eval()
        self.global_recognizer.eval()

        correct = 0
        total = 0

        # 显存优化配置
        with torch.cuda.amp.autocast(enabled=False):  # 禁用混合精度
            with torch.no_grad():
                for item in tqdm(dataset, desc="Evaluating"):
                    try:
                        # 使用原有检测识别流程
                        result_list = det_rec_plate(
                            item['image'],
                            item['image'].copy(),
                            self.global_detector,
                            self.global_recognizer
                        )

                        # 每10次迭代清理一次缓存
                        if total % 10 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        if result_list:
                            pred = result_list[0]
                            if (pred['plate_no'] == item['plate_number'] and
                                    pred['plate_color'] == item['plate_color']):
                                correct += 1
                        total += 1
                    except Exception as e:
                        print(f"Evaluation error: {e}")
                        continue

        # 评估完成后释放资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return correct / total if total > 0 else 0.0

    # 新增内存诊断工具函数
    def print_memory_stats(prefix=""):
        if torch.cuda.is_available():
            print(f"{prefix} Memory - Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB, "
                  f"Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
        else:
            print(f"{prefix} CPU Memory in use")