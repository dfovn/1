import os
import sys
import time
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from server import FedServer
from client import FedClient
from utils import PlateDataset


def main():

    # =============== 初始化阶段 =============== #
    print("🚀 系统初始化中...")
    start_time = time.time()

    # 初始化联邦学习服务器
    server = FedServer()
    print(f"✅ 服务器初始化完成（耗时：{time.time() - start_time:.2f}s）")

    # =============== 客户端初始化 =============== #
    print("\n🚀 正在初始化客户端...")
    client_init_start = time.time()

    clients = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 并行创建客户端实例
        futures = {executor.submit(FedClient, i, './processed', server): i for i in range(5)}

        for future in as_completed(futures):
            client_id = futures[future]
            try:
                client = future.result()
                clients.append(client)
                print(f"  客户端 {client_id} 初始化完成")
            except Exception as e:
                print(f"⚠️ 客户端 {client_id} 初始化失败: {str(e)}")

    print(f"✅ 所有客户端初始化完成（共耗时：{time.time() - client_init_start:.2f}s）")

    # =============== 模型预加载 =============== #
    print("\n🔥 正在预加载模型...")
    model_load_start = time.time()

    # 显式初始化所有客户端的模型
    for idx, client in enumerate(clients):
        try:
            client.initialize_models()
            print(f"  客户端 {idx} 模型加载成功")
        except Exception as e:
            print(f"⚠️ 客户端 {idx} 模型加载失败: {str(e)}")
            continue

    print(f"✅ 模型预加载完成（共耗时：{time.time() - model_load_start:.2f}s）")

    # =============== 数据准备 =============== #
    print("\n📂 正在加载测试数据集...")
    test_set = PlateDataset('./processed', mode='test')
    print(f"✅ 测试集加载完成（样本数：{len(test_set)}）")

    # =============== 预热阶段 =============== #
    print("\n🔥 执行系统预热...")
    warmup_start = time.time()

    if clients:
        # 使用第一个客户端进行预热
        sample_client = clients[0]

        # 并行执行预热操作
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(lambda: sample_client.loader.dataset[0]),  # 数据加载预热
                executor.submit(  # 模型推理预热
                    lambda: sample_client.local_recognizer(
                        torch.randn(1, 3, 48, 168).to(sample_client.device)
                    )
                )
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                    print("  预热操作完成")
                except Exception as e:
                    print(f"⚠️ 预热失败: {str(e)}")

    print(f"✅ 预热完成（耗时：{time.time() - warmup_start:.2f}s）")

    # =============== 训练配置 =============== #
    num_rounds = 10
    local_epochs = 2
    model_save_dir = "models"
    os.makedirs(model_save_dir, exist_ok=True)

    # =============== 联邦训练循环 =============== #
    print(f"\n🏁 开始联邦训练（总轮次：{num_rounds}）")

    for round_idx in range(num_rounds):
        round_start = time.time()
        print(f"\n=== 第 {round_idx + 1}/{num_rounds} 轮 ===")

        try:
            # =============== 客户端本地训练 =============== #
            updates = []
            print(f"⌛ 客户端本地训练中...")
            train_start = time.time()

            with ThreadPoolExecutor() as executor:
                future_to_client = {
                    executor.submit(client.local_train, local_epochs): client
                    for client in clients
                }

                for future in as_completed(future_to_client, timeout=7200):
                    client = future_to_client[future]
                    try:
                        update = future.result()
                        updates.append(update)
                        print(f"  客户端 {client.client_id} 训练完成")
                    except Exception as e:
                        print(f"⚠️ 客户端 {client.client_id} 训练失败: {str(e)}")

            print(f"✅ 本地训练完成（耗时：{time.time() - train_start:.2f}s）")

            # =============== 模型聚合 =============== #
            if updates:
                print("🔄 正在聚合模型更新...")
                aggregate_start = time.time()
                server.aggregate(updates)
                print(f"✅ 模型聚合完成（耗时：{time.time() - aggregate_start:.2f}s）")

                # =============== 模型评估 =============== #
                print("📊 正在评估模型性能...")
                eval_start = time.time()
                accuracy = server.evaluate(test_set)
                print(f"✅ 评估完成 | 准确率：{accuracy:.2%}（耗时：{time.time() - eval_start:.2f}s）")
            else:
                print("⚠️ 本轮无有效更新，跳过聚合和评估")

            # =============== 模型保存 =============== #
            if (round_idx + 1) % 5 == 0:
                version = (round_idx + 1) // 5
                save_start = time.time()
                print(f"💾 正在保存第 {version} 版模型...")

                detector_path = os.path.join(model_save_dir, f"detector_v{version}_round{round_idx + 1}.pt")
                recognizer_path = os.path.join(model_save_dir, f"recognizer_v{version}_round{round_idx + 1}.pt")

                torch.save(server.global_detector.state_dict(), detector_path)
                torch.save(server.global_recognizer.state_dict(), recognizer_path)
                print(f"✅ 模型保存完成（耗时：{time.time() - save_start:.2f}s）")

            # 每轮结束后添加
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 确保CUDA操作完成
                torch.cuda.empty_cache()  # 清空缓存

            # 内存监控
            print(f"显存状态 - 已分配: {torch.cuda.memory_allocated() / 1e6:.2f}MB, "
                    f"保留: {torch.cuda.memory_reserved() / 1e6:.2f}MB")

            # 添加内存诊断
            if round_idx % 2 == 0:
                print(f"内存状态 - 已分配：{torch.cuda.memory_allocated() / 1e6:.2f}MB, "
                      f"缓存：{torch.cuda.memory_reserved() / 1e6:.2f}MB")

        except Exception as e:
            print(f"⚠️ 第 {round_idx + 1} 轮训练异常: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"⏱️ 本轮总耗时：{time.time() - round_start:.2f}s")

    print("\n🎉 联邦训练完成！")


if __name__ == "__main__":
    # =============== 系统配置优化 =============== #
    if sys.platform.startswith('win'):
        import ctypes

        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, 1 << 28, 1 << 30)  # 限制内存工作集
        # Windows系统优化
        import multiprocessing

        multiprocessing.freeze_support()
        multiprocessing.set_start_method('spawn', force=True)
        print("⚠️ Windows系统已启用兼容模式")
    else:
        # Linux/Mac系统优化
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        print("🐧 Linux/Mac系统已启用加速模式")

    # =============== 启动训练 =============== #
    total_start = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    finally:
        print(f"\n🏁 总运行时间：{time.time() - total_start:.2f}秒")