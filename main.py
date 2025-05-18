import os
import time
import torch
from server import FedServer
from client import FedClient
from utils import PlateDataset


def main():
    # 初始化
    server = FedServer()
    clients = [FedClient(i, './processed', server) for i in range(5)]
    test_set = PlateDataset('./processed', mode='test')

    # 训练参数
    num_rounds = 10
    local_epochs = 2

    # 日志文件
    log_file = open("fed_log.txt", "w")

    for round in range(num_rounds):
        start_time = time.time()
        print(f"\n=== Round {round + 1}/{num_rounds} ===")

        # 客户端训练
        updates = []
        for client in clients:
            print(f"Training Client {client.client_id}...")
            update = client.local_train(epochs=local_epochs)
            updates.append(update)

        # 聚合更新
        server.aggregate(updates)

        # 评估
        acc = server.evaluate(test_set)
        time_cost = time.time() - start_time

        # 记录结果
        log_str = f"Round {round + 1}, Acc: {acc:.2%}, Time: {time_cost:.1f}s\n"
        print(log_str)
        log_file.write(log_str)

        # 保存模型
        if (round + 1) % 5 == 0:
            torch.save(server.global_detector.state_dict(), f"models/detector_round{round + 1}.pt")
            torch.save(server.global_recognizer.state_dict(), f"models/recognizer_round{round + 1}.pt")

    log_file.close()


if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("models", exist_ok=True)
    main()