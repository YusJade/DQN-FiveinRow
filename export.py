import argparse
from datetime import datetime
import os
import loguru
import torch
from config import Config
from dqn import QNetwork


def export_to_onnx(cfg: Config, weight: str, output_path: str):
    # 加载 PyTorch 模型
    model = QNetwork(cfg.state_size, cfg.action_size, cfg.hidden_dim)
    model.load_state_dict(torch.load(weight, weights_only=True))
    model.eval()

    # 创建一个虚拟输入张量
    dummy_input = torch.randn(cfg.state_size)

    # 导出为 ONNX 格式
    torch.onnx.export(
        model,                      # PyTorch 模型
        dummy_input,                # 虚拟输入张量
        f'{output_path}/net.onnx',                # 输出 ONNX 文件路径
        export_params=True,         # 导出所有参数
        opset_version=11,           # ONNX opset 版本
        do_constant_folding=True,   # 是否执行常量折叠优化
        input_names=['input'],      # 输入节点名称
        output_names=['output'],    # 输出节点名称
        dynamic_axes={              # 动态轴设置
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    loguru.logger.info(f"ONNX model exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, required=True,
                        help="the path of weight file.")
    args = parser.parse_args()
    weight = args.weight
    str_date = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    output_path = f'./runs/export_{str_date}'
    os.makedirs(output_path, exist_ok=True)
    export_to_onnx(Config(), weight, output_path)
