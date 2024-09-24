import torch
from torchsummary import summary
import matplotlib.pyplot as plt
from io import StringIO
import sys

# 假设你的模型在 'nets' 文件夹中
def load_model(model_name):
    if model_name == 'VGG16':
        from nets.VGG import VGG
        return VGG('VGG16')
    elif model_name == 'VGG19':
        from nets.VGG import VGG
        return VGG('VGG19')
    elif model_name == 'ResNet18':
        from nets.ResNet import ResNet18
        return ResNet18()
    elif model_name == 'ResNet34':
        from nets.ResNet import ResNet34
        return ResNet34()
    elif model_name == 'LeNet5':
        from nets.LeNet5 import LeNet5
        return LeNet5()
    elif model_name == 'AlexNet':
        from nets.AlexNet import AlexNet
        return AlexNet()
    elif model_name == 'DenseNet':
        from nets.DenseNet import densenet_cifar
        return densenet_cifar()
    elif model_name == 'MobileNetv1':
        from nets.MobileNetv1 import MobileNet
        return MobileNet()
    elif model_name == 'MobileNetv2':
        from nets.MobileNetv2 import MobileNetV2
        return MobileNetV2()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

if __name__ == '__main__':
    # 设定模型名称
    model_name = 'VGG16'  # 你可以根据需要更改这个名称

    # 创建模型
    model = load_model(model_name)

    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 捕获模型摘要输出
    buffer = StringIO()
    sys.stdout = buffer  # 重定向标准输出到 StringIO
    summary(model, (3, 32, 32))  # CIFAR-10 图像的输入尺寸
    sys.stdout = sys.__stdout__  # 恢复标准输出

    # 获取摘要字符串
    summary_str = buffer.getvalue()

    # 创建图形
    plt.figure(figsize=(12, 8))
    plt.text(0.01, 0.05, summary_str, {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')  # 不显示坐标轴
    # plt.title(f'Model Summary for {model_name}', fontsize=16)  # 注释掉标题行

    # 保存为图片
    output_image = f"{model_name}_summary.png"
    plt.savefig(output_image, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Model summary saved as image: {output_image}")
