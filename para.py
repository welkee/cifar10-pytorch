import torch
from torchsummary import summary

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
    model_name = 'ResNet34'  # 你可以根据需要更改这个名称

    # 创建模型
    model = load_model(model_name)

    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 打印模型的参数数量和 FLOPS
    print("Model Summary:")
    summary(model, (3, 32, 32))  # CIFAR-10 图像的输入尺寸