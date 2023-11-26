python

# ECA-Net 注意力模块
class ECAAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)  # 将特征图变形为适合1D卷积的形状
        # 1D卷积
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        # Sigmoid函数获取权重
        y = self.sigmoid(y)
        # 通道权重和原始特征图相乘
        return x * y.expand_as(x)

# CBAM 注意力模块
class CBAMAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAMAttention, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# CBAM的通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

# CBAM的空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return x * self.sigmoid(x)

# 使用这些模块的YOLOv5改进模型可能看起来像这样
class YOLOv5ECA_CBAM(nn.Module):
    def __init__(self):
        super(YOLOv5ECA_CBAM, self).__init__()
        # 假设我们有一个预训练的YOLOv5模型
        # self.yolov5 = ...
        # 在YOLOv5的适当层添加ECA和CBAM模块
        self.eca = ECAAttention(channels=...)  # 填入对应通道数
        self.cbam = CBAMAttention(channels=...)  # 填入对应通道数

    def forward(self, x):
        # 假设我们通过YOLOv5的一部分
        # x = self.yolov5(x)
        # 应用ECA注意力模块
        x = self.eca(x)
        # 应用CBAM注意力模块
        x = self.cbam(x)
        # 继续YOLOv5的其余部分
        # x = ...
        return x
