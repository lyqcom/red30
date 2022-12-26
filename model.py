import mindspore.nn as nn


class REDNet30(nn.Cell):
    def __init__(self, num_layers=15, num_features=64):
        super(REDNet30, self).__init__()
        self.num_layers = num_layers

        self.conv_layers_1 = nn.SequentialCell(
            [nn.Conv2d(3, num_features, kernel_size=3, stride=2, pad_mode="pad", padding=1),
             nn.ReLU()])
        self.conv_layers_2 = nn.SequentialCell(
            [nn.Conv2d(num_features, num_features, kernel_size=3, pad_mode="pad", padding=1),
             nn.ReLU()])

        self.deconv_layers_1 = nn.SequentialCell(
            [nn.Conv2dTranspose(num_features, num_features, kernel_size=3, pad_mode="pad", padding=1),
             nn.ReLU()])
        self.deconv_layers_2 = nn.SequentialCell(
            [nn.Conv2dTranspose(num_features, 3, kernel_size=3, stride=2, pad_mode="pad", padding=1),
             nn.Conv2d(3, 3, kernel_size=4, stride=1, pad_mode="pad", padding=2)])

        self.relu = nn.ReLU()

    def construct(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            if i == 0:
                x = self.conv_layers_1(x)
            else:
                x = self.conv_layers_2(x)
            if (i + 1) % 2 == 0 and len(conv_feats) < 7:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            if i == (self.num_layers - 1):
                x = self.deconv_layers_2(x)
            else:
                x = self.deconv_layers_1(x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)
        return x


if __name__ == '__main__':
    from mindspore import Tensor, context
    import numpy as np

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=4)
    net = REDNet30()
    x = Tensor(np.ones([2, 3, 180, 180]).astype(np.float32))
    y = net(x)
    print(y.shape)
