# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


######################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


###########################################################################
## --------  Spatial Attention  ------------
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class sa_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(sa_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, bias=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

##########################################################################
## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
##---------- CSA Module ----------
class CSA(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8, bias=False, act=nn.PReLU()):
        super(CSA, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = sa_layer()

        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res += x
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res

# --- Downsampling block in GridDehazeNet  --- #
class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels, stride*in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out


# --- Upsampling block in GridDehazeNet  --- #
class UpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=stride, padding=1)
        self.conv = nn.Conv2d(in_channels, in_channels // stride, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return out


##########################################################################
class Conv_In(nn.Module):

    def __init__(self, in_ch, out_ch, k, s, p):
        super(Conv_In, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p)
        self.relu = nn.PReLU(out_ch)
        self.In = nn.InstanceNorm2d(num_features=out_ch)

    def forward(self, input):
        return self.relu(self.In(self.conv(input)))


##################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
class RGB_Feat(nn.Module):
    def __init__(self, act, bias):
        super(RGB_Feat, self).__init__()
        self.v = torch.nn.Parameter(torch.FloatTensor(4))

        self.layer1_1 = Conv_In(1, 16, 3, 1, 1)
        self.layer1_2 = Conv_In(1, 16, 3, 1, 1)
        self.layer1_3 = Conv_In(1, 16, 3, 1, 1)

        self.layer2_1 = Conv_In(16, 16, 3, 1, 1)
        self.layer2_2 = Conv_In(16, 16, 3, 1, 1)
        self.layer2_3 = Conv_In(16, 16, 3, 1, 1)

        self.attn_r = CSA(32)
        self.attn_g = CSA(32)
        self.attn_b = CSA(32)

        self.layer3_1 = Conv_In(32, 1, 3, 1, 1)
        self.layer3_2 = Conv_In(32, 1, 3, 1, 1)
        self.layer3_3 = Conv_In(32, 1, 3, 1, 1)

    def forward(self, x):
        input_1 = torch.unsqueeze(x[:, 0, :, :], dim=1)
        input_2 = torch.unsqueeze(x[:, 1, :, :], dim=1)
        input_3 = torch.unsqueeze(x[:, 2, :, :], dim=1)

        # layer 1
        l1_1 = self.layer1_1(input_1)  # 16
        l1_2 = self.layer1_2(input_2)  # 16
        l1_3 = self.layer1_3(input_3)  # 16

        # Input to layer 2
        input_l2_1 = l1_1 + self.v[0] * l1_2  # 16
        input_l2_2 = l1_2  # 16
        input_l2_3 = l1_3 + self.v[1] * l1_2  # 16

        # layer 2
        l2_1 = self.layer2_1(input_l2_1)  # 16
        l2_1 = self.attn_r(torch.cat((l2_1, l1_1), 1))  # 32

        l2_2 = self.layer2_2(input_l2_2)
        l2_2 = self.attn_g(torch.cat((l2_2, l1_2), 1))  # 32

        l2_3 = self.layer2_3(input_l2_3)
        l2_3 = self.attn_b(torch.cat((l2_3, l1_3), 1))  # 32

        # Input to layer 3
        input_l3_1 = l2_1 + self.v[2] * l2_2  # 32
        input_l3_2 = l2_2  # 32
        input_l3_3 = l2_3 + self.v[3] * l2_2  # 32

        # layer 3
        l3_1 = self.layer3_1(input_l3_1)
        l3_2 = self.layer3_2(input_l3_2)
        l3_3 = self.layer3_3(input_l3_3)

        l3 = torch.cat((l3_1, l3_2), 1)
        l3 = torch.cat((l3, l3_3), 1)

        output = l3

        return output

#################################################################################################################
class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=29, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=.20)

    def forward(self, x):
        x, input_x = x
        a = self.relu(self.conv1(self.relu(self.drop(self.conv(self.relu(self.drop(self.conv(x))))))))
        out = torch.cat((a, input_x), 1)
        return (out, input_x)


def StackBlock(block, layer_num):
    layers = []
    for _ in range(layer_num):
        layers.append(block())
    return nn.Sequential(*layers)



# --- Main model  --- #
class GridDehazeNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(GridDehazeNet, self).__init__()
        self.rdb_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.stride = stride
        self.depth_rate = depth_rate
        self.out_channels = out_channels
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, depth_rate*stride**(height-1)))), requires_grad=attention)
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb_out = RDB(depth_rate, num_dense_layer, growth_rate)
        rdb_in_channels = depth_rate
        for i in range(height):
            for j in range(width - 1):
                self.rdb_module.update({'{}_{}'.format(i, j): RDB(rdb_in_channels, num_dense_layer, growth_rate)})
            rdb_in_channels *= stride

        _in_channels = depth_rate
        for i in range(height - 1):
            for j in range(width // 2):
                self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
            _in_channels *= stride

        for i in range(height - 2, -1, -1):
            for j in range(width // 2, width):
                self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
            _in_channels //= stride
        
        
        act = nn.PReLU()
        bias = False
        
        #  RGB Features
        self.rgb_feature = RGB_Feat(act, bias)
        self.rgb_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)
        self.inputa = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.outputa = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = StackBlock(ConvBlock, 2)
        
    def forward(self, x):
        inp = self.conv_in(x)
        
        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        i, j = 0, 0
        
        x_index[0][0] = self.rdb_in(inp)

        for j in range(1, self.width // 2):
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, j-1)](x_index[0][j-1])

        for i in range(1, self.height):
            x_index[i][0] = self.downsample_module['{}_{}'.format(i-1, 0)](x_index[i-1][0])

        for i in range(1, self.height):
            for j in range(1, self.width // 2):
                channel_num = int(2**(i-1)*self.stride*self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.downsample_module['{}_{}'.format(i-1, j)](x_index[i-1][j])

        x_index[i][j+1] = self.rdb_module['{}_{}'.format(i, j)](x_index[i][j])
        k = j

        for j in range(self.width // 2 + 1, self.width):
            x_index[i][j] = self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1])

        for i in range(self.height - 2, -1, -1):
            channel_num = int(2 ** (i-1) * self.stride * self.depth_rate)
            x_index[i][k+1] = self.coefficient[i, k+1, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, k)](x_index[i][k]) + \
                              self.coefficient[i, k+1, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, k+1)](x_index[i+1][k+1], x_index[i][k].size())

        for i in range(self.height - 2, -1, -1):
            for j in range(self.width // 2 + 1, self.width):
                channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, j)](x_index[i+1][j], x_index[i][j-1].size())

        outb = self.rdb_out(x_index[i][j])
        outb = F.relu(self.conv_out(outb))
        
        rgb_out = self.rgb_feature(outb + x)
        rgb_out = F.relu(self.rgb_conv(rgb_out))
        
        input_x = rgb_out
        x1 = self.relu(self.inputa(rgb_out))
        out, _ = self.blocks((x1, input_x))
        outa = self.outputa(out)

        return outa