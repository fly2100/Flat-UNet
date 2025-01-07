import torch
import torch.nn as nn

"""
Flat U-Net
"""


class Flat_Unet(nn.Module):
    """
    Flat U-Net

    Attributes:
    -----------
    num_layers : int
        Number of encoding and decoding layers in the model.
    flat_channels : int
        Number of channels used in the encoding and decoding blocks.
    out_channels : int
        Number of output channels for the final segmentation map.
    is_simp : bool
        Determines whether to use the simplified channel attention block (SCA_ConvBlock)
        or the more complex version (CSA_ConvBlock).

    Methods:
    --------
    __init__():
        pass
        
    forward(inputs):
        pass
    """

    def __init__(self, simp_list, num_layers=4, flat_channels=4, out_channels=1):
        super().__init__()
        self.simp_list = simp_list
        self.conv = nn.Conv2d(1, flat_channels, kernel_size=3, padding=1)

        self.e_list = nn.ModuleList()
        self.d_list = nn.ModuleList()
        for i in range(num_layers):
            is_simp_encoder = self.simp_list[i]
            is_simp_decoder = self.simp_list[num_layers - i - 1]
            self.e_list.append(EncoderBlock(flat_channels, is_simp_encoder))
            self.d_list.append(DecoderBlock(flat_channels, is_simp_decoder))

        self.b = SCA_ConvBlock(flat_channels) if self.simp_list[-1] else CSA_ConvBlock(flat_channels)

        self.outputs = nn.Conv2d(flat_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        x = self.conv(inputs)

        enc_features = []
        for e_layer in self.e_list:
            s, x = e_layer(x)
            enc_features.append(s)

        b = self.b(x)

        for i, d_layer in enumerate(self.d_list):
            b = d_layer(b, enc_features[-(i + 1)])

        outputs = self.outputs(b)
        return outputs


class CSA_ConvBlock(nn.Module):
    """
    Channel Self-Attention (CSA) ConvBlock: Applies self-attention across channels to 
    enhance feature representation.

    Attributes:
    -----------
    c : int
        Number of input and output channels for the block.

    Methods:
    --------
    __init__():
        pass
       
    forward(inputs):
        pass
    """

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.fq = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.fk = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.fv = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        fq = self.fq(inputs)
        fk = self.fk(inputs)
        fv = self.fv(inputs)

        h, w = inputs.size(2), inputs.size(3)
        fq = fq.unsqueeze(2)
        fk = fk.unsqueeze(1).permute(0, 1, 2, 4, 3)
        f_sim_tensor = torch.matmul(fq, fk) / (fq.size(-1) ** 0.5)
        f_sum_tensor = torch.sum(f_sim_tensor, dim=2)
        f_scores = torch.sum(f_sum_tensor, dim=(-2, -1)) / (h ** 2)
        scores = torch.softmax(f_scores, dim=1).unsqueeze(-1).unsqueeze(-1)

        r = (scores * fv) + inputs
        r = self.bn(r)
        r = self.relu(r)
        return r


class SCA_ConvBlock(nn.Module):
    """
    Simplified Channel Attention (SCA) ConvBlock: A simplified channel attention
    block for faster computation.

    Attributes:
    -----------
    c : int
        Number of input and output channels for the block.

    Methods:
    --------
    __init__():
        pass
       
    forward(inputs):
        pass
    """

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.fq = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.fk = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.fv = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        fq = self.fq(inputs)
        fk = self.fk(inputs)
        fv = self.fv(inputs)

        f_sim_tensor = torch.matmul(fq, fk.transpose(2, 3)) / (fq.size(-1) ** 0.5)
        f_sum_tensor = torch.sum(f_sim_tensor, dim=(2, 3))
        scores = torch.softmax(f_sum_tensor, dim=1).unsqueeze(2).unsqueeze(3)

        r = scores * fv + inputs
        r = self.bn(r)
        r = self.relu(r)
        return r


class EncoderBlock(nn.Module):
    """
    EncoderBlock: A building block for the encoder part of the Flat U-Net.

    Attributes:
    -----------
    c : int
        Number of input and output channels for the block.
    is_simp : bool
        Whether to use the simplified channel attention block.

    Methods:
    --------
    Methods:
    --------
    __init__():
        pass
       
    forward(inputs):
        pass
    """

    def __init__(self, c, is_simp=True):
        super().__init__()
        self.conv = SCA_ConvBlock(c) if is_simp else CSA_ConvBlock(c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    """
    DecoderBlock: A building block for the decoder part of the Flat U-Net.

    Attributes:
    -----------
    c : int
        Number of input and output channels for the block.
    is_simp : bool
        Whether to use the simplified channel attention block.

    Methods:
    --------
    __init__():
        pass
       
    forward(inputs):
        pass
    """

    def __init__(self, c, is_simp=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_down = nn.Conv2d(2 * c, c, kernel_size=1, stride=1, padding=0)
        self.conv = SCA_ConvBlock(c) if is_simp else CSA_ConvBlock(c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_down(x)
        x = self.conv(x)
        return x
