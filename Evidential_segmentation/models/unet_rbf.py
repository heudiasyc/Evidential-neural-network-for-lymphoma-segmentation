# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, export
from torch.nn.parameter import Parameter
import numpy as np

@export("monai.networks.nets")
@alias("Unet_rbf")

#####RBF for two class
class RBF(nn.Module):
    def __init__(self, input_dim, class_dim, prototype_dim):
        super(RBF, self).__init__()
        self.input_dim = input_dim
        self.class_dim = class_dim
        self.prototype_dim = prototype_dim

        self.P=Parameter(torch.Tensor(self.prototype_dim, self.input_dim))
        self.gamm=Parameter(torch.Tensor(self.prototype_dim,1))
        self.V=Parameter(torch.Tensor(self.prototype_dim,1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.P)
        nn.init.normal_(self.V)
        nn.init.constant_(self.gamm, 0.1)

    def forward(self, input):
        [batch_size, in_channel,height,weight,depth] = input.size()# input feature vector x (N*D)
        gamma=self.gamm**2
        w=torch.zeros(self.prototype_dim, batch_size,height,weight,depth,device=input.device)

        #########calculate pm, mass############
        for j in range(self.prototype_dim):

            feature = input.permute(1, 0, 2, 3, 4)
            pro = torch.mm(self.P[j, :].unsqueeze(1), torch.ones(1, batch_size,device=input.device)).unsqueeze(
                2).unsqueeze(3).unsqueeze(4)
            d = (feature - pro) ** 2
            d = d.sum(0)
            s = torch.exp(-0.5 * gamma[j] * d)
            w_j = torch.mul(self.V[j, :], s)
            w[j, :, :, :] = w_j

        o_c = torch.zeros_like(w)
        w_p_c = torch.where(w > 0, w, o_c)
        w_pc = w_p_c.sum(0)
        w_n_c = torch.where(w < 0, -w, o_c)
        w_nc = w_n_c.sum(0)

        denominator=torch.exp(-w_nc)+torch.exp(-w_pc)
        pm1=torch.exp(-w_nc)/denominator
        pm2=torch.exp(-w_pc)/denominator
        pm=torch.cat((pm1.unsqueeze(1),pm2.unsqueeze(1)),1)


        kapa_c=(1-torch.exp(-w_pc))*(1-torch.exp(-w_nc))
        kapa = 1 / (1 - kapa_c)
        mass1=kapa*(1-torch.exp(-w_pc))*torch.exp(-w_nc)
        mass2=kapa*(1-torch.exp(-w_nc))*torch.exp(-w_pc)
        mass_omega=kapa*torch.exp(-w_pc-w_nc)

        mass_all = torch.cat((mass1.unsqueeze(1),mass2.unsqueeze(1),mass_omega.unsqueeze(1)), 1)

        return pm,mass_all


class UNet_RBF(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=0,
    ) -> None:
        """
        Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
        The residual part uses a convolution to change the input dimensions to match the output dimensions
        if this is necessary but will use nn.Identity if not.
        Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

        Args:
            dimensions: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            channels: sequence of channels. Top block first.
            strides: convolution stride.
            kernel_size: convolution kernel size. Defaults to 3.
            up_kernel_size: upsampling convolution kernel size. Defaults to 3.
            num_res_units: number of residual units. Defaults to 0.
            act: activation type and arguments. Defaults to PReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()

        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout


        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Sequential:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: Union[nn.Sequential, ResidualUnit, Convolution]# nn.Sequential=residual?

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return nn.Sequential(down, SkipConnection(subblock), up)
            #return nn.Sequential(down, SkipConnection(subblock))

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)
        for p in self.parameters():
            p.requires_grad=False
        self.rbf = RBF(2,2,10)
    def _get_down_layer(
        self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> Union[ResidualUnit, Convolution]:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )
        else:
            return Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> Union[ResidualUnit, Convolution]:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(
        self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> Union[Convolution, nn.Sequential]:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                #last_conv_only=False#is_top,
                last_conv_only=is_top,
            )
            conv = nn.Sequential(conv, ru)

        return conv


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        pm,mass=self.rbf(x)
        return pm,mass

Unet_rbf = unet_rbf = UNet_RBF


