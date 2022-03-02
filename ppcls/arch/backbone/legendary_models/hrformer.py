# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn, Tensor
from paddle import ParamAttr
import paddle.nn.functional as F
from paddle.nn.functional import upsample
from paddle.nn.initializer import Uniform

from ppcls.arch.backbone.base.theseus_layer import TheseusLayer, Identity
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

from .hrt_utils import Bottleneck, _make_layer, GeneralTransformerBlock, BottleneckDWP

MODEL_URLS = {
    "HRFormer_tiny": None,
    "HRFormer_small": None,
    "HRFormer_base": None,
}

blocks_dict = {
    "BOTTLENECK": Bottleneck,
    "TRANSFORMER_BLOCK": GeneralTransformerBlock,
    "BOTTLENECKDWP": BottleneckDWP,
}

__all__ = list(MODEL_URLS.keys())
BN_MOMENTUM = 0.1


class HRTransformer(TheseusLayer):
    '''
    '''

    def __init__(self,
                 in_chans=3,
                 class_num=1000,
                 num_blocks=[2, 2, 2, 2],
                 num_modules=[1, 1, 3, 2],
                 num_channels1=64,
                 num_channels=[18, 36, 72, 144],
                 num_heads=[1, 2, 4, 8],
                 num_branches=[1, 2, 3, 4],
                 num_window_sizes=[7, 7, 7, 7],
                 num_mlp_ratios=[4, 4, 4, 4],
                 drop_path_rate=0,
                 blocks=['BOTTLENECK', 'TRANSFORMER_BLOCK', 'TRANSFORMER_BLOCK', 'TRANSFORMER_BLOCK'],
                 resolutions=[56, 28, 14, 7],
                 attn_types=['isa_local', [2, 2, 1], [2, 3, 3], [2, 4, 2]],
                 ffn_types=['conv_mlp', [2, 2, 1], [2, 3, 3], [2, 4, 2]],
                 final_channels=2048,
                 head_block='BOTTLENECKDWP',
                 head_channels=[32, 64, 128, 256],
                 ):
        super().__init__()
        # parameters
        depth2 = num_blocks[1] * num_modules[1]
        depth3 = num_blocks[2] * num_modules[2]
        depth4 = num_blocks[3] * num_modules[3]
        depths = [depth2, depth3, depth4]
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))]
        num_inchannels2 = [num_channels[i] * blocks_dict[blocks[1]].expansion for i in range(len(num_channels[:2]))]
        num_inchannels3 = [num_channels[i] * blocks_dict[blocks[2]].expansion for i in range(len(num_channels[:3]))]
        num_inchannels4 = [num_channels[i] * blocks_dict[blocks[3]].expansion for i in range(len(num_channels[:4]))]
        num_outchannels1 = [blocks_dict[blocks[0]].expansion * num_channels1]

        # archeticture
        self.down_stem = Conv_Stem(in_chans=in_chans)
        self.layer1 = _make_layer(block=blocks_dict[blocks[0]],
                                  inplanes=64,
                                  planes=num_channels1,
                                  blocks=num_blocks[0])
        self.transition1 = Transition(num_branches=num_branches[1],
                                      num_channels_pre_layer=num_outchannels1,
                                      num_channels_cur_layer=num_channels[:2])

        self.stage2 = Stage(block=blocks_dict[blocks[1]],
                            num_modules=num_modules[1],
                            num_branches=num_branches[1],
                            num_blocks=num_blocks[:2],
                            num_channels=num_channels[:2],
                            num_inchannels=num_inchannels2,
                            num_heads=num_heads[:2],
                            num_window_sizes=num_window_sizes[:2],
                            num_mlp_ratios=num_mlp_ratios[:2],
                            num_input_resolutions=resolutions[:2],
                            drop_paths=dpr[0:depth2],
                            attn_types=[attn_types[0], attn_types[1]],
                            ffn_types=[ffn_types[0], ffn_types[1]],
                            )
        self.transition2 = Transition(num_branches=num_branches[2],
                                      num_channels_pre_layer=self.stage2.num_inchannels,
                                      num_channels_cur_layer=num_channels[:3])

        self.stage3 = Stage(block=blocks_dict[blocks[2]],
                            num_modules=num_modules[2],
                            num_branches=num_branches[2],
                            num_blocks=num_blocks[:3],
                            num_channels=num_channels[:3],
                            num_inchannels=num_inchannels3,
                            num_heads=num_heads[:3],
                            num_window_sizes=num_window_sizes[:3],
                            num_mlp_ratios=num_mlp_ratios[:3],
                            num_input_resolutions=resolutions[:3],
                            drop_paths=dpr[depth2: depth2 + depth3],
                            attn_types=[attn_types[0], attn_types[2]],
                            ffn_types=[ffn_types[0], ffn_types[2]],
                            )
        self.transition3 = Transition(num_branches=num_branches[3],
                                      num_channels_pre_layer=self.stage3.num_inchannels,
                                      num_channels_cur_layer=num_channels[:4])

        self.stage4 = Stage(block=blocks_dict[blocks[3]],
                            num_modules=num_modules[3],
                            num_branches=num_branches[3],
                            num_blocks=num_blocks[:4],
                            num_channels=num_channels[:4],
                            num_inchannels=num_inchannels4,
                            num_heads=num_heads[:4],
                            num_window_sizes=num_window_sizes[:4],
                            num_mlp_ratios=num_mlp_ratios[:4],
                            num_input_resolutions=resolutions[:4],
                            drop_paths=dpr[depth2 + depth3:],
                            attn_types=[attn_types[0], attn_types[3]],
                            ffn_types=[ffn_types[0], ffn_types[3]],
                            )

        self.class_head = Class_Head(class_num,
                                     self.stage4.num_inchannels,
                                     inter_channels=final_channels,
                                     head_block=blocks_dict[head_block],
                                     head_channels=head_channels)

    def forward(self, x):
        x = self.down_stem(x)

        s1 = self.layer1(x)
        t1 = self.transition1(s1)

        s2 = self.stage2(t1)
        t2 = self.transition2(s2)

        s3 = self.stage3(t2)
        t3 = self.transition3(s3)

        s4 = self.stage4(t3)

        y = self.class_head(s4)
        return y


class Conv_Stem(TheseusLayer):
    '''
    conv stem for HRFormer
    '''

    def __init__(self, in_chans=3):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2D(64, 64, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class Transition(TheseusLayer):
    '''

    '''

    def __init__(self, num_branches, num_channels_pre_layer, num_channels_cur_layer):
        super().__init__()
        self.num_branches = num_branches
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2D(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias_attr=False,
                            ),
                            nn.BatchNorm2D(
                                num_channels_cur_layer[i], momentum=BN_MOMENTUM
                            ),
                            nn.ReLU(),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2D(inchannels, outchannels, 3, 2, 1, bias_attr=False),
                            nn.BatchNorm2D(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))
        self.transition = nn.LayerList(transition_layers)

    def forward(self, x):
        x_list = []
        for i in range(self.num_branches):
            if self.transition[i] is not None:
                if isinstance(x, list):
                    x_input = x[-1]
                else:
                    x_input = x
                x_list.append(self.transition[i](x_input))
            else:
                if isinstance(x, list):
                    x_input = x[i]
                else:
                    x_input = x
                x_list.append(x_input)
        return x_list


class Stage(TheseusLayer):
    def __init__(self,
                 block,
                 num_modules,
                 num_branches,
                 num_blocks,
                 num_channels,
                 num_inchannels,
                 num_heads,
                 num_window_sizes,
                 num_mlp_ratios,
                 num_input_resolutions,
                 drop_paths,
                 multi_scale_output=True,
                 attn_types=None,
                 ffn_types=None,
                 ):
        super().__init__()
        num_input_resolutions = [[res, res] for res in num_input_resolutions]
        if isinstance(attn_types, list) and attn_types[1] is not None:
            attn_types = [[[attn_types[0]] * attn_types[1][0]] * attn_types[1][1]] * attn_types[1][2]
        else:
            attn_types = None
        if isinstance(ffn_types, list) and ffn_types[1] is not None:
            ffn_types = [[[ffn_types[0]] * ffn_types[1][0]] * ffn_types[1][1]] * ffn_types[1][2]
        else:
            ffn_types = None

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionTransformerModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    num_input_resolutions,
                    attn_types[i],
                    ffn_types[i],
                    reset_multi_scale_output,
                    drop_paths=drop_paths,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        self.stage = nn.Sequential(*modules)
        self.num_inchannels = num_inchannels

    def forward(self, x):
        x = self.stage(x)
        return x


class HighResolutionTransformerModule(nn.Layer):
    def __init__(
            self,
            num_branches,
            blocks,
            num_blocks,
            num_inchannels,
            num_channels,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            num_input_resolutions,
            attn_types,
            ffn_types,
            multi_scale_output=True,
            drop_paths=0.0,
    ):
        """
        Args:
            num_heads: the number of head witin each MHSA
            num_window_sizes: the window size for the local self-attention
            num_input_resolutions: the spatial height/width of the input feature maps.
        """
        super(HighResolutionTransformerModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_input_resolutions,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            attn_types,
            ffn_types,
            drop_paths,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()

        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios
        self.num_input_resolutions = num_input_resolutions
        self.attn_types = attn_types
        self.ffn_types = ffn_types

    def _check_branches(
            self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            print(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(
            self,
            branch_index,
            block,
            num_blocks,
            num_channels,
            num_input_resolutions,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            attn_types,
            ffn_types,
            drop_paths,
            stride=1,
    ):
        downsample = None
        if (
                stride != 1
                or self.num_inchannels[branch_index]
                != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(
                    num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                input_resolution=num_input_resolutions[branch_index],
                num_heads=num_heads[branch_index],
                window_size=num_window_sizes[branch_index],
                mlp_ratio=num_mlp_ratios[branch_index],
                attn_type=attn_types[branch_index][0],
                ffn_type=ffn_types[branch_index][0],
                drop_path=drop_paths[0],
            )
        )

        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    input_resolution=num_input_resolutions[branch_index],
                    num_heads=num_heads[branch_index],
                    window_size=num_window_sizes[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    attn_type=attn_types[branch_index][i],
                    ffn_type=ffn_types[branch_index][i],
                    drop_path=drop_paths[i],
                )
            )
        return nn.Sequential(*layers)

    def _make_branches(
            self,
            num_branches,
            block,
            num_blocks,
            num_channels,
            num_input_resolutions,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            attn_types,
            ffn_types,
            drop_paths,
    ):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    num_input_resolutions,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    attn_types,
                    ffn_types,
                    drop_paths,
                )
            )

        return nn.LayerList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2D(
                                num_inchannels[j],
                                num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                bias_attr=False,
                            ),
                            nn.BatchNorm2D(num_inchannels[i], momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2D(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                        bias_attr=False,
                                    ),
                                    nn.BatchNorm2D(
                                        num_inchannels[j], momentum=BN_MOMENTUM
                                    ),
                                    nn.Conv2D(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                        bias_attr=False,
                                    ),
                                    nn.BatchNorm2D(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2D(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                        bias_attr=False,
                                    ),
                                    nn.BatchNorm2D(
                                        num_inchannels[j], momentum=BN_MOMENTUM
                                    ),
                                    nn.Conv2D(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                        bias_attr=False,
                                    ),
                                    nn.BatchNorm2D(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                    nn.ReLU(False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.LayerList(fuse_layer))

        return nn.LayerList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class Class_Head(TheseusLayer):
    '''
    class head for HRFormer
    '''

    def __init__(self,
                 class_num,
                 pre_stage_channels,
                 inter_channels=2048,
                 head_block=BottleneckDWP,
                 head_channels=[32, 64, 128, 256]):
        super().__init__()
        self.inter_channels = inter_channels
        self.class_num = class_num

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = _make_layer(
                head_block, channels, head_channels[i], 1, stride=1
            )
            incre_modules.append(incre_module)
        self.incre_modules = nn.LayerList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(
                nn.Conv2D(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                ),
                nn.BatchNorm2D(in_channels, momentum=BN_MOMENTUM),
                nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2D(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(),
            )
            downsamp_modules.append(downsamp_module)
        self.downsamp_modules = nn.LayerList(downsamp_modules)

        self.final_layer = nn.Sequential(
            nn.Conv2D(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2D(self.inter_channels, momentum=BN_MOMENTUM),
            nn.ReLU(),
        )
        # incre_modules, downsamp_modules, final_layer
        self.classifier = nn.Linear(self.inter_channels, self.class_num)

    def forward(self, x):
        y = self.incre_modules[0](x[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](x[i + 1]) + self.downsamp_modules[i](y)

        y = self.final_layer(y)
        y = F.avg_pool2d(y, kernel_size=y.shape[2:]).reshape((y.shape[0], -1))
        y = self.classifier(y)
        return y


def _load_pretrained(pretrained, model, model_url, use_ssld):
    pretrained = False
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )

def load_from_pdparams(model, dict_path):
    model_state_dict = model.state_dict()
    para_state_dict = paddle.load(dict_path)
    mkeys = model_state_dict.keys()
    num_params_loaded = 0
    num_params_skip = 0


    # print(list(model_state_dict.keys())[:5])
    # print(list(para_state_dict.keys())[:5])

    # region base model load&check
    for k in mkeys:

        if k.startswith('down_stem.'):
            kt = k[len('down_stem.'):]
        elif k.startswith('class_head.'):
            kt = k[len('class_head.'):]
        elif k.startswith('stage2.stage.'):
            kt = k.replace('stage2.stage.', 'stage2.')
        elif k.startswith('stage3.stage.'):
            kt = k.replace('stage3.stage.', 'stage3.')
        elif k.startswith('stage4.stage.'):
            kt = k.replace('stage4.stage.', 'stage4.')
        elif k.startswith('transition1.transition.'):
            kt = k.replace('transition1.transition.', 'transition1.')
        elif k.startswith('transition2.transition.'):
            kt = k.replace('transition2.transition.', 'transition2.')
        elif k.startswith('transition3.transition.'):
            kt = k.replace('transition3.transition.', 'transition3.')
        else:
            kt = k

        if kt.endswith('._mean'):
            kt = kt.replace('._mean', '.running_mean')
        elif kt.endswith('._variance'):
            kt = kt.replace('._variance', '.running_var')
        else:
            kt = kt

        if k in para_state_dict:
            if list(para_state_dict[k].shape) != list(model_state_dict[k].shape):
                print("[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                               .format(k, para_state_dict[k].shape, model_state_dict[k].shape))
                num_params_skip += 1
            else:
                model_state_dict[k] = para_state_dict[k]
                num_params_loaded += 1
        elif kt in para_state_dict:
            if list(para_state_dict[kt].shape) != list(model_state_dict[k].shape):
                print("[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                               .format(kt, para_state_dict[kt].shape, model_state_dict[k].shape))
                num_params_skip += 1
            else:
                model_state_dict[k] = para_state_dict[kt]
                num_params_loaded += 1
        else:
            print("[Drop] Pretrained params {}{} doesn't exist!"
                           .format(k, model_state_dict[k].shape))
    # endregion

    print('total: loaded %d, model contain %d, file contain %d, skip %d' %
          (num_params_loaded, len(model_state_dict), len(para_state_dict), num_params_skip))
    return model


def HRFormer_tiny(pretrained=False, use_ssld=False, **kwargs):
    """
    """
    model = HRTransformer(num_modules=[1, 1, 3, 2],
                          num_channels=[18, 36, 72, 144],
                          num_heads=[1, 2, 4, 8],
                          drop_path_rate=0.1,
                          attn_types=['isa_local', [2, 2, 1], [2, 3, 3], [2, 4, 2]],
                          ffn_types=['conv_mlp', [2, 2, 1], [2, 3, 3], [2, 4, 2]],
                          **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["HRFormer_tiny"], use_ssld)
    return model


def HRFormer_small(pretrained=False, use_ssld=False, **kwargs):
    """
    """
    model = HRTransformer(num_modules=[1, 1, 4, 2],
                          num_channels=[32, 64, 128, 256],
                          num_heads=[1, 2, 4, 8],
                          drop_path_rate=0.1,
                          attn_types=['isa_local', [2, 2, 1], [2, 3, 4], [2, 4, 2]],
                          ffn_types=['conv_mlp', [2, 2, 1], [2, 3, 4], [2, 4, 2]],
                          **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["HRFormer_small"], use_ssld)
    return model


def HRFormer_base(pretrained=False, use_ssld=False, **kwargs):
    """
    """
    model = HRTransformer(num_modules=[1, 1, 4, 2],
                          num_channels=[78, 156, 312, 624],
                          num_heads=[2, 4, 8, 16],
                          drop_path_rate=0.2,
                          attn_types=['isa_local', [2, 2, 1], [2, 3, 4], [2, 4, 2]],
                          ffn_types=['conv_mlp', [2, 2, 1], [2, 3, 4], [2, 4, 2]],
                          **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["HRFormer_base"], use_ssld)
    return model
