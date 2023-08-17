
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import torch.nn.functional as F


@HEADS.register_module()
class SimpleHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, is_dw=False, **kwargs):
        super(SimpleHead, self).__init__(input_transform='multiple_select', **kwargs)

        embedding_dim = self.channels

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            groups=embedding_dim if is_dw else 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
    
    def agg_res(self, preds):
        outs = preds[0]
        #outs = F.interpolate(preds[0], (64,64), None, 'bilinear', False)#新加的
        for pred in preds[1:]:
            pred = resize(pred, size=outs.size()[2:], mode='bilinear', align_corners=False) #原始代码size=outs.size()[2:],改动为size=[64,64]
            #pred = resize(pred, size=[36,36], mode='bilinear', align_corners=False)
            outs += pred
        return outs

    def forward(self, inputs):
        #xx = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        x = self.agg_res(inputs)
        _c = self.linear_fuse(x)
        x = self.cls_seg(_c)
        # print("self.cls_seg x.shape:{}".format(x.shape))
        return x