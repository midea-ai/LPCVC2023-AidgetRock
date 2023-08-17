import torch
import mmcv
import sys
from mmrazor.models.builder import build_algorithm
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmseg.models import build_segmentor


def split_student_model(cls_cfg_path, cls_model_path, device='cuda', save_path=None):
    """
    :param: cls_cfg_path: your normal classifier config file path which is not disitilation cfg path
    :param: cls_model_path: your distilation checkpoint path
    :param: save_path: student model save path
    """
    cfg = mmcv.Config.fromfile(cls_cfg_path)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    model = build_segmentor(cfg.model)
    model_ckpt = torch.load(cls_model_path, map_location='cuda')
    pretrained_dict = model_ckpt['state_dict']
    model_dict = model.state_dict()
    new_dict = {k.replace('architecture.model.', ''): v for k, v in pretrained_dict.items() if k.replace('architecture.model.', '') in model_dict.keys()}
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    torch.save({'state_dict': model.state_dict(), 'meta': model_ckpt['meta'],
                'optimizer': model_ckpt['optimizer']}, save_path)


split_student_model(cls_cfg_path='local_configs/topformer/topformer_tiny_288x288_160k_2x8_ade20k.py',cls_model_path='distill_8-5/512b_288tt_tau3/iter_22500.pth', save_path='tiny_288.pth')