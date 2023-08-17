
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class LPCVCDataset(CustomDataset):
    CLASSES = (
        'background', 'avalanche', 'building_undamaged', 'building_damaged', 'cracks/fissure/subsidence', 'debris/mud//rock flow',
        'fire/flare', 'flood/water/river/sea ', 'ice_jam_flow', 'lava_flow', 'person', 'pyroclastic_flow','road/railway/bridge','vehicle')

    # PALETTE = [[10,10,10], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    #            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
    #            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
    #            [192, 0, 128]]
    
    PALETTE = [[0,0,0], [1, 1, 1], [2, 2,2], [3, 3, 3], [4, 4, 4],
           [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8],
           [9, 9,9], [10, 10, 10], [11, 11, 11], [12, 12, 12],
           [13, 13, 13]]

    def __init__(self, **kwargs):
        super(LPCVCDataset, self).__init__(
            # img_suffix='.jpg',
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files
