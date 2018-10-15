"""
Visual Genome in Scene Graph Generation by Iterative Message Passing split
"""
import os
import cv2
import json
import h5py
import pickle
import numpy as np
import scipy.sparse
import os.path as osp

#from .vg_eval import vg_eval
#import datasets.ds_utils as ds_utils
from datasets.imdb import imdb
from model.utils.config import cfg

from IPython import embed


class vg_sggimp(imdb):
    """
        Visual Genome with sgg split
    """
    def __init__(self, split):
        """
        Args:
            split: integer, 0 is training, 1 is val, 2 is test
        """
        imdb.__init__(self, 'vg_sggimp')
        # TODO: add file existence asserts
        # load files
        self.data_dir = osp.join(cfg.DATA_DIR, 'visual_genome')
        self.cache_dir = osp.join(cfg.DATA_DIR, 'cache')
        self.anno_dir = osp.join(self.data_dir, 'sggimp')
        self.img_dir = osp.join(self.data_dir, 'VG_100K')

        with open(osp.join(self.anno_dir, 'VG-SGG-dicts.json'), 'r') as f:
            self.vg_dicts = json.load(f)
        with open(osp.join(self.anno_dir, 'image_data.json'), 'r') as f:
            self.img_meta = json.load(f)
        self.vg_h5 = h5py.File(osp.join(self.anno_dir, 'VG-SGG.h5'))

        # filter boxes
        self.filter_corrupted_imgs()
        # there are 108073 images now
        self.load_statistic()

        # shape : (NumOfImages, )
        self.img_to_first_box = self.vg_h5['img_to_first_box'].value
        self.img_to_last_box = self.vg_h5['img_to_first_box'].value
        self.img_to_first_rel = self.vg_h5['img_to_first_rel'].value
        self.img_to_last_rel = self.vg_h5['img_to_last_rel'].value
        # shape: (NumOfBoxes, 4)
        self.bboxes = self.vg_h5['boxes_%s' % cfg.BOX_SCALE_H5].value

        # covert from xcenter, ycenter, w, h to x0, y0, x1, y1
        self.bboxes[:, :2] = self.bboxes[:, :2] - np.floor(self.bboxes[:, 2:] / 2)
        self.bboxes[:, 2:] += self.bboxes[:, :2] - 1

        # shape: (NumOfBoxes, )
        self.bbox_labels = self.vg_h5['labels'].value
        # predicates, shape: (NumOfRelationships, )
        self.predicates = self.vg_h5['predicates'].value
        # box relationships, shape: (NumOfRelationships, 2)
        # specify the ids of two boxes related to the relationship 
        # e.g. bbox_rels[0] is [boxid1, boxid2]
        self.bbox_rels = self.vg_h5['relationships'].value
        # split, shape (NumOfImages, )
        self.split_indicator = self.vg_h5['split'].value

        self.split_data(split)
        self.filter_invalid_box()

        self.nr_img = len(self.img_meta)
        # set imdb shit
        self._image_index
        self._roidb_handler = self.gt_roidb
        self._image_ext = '.jpg'
        self._classes = 
        self

    def load_statistic(self):
        self.idx_to_labels = dict(map(lambda x: (int(x[0]), x[1]), self.vg_dicts['idx_to_label'].items()))
        self.idx_to_labels.update({0: 'background'})
        self.idx_to_predicates = dict(map(lambda x: (int(x[0]), x[1]),self.vg_dicts['idx_to_predicate'].items()))
        self.idx_to_predicates.update({0: '__irrelevant__'})
        self.nr_classes = len(self.idx_to_labels)
        self.nr_predicates = len(self.idx_to_predicates)

    def filter_corrupted_imgs(self):
        """
        Args:
        """
# TODO: put magic numbers in config.py
        # corrupted files: 1592.jpg 1722.jpg 4616.jpg 4617.jpg
        del self.img_meta[1591], self.img_meta[1720]
        del self.img_meta[4613], self.img_meta[4613]

        self.imgs_path = []
        for meta in self.img_meta:
            fn = '{}.jpg'.format(meta['image_id'])
            img_path = osp.join(self.img_dir, fn)
            if osp.exists(img_path):
                self.imgs_path.append(img_path)
        self.imgs_path = np.array(self.imgs_path)
        self.img_meta = np.array(self.img_meta)
        assert len(self.imgs_path) == 108073

    def split_data(self, split):
        """
        Args:
            split: integer, 0,1,2, train val test
        """
        split_mask = self.split_indicator == split
        self._filter(split_mask)

    def filter_invalid_box(self):
        """
            delelte those image without boxes
        """
        valid_mask = self.img_to_first_box >=0 
        assert np.all(
                valid_mask == (self.img_to_last_box >= 0)
                )
        self._filter(valid_mask)

    def _filter(self, mask):
        """
        Args:
            mask: numpy array of boolean
        """
        self.img_to_first_box = self.img_to_first_box[mask]
        self.img_to_last_box = self.img_to_last_box[mask]
        self.img_to_first_rel = self.img_to_first_rel[mask]
        self.img_to_last_rel = self.img_to_last_rel[mask]
        self.imgs_path = self.imgs_path[mask]
        self.img_meta = self.img_meta[mask]

    def gt_roidb(self):
        cache_path = osp.join(self.cache_dir, '%s_roidb.pkl' % self._name)
        if osp.exists(cache_path):
            with open(cache_path, 'rb') as f:
                roidb = pickle.load(f)
            return roidb
        roidb = [self._load_vg_anno(i) for i in range(self.nr_img)]
        with open(cache_path, 'wb') as f:
            pickle.dump(roidb, f)
        return roidb



    def get_size_after_resizing(self, height, width, scale):
        if height > width:
            return int(scale), int(width / height * scale)
        else:
            return int(height / width * scale), int(scale)

    def _load_vg_anno(self, idx):
        """
        load visual genome annotations of image with index `idx`
        you should know the difference between image index and image id
        image id is in annotation file, image index is the index of img_meta
        Args:
            idx: integer, index of image
        """
        idx_roidb = {}
        # image annotations
        oh, ow = self.img_meta[idx]['height'], self.img_meta[idx]['width']
        height, width = self.get_size_after_resizing(oh, ow, cfg.BOX_SCALE_H5)

        # bounding boxes annotations
        bboxes = self.bboxes[self.img_to_first_box[idx]: self.img_to_last_box[idx] + 1, :]
        bbox_labels = self.bbox_labels[self.img_to_first_box[idx]: self.img_to_last_box[idx] + 1]
        overlaps = np.zeros((bboxes.shape[0], self.nr_classes))
        for ci, o in enumerate(overlaps):
            o[bbox_labels[ci]] = 1.
        overlaps = scipy.sparse.csr_matrix(overlaps)
        seg_areas = np.multiply(bboxes[:, 2] - bboxes[:, 0] + 1, 
                bboxes[:, 3] - bboxes[:, 1] + 1)

        # relation annotations
        rels = []
        first_rel_idx = self.img_to_first_rel[idx]
        last_rel_idx = self.img_to_last_rel[idx]
        if first_rel_idx >= 0:
            assert last_rel_idx >= 0
            predicates = self.predicates[first_rel_idx: last_rel_idx + 1]
            bbox_rels = self.bbox_rels[first_rel_idx: last_rel_idx + 1]
            # img_to_first_box validness has been checked
            bbox_rels -= self.img_to_first_box[idx]
            assert bbox_rels.shape[0] == predicates.shape[0]
            for ri, predicate in enumerate(predicates):
                rels.append([bbox_rels[ri][0], predicate, bbox_rels[ri][1]])
        rels = np.array(rels)
        idx_roidb.update(
                {
                    'boxes': bboxes,
                    'gt_classes': bbox_labels,
                    'gt_rels': rels,
                    'gt_overlaps': overlaps,
                    'seg_areas': seg_areas,
                    'flipped': False,
                    'width': width,
                    'height': height
                    }
                )

        return idx_roidb


if __name__ == '__main__':
    fuck = vg_sggimp(0)
    embed(header='myvg.py in lib/datasets')
