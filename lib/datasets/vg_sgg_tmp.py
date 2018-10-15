
from __future__ import print_function
from __future__ import absolute_import

import os
import PIL
import pdb
import gzip
import json
import h5py
import pickle
import numpy as np
import scipy.sparse
import os.path as osp
import xml.etree.ElementTree as ET

from .vg_eval import vg_eval
import datasets.ds_utils as ds_utils
from datasets.imdb import imdb
from model.utils.config import cfg

from IPython import embed


class vg_sgg(imdb):
    """
    imdb of Visual Genome by SGG split
    """
    def __init__(self, version):
        imdb.__init__(self, 'vg_%s' % version)
        self._version = version
        self._data_dir = osp.join(cfg.DATA_DIR, 'visual_genome')
        self._img_dir = osp.join(self._data_dir, 'VG_100K')
        self._anno_dir = osp.join(self._data_dir, 'sggimp')  # Scene Graph Generation by Iterative Message Passing
        self._cache_dir = osp.join(cfg.DATA_DIR, 'cache')
        self._img_ext = '.jpg'
        self._check_sggfile_valid()

        with open(osp.join(self._anno_dir, 'VG-SGG-dicts.json'), 'r') as f:
            vg_sgg_dicts = json.load(f)
            self._object_count = vg_sgg_dicts.get('object_count')
            self._predicate_count = vg_sgg_dicts.get('predicate_count')
            self._idx_to_label = vg_sgg_dicts.get('idx_to_label')
            self._idx_to_predicate = vg_sgg_dicts.get('idx_to_predicate')
            self._label_to_idx = vg_sgg_dicts.get('label_to_idx')
            self._predicate_to_idx = vg_sgg_dicts.get('predicate_to_idx')

        self._idx_to_label = dict(
            map(
                lambda x: (int(x[0]), x[1]), self._idx_to_label.items()
            )
        ).update({0: '__background__'})
        self._label_to_idx.update({'__background__': 0})

        self._idx_to_predicate = dict(
            map(
                lambda x: (int(x[0]), x[1]), self._idx_to_predicate.items()
            )
        ).update({0: '__no_relationship__'})
        self._predicate_to_idx.update({'__no_relationship__': 0})

        # VG-SGG.h5 is only 57M, it can be loaded into memory directly
        with h5py.File(osp.join(self._anno_dir, 'VG-SGG.h5'), 'r') as f:
            # box number to box label index, with shape of (NumOfBoxes,)
            self.labels = f['labels']
            # relation number to predicate index, with shape of (NumOfRelationships,)
            self.predicates = f['predicates']
            # boxes in shape of 1024, with shape of (NumOfBoxes, 4)
            self.boxes_1024 = f['boxes_1024']
            # boxes in shape of 512, with shape of (NumOfBoxes, 4)
            self.boxes_512 = f['boxes_512']
            #  relation number to two box number, with shape of (NumOfRelationships, 2)
            self.relationships = f['relationships']
            # image number to box number, with shape of (NumOfImgs,)
            self.img_to_first_box = f['img_to_first_box']
            # image number to box number, with shape of (NumOfImgs,)
            self.img_to_last_box = f['img_to_last_box']
            # image number to relationship number, with shape of (NumOfImgs,)
            self.img_to_first_rel = f['img_to_first_rel']
            # image number to relationship number, with shape of (NumOfImgs,)
            self.img_to_last_rel = f['img_to_last_rel']
            # image splits indicator, 0-train, 1-val, 2-test, with shape of (NumOfImgs,)
            self.split = f['split']

        self.imgs_path, self.imgs_meta = self.load_imgs_path_and_meta()

    def _check_sggfile_valid(self):
        """
        Check completeness of files from SGGIMP
        """
        assert osp.exists(osp.join(self._anno_path, 'VG-SGG.h5'))
        assert osp.exists(osp.join(self._anno_path, 'VG-SGG-dicts.json'))
        assert osp.exists(osp.join(self._anno_path, 'image_data.json'))

    def load_imgs_path_and_meta(self):
        """
        Loads the image filenames from visual genome from the JSON file that contains them.
        This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
        """
        with open(osp.join(self._anno_path, 'image_data.json'), 'r') as f:
            img_meta = json.load(f)
        # corrupted images are `1592.jpg`, `1722.jpg`, `4616.jpg`, `4617.jpg`
        # notice: the index decrease 1 when del happens
        del img_meta[1591], img_meta[1720], img_meta[4613], img_meta[4613]
        paths = []
        for meta in img_meta:
            fn = '{}.jpg'.format(meta['image_id'])
            path = os.path.join(self._img_dir, fn)
            if os.path.exists(path):
                paths.append(path)
        assert len(fns) == 108073
        return paths, img_met

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a pickle file to speed up future calls.
        """
        cache_file = os.path.join(self._cache_path, '%s_gt_roidb.pkl' % self._name)
        if os.path.exists(cache_file):
            with gzip.open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self._name, cache_file))
            return roidb
        gt_roidb = [self._vg_sgg_anno(index) for index in range(self._nr_img)]
        with gzip.open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _get_size(self, index):
        """
        get size of image with index
        Args:
           index: integer, index of image
        Returns:
           height: integer
           width: integer
        """
        height = self._image_data[index]['height']
        width = self._image_data[index]['width']
        return height, width

    def _annotation_path(self, index):
        # todo
        return os.path.join(self._data_path, 'xml', str(index) + '.xml')

    def _vg_sgg_anno(self, index):
        """
        load Visual Genome annotations
        """
        height, width = self._get_size(index)




        filename = self._annotation_path(index)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        # Max of 16 attributes are observed in the data
        gt_attributes = np.zeros((num_objs, 16), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        obj_dict = {}
        ix = 0
        for obj in objs:
            obj_name = obj.find('name').text.lower().strip()
            if obj_name in self._class_to_ind:
                bbox = obj.find('bndbox')
                x1 = max(0,float(bbox.find('xmin').text))
                y1 = max(0,float(bbox.find('ymin').text))
                x2 = min(width-1,float(bbox.find('xmax').text))
                y2 = min(height-1,float(bbox.find('ymax').text))
                # If bboxes are not positive, just give whole image coords (there are a few examples)
                if x2 < x1 or y2 < y1:
                    print('Failed bbox in %s, object %s' % (filename, obj_name))
                    x1 = 0
                    y1 = 0
                    x2 = width-1
                    y2 = width-1
                cls = self._class_to_ind[obj_name]
                obj_dict[obj.find('object_id').text] = ix
                atts = obj.findall('attribute')
                n = 0
                for att in atts:
                    att = att.text.lower().strip()
                    if att in self._attribute_to_ind:
                        gt_attributes[ix, n] = self._attribute_to_ind[att]
                        n += 1
                    if n >= 16:
                        break
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
                ix += 1
        # clip gt_classes and gt_relations
        gt_classes = gt_classes[:ix]
        gt_attributes = gt_attributes[:ix, :]

        overlaps = scipy.sparse.csr_matrix(overlaps)
        gt_attributes = scipy.sparse.csr_matrix(gt_attributes)

        rels = tree.findall('relation')
        num_rels = len(rels)
        gt_relations = set() # Avoid duplicates
        for rel in rels:
            pred = rel.find('predicate').text
            if pred: # One is empty
                pred = pred.lower().strip()
                if pred in self._relation_to_ind:
                    try:
                        triple = []
                        triple.append(obj_dict[rel.find('subject_id').text])
                        triple.append(self._relation_to_ind[pred])
                        triple.append(obj_dict[rel.find('object_id').text])
                        gt_relations.add(tuple(triple))
                    except:
                        pass # Object not in dictionary
        gt_relations = np.array(list(gt_relations), dtype=np.int32)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_attributes': gt_attributes,
                'gt_relations': gt_relations,
                'gt_overlaps': overlaps,
                'width': width,
                'height': height,
                'flipped': False,
                'seg_areas': seg_areas}

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(self.classes, all_boxes, output_dir)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_vg_results_file_template(output_dir).format(cls)
                os.remove(filename)

    def evaluate_attributes(self, all_boxes, output_dir):
        self._write_voc_results_file(self.attributes, all_boxes, output_dir)
        self._do_python_eval(output_dir, eval_attributes = True)
        if self.config['cleanup']:
            for cls in self._attributes:
                if cls == '__no_attribute__':
                    continue
                filename = self._get_vg_results_file_template(output_dir).format(cls)
                os.remove(filename)

    def _get_vg_results_file_template(self, output_dir):
        filename = 'detections_' + self._image_set + '_{:s}.txt'
        path = os.path.join(output_dir, filename)
        return path

    def _write_voc_results_file(self, classes, all_boxes, output_dir):
        for cls_ind, cls in enumerate(classes):
            if cls == '__background__':
                continue
            print('Writing "{}" vg results file'.format(cls))
            filename = self._get_vg_results_file_template(output_dir).format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(index), dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))


    def _do_python_eval(self, output_dir, pickle=True, eval_attributes = False):
        # We re-use parts of the pascal voc python code for visual genome
        aps = []
        nposs = []
        thresh = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # Load ground truth
        gt_roidb = self.gt_roidb()
        if eval_attributes:
            classes = self._attributes
        else:
            classes = self._classes
        for i, cls in enumerate(classes):
            if cls == '__background__' or cls == '__no_attribute__':
                continue
            filename = self._get_vg_results_file_template(output_dir).format(cls)
            rec, prec, ap, scores, npos = vg_eval(
                filename, gt_roidb, self.image_index, i, ovthresh=0.5,
                use_07_metric=use_07_metric, eval_attributes=eval_attributes)

            # Determine per class detection thresholds that maximise f score
            if npos > 1:
                f = np.nan_to_num((prec*rec)/(prec+rec))
                thresh += [scores[np.argmax(f)]]
            else:
                thresh += [0]
            aps += [ap]
            nposs += [float(npos)]
            print('AP for {} = {:.4f} (npos={:,})'.format(cls, ap, npos))
            if pickle:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap,
                        'scores': scores, 'npos':npos}, f)

        # Set thresh to mean for classes with poor results
        thresh = np.array(thresh)
        avg_thresh = np.mean(thresh[thresh!=0])
        thresh[thresh==0] = avg_thresh
        if eval_attributes:
            filename = 'attribute_thresholds_' + self._image_set + '.txt'
        else:
            filename = 'object_thresholds_' + self._image_set + '.txt'
        path = os.path.join(output_dir, filename)
        with open(path, 'wt') as f:
            for i, cls in enumerate(classes[1:]):
                f.write('{:s} {:.3f}\n'.format(cls, thresh[i]))

        weights = np.array(nposs)
        weights /= weights.sum()
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('Weighted Mean AP = {:.4f}'.format(np.average(aps, weights=weights)))
        print('Mean Detection Threshold = {:.3f}'.format(avg_thresh))
        print('~~~~~~~~')
        print('Results:')
        for ap,npos in zip(aps,nposs):
            print('{:.3f}\t{:.3f}'.format(ap,npos))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** PASCAL VOC Python eval code.')
        print('--------------------------------------------------------------')


if __name__ == '__main__':
    d = vg('val')
    res = d.roidb
    from IPython import embed; embed()
