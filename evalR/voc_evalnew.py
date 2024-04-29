import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np
from rich.progress import track

from utils.utils_basic import *


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def parse_poly(filename):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                classes = cfg.DATA["CLASSES"]
                object_struct['name'] = classes[int(splitlines[0])]
                # object_struct['name'] = splitlines[0]
                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7]),
                                         float(splitlines[8])]
                objects.append(object_struct)
            else:
                break
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


''''''


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    # first load gt
    # print(cachedir)
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # print(cachefile)
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            ####################parse_poly
            recs[imagename] = parse_poly(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
        print('Save done')
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0


    for imagename in track(imagenames, description='Cal AP =>' + classname):
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool_)
        # det = [False] * len(R)
        # TODO
        # Bug 浅拷贝
        # det = [[False] * len(R)] * 10
        det = [[0 for i in range(len(R))] for j in range(10)]

        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets#######################
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # TODO
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    # tp = np.zeros(nd)
    # fp = np.zeros(nd)
    ovthreshs = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    # NOTE 10, 0.5:0.95
    tp = np.zeros((nd, 10))
    fp = np.zeros((nd, 10))
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            ###############################

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    overlap = polygen_iou_xy4_numpy_eval(BBGT_keep[index], bb)
                    # overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
                #############################

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        # if ovmax > 0.55:
        #     # print(ovmax)
        #     if not R['difficult'][jmax]:
        #         if notkj['det'][jmax]:
        #             tp[d] = 1.
        #             R['det'][jmax] = 1
        #         else:
        #             fp[d] = 1.
        # else:
        #     fp[d] = 1.

        for i, thresh in enumerate(ovthreshs):
            if ovmax > thresh:
                # print(ovmax)
                if not R['difficult'][jmax]:
                    if not R['det'][i][jmax]:
                        tp[d, i] = 1.
                        # TODO R['det'] 问题
                        R['det'][i][jmax] = 1
                    else:
                        fp[d, i] = 1.
            else:
                fp[d, i] = 1.

    # compute precision recall
    # print(tp)
    # fp = np.cumsum(fp)
    # tp = np.cumsum(tp)
    # rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    # prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # ap = voc_ap(rec, prec, use_07_metric)
    # print(fp)
    # print(tp)
    # print(rec)
    # print(prec)
    # print(ap)

    # TODO
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap50 = voc_ap(rec[:, 0], prec[:, 0], use_07_metric)
    # ap55 = voc_ap(rec[:, 1], prec[:, 1], use_07_metric)
    # ap60 = voc_ap(rec[:, 2], prec[:, 2], use_07_metric)
    # ap65 = voc_ap(rec[:, 3], prec[:, 3], use_07_metric)
    # ap70 = voc_ap(rec[:, 4], prec[:, 4], use_07_metric)
    # ap75 = voc_ap(rec[:, 5], prec[:, 5], use_07_metric)
    # ap80 = voc_ap(rec[:, 6], prec[:, 6], use_07_metric)
    # ap85 = voc_ap(rec[:, 7], prec[:, 7], use_07_metric)
    # ap90 = voc_ap(rec[:, 8], prec[:, 8], use_07_metric)
    # ap95 = voc_ap(rec[:, 9], prec[:, 9], use_07_metric)
    # ap = [ap50, ap55, ap60, ap65, ap70, ap75, ap80, ap85, ap90, ap95]
    # print(ap50)
    # print(ap55)
    return rec, prec, ap50
