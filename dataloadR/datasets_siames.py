import os
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
import config.config as cfg
import dataloadR.aug_siames as DataAug

class Construct_Dataset(Dataset):
    def __init__(self, anno_file_name, img_size=int(cfg.TRAIN["TRAIN_IMG_SIZE"])):
        self.img_size = img_size
        self.num_classes = len(cfg.DATA["CLASSES"])
        self.stride = [8, 16, 32]
        self.IOU_thresh = 0.3
        self.thresh_gh = 0.05
        self.__annotations = self.__load_annotations(anno_file_name)
        self.mosaic_p = 0.0
        self.Temp_Mixup = False
    def __len__(self):
        return len(self.__annotations)

    def __getitem__(self, item):
        img_ori_rgb, img_ori_dsm, bboxes_ori = self.__parse_annotation(self.__annotations[item])
        img_ori_rgb = img_ori_rgb.transpose(2, 0, 1)  # HWC->CHW
        img_ori_dsm = img_ori_dsm.transpose(2, 0, 1)  # HWC->CHW
        img_rgb, img_dsm, bboxes = DataAug.Mixup_False()(img_ori_rgb, img_ori_dsm, bboxes_ori)
        del img_ori_rgb, img_ori_dsm, bboxes_ori
        label_sbbox, label_mbbox, label_lbbox = self.__creat_label(bboxes)
        img_rgb = torch.from_numpy(img_rgb).float()
        img_dsm = torch.from_numpy(img_dsm).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        return img_rgb, img_dsm, label_sbbox, label_mbbox, label_lbbox

    def __load_annotations(self, anno_name):
        anno_path = os.path.join(cfg.PROJECT_PATH, 'dataR', anno_name + ".txt")
        with open(anno_path, 'r') as f:
            annotations = list(filter(lambda x: len(x) > 0, f.readlines()))
        assert len(annotations) > 0, "No images found in {}".format(anno_path)
        return annotations

    def __parse_annotation(self, annotation):
        anno = annotation.strip().split(' ')
        img_path = anno[0]
        img_rgb = cv2.imread(img_path)
        img_dsm = cv2.imread(img_path.replace('rgb', 'dsm'))
        assert img_rgb is not None, 'File Not Found ' + img_path
        bboxes = np.array([list(map(float, box.split(','))) for box in anno[1:]])
        img_rgb, img_dsm, bboxes = DataAug.RandomVerticalFilp()(np.copy(img_rgb), np.copy(img_dsm), np.copy(bboxes))
        img_rgb, img_dsm, bboxes = DataAug.RandomHorizontalFilp()(np.copy(img_rgb), np.copy(img_dsm), np.copy(bboxes))
        img_rgb, bboxes = DataAug.HSV()(np.copy(img_rgb), np.copy(bboxes))
        img_rgb, img_dsm, bboxes = DataAug.RandomCrop()(np.copy(img_rgb), np.copy(img_dsm), np.copy(bboxes))
        img_rgb, img_dsm, bboxes = DataAug.RandomAffine()(np.copy(img_rgb), np.copy(img_dsm), np.copy(bboxes))
        img_rgb, img_dsm, bboxes = DataAug.Resize((self.img_size, self.img_size), True)(np.copy(img_rgb), np.copy(img_dsm), np.copy(bboxes))
        img_rgb, img_dsm, bboxes = DataAug.RandomRot()(np.copy(img_rgb), np.copy(img_dsm), np.copy(bboxes))
        return img_rgb, img_dsm, bboxes

    def generate_label(self, k, gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                       ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                       gt_label, class_id):
        ws = self.img_size // self.stride[k]
        hs = self.img_size // self.stride[k]
        grid_x = int(c_x_r // self.stride[k])
        grid_y = int(c_y_r // self.stride[k])
        r_w = len_w / self.stride[k] * 0.5 + 1e-16
        r_h = len_h / self.stride[k] * 0.5 + 1e-16
        r_w_max = int(np.clip(np.power(box_w / self.stride[k] / 2, 1), 1, np.inf))#
        r_h_max = int(np.clip(np.power(box_h / self.stride[k] / 2, 1), 1, np.inf))#
        sub_xmin = max(grid_x - r_w_max - 1 + 1, 0)
        sub_xmax = min(grid_x + r_w_max + 1 + 1, ws - 1)
        sub_ymin = max(grid_y - r_h_max - 1 + 1, 0)
        sub_ymax = min(grid_y + r_h_max + 1 + 1, hs - 1)
        gt_tensor_oval_1 = np.zeros([hs, ws, 1])
        R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        Eig = np.array([[2 / (r_w), 0], [0, 2 / (r_h)]])
        top_gaussian = []
        for i in range(sub_xmin, sub_xmax + 1):
            for j in range(sub_ymin, sub_ymax + 1):
                ax = np.array([[i - grid_x, j - grid_y]]).transpose()
                axnew = np.dot(np.dot(Eig, R), ax)
                v = np.exp(- (axnew[0, 0] ** 2 + axnew[1, 0] ** 2) / 2) # / (2 * np.pi)
                if v > self.thresh_gh: top_gaussian.append(v)
        if len(top_gaussian) == 0:
            top_gaussian.append(0)
        N = max(int(np.ceil(len(top_gaussian) * (1 - self.IOU_thresh))), 5)
        if len(top_gaussian) < N:
            gauss_thr = min(top_gaussian)
        else:
            top_gaussian.sort(reverse=True)
            gauss_thr = top_gaussian[N-1]
        for i in range(sub_xmin, sub_xmax + 1):
            for j in range(sub_ymin, sub_ymax + 1):
                ax = np.array([[i - grid_x, j - grid_y]]).transpose()
                axnew = np.dot(np.dot(Eig, R), ax)
                v = np.exp(- (axnew[0, 0] ** 2 + axnew[1, 0] ** 2) / 2) # / (2 * np.pi)
                pre_v_oval = gt_tensor_oval_1[j, i, 0:1]
                maxv = max(v, pre_v_oval)
                l1 = (j * self.stride[k] + self.stride[k] / 2) - ymin
                l2 = xmax - (i * self.stride[k] + self.stride[k] / 2)
                l3 = ymax - (j * self.stride[k] + self.stride[k] / 2)
                l4 = (i * self.stride[k] + self.stride[k] / 2) - xmin
                ori_gh_fg = np.max(gt_tensor[k][j, i, 16 + class_id: 16 + class_id + 1], axis=-1)
                ori_gh_bg = np.max(gt_tensor[k][j, i, 13: 14], axis=-1)
                ori_gh = max(ori_gh_fg, ori_gh_bg)
                if (ori_gh <= maxv) and maxv > self.thresh_gh and min(l1, l2, l3, l4) > 0:
                    gt_tensor[k][j, i, 0:8] = np.array([c_x, c_y, box_w, box_h, l1, l2, l3, l4])
                    gt_tensor[k][j, i, 8:12] = np.array([a1, a2, a3, a4])
                    gt_tensor[k][j, i, 12] = gt_label[13]
                    gt_tensor[k][j, i, 15] = 2 * np.log(2) / np.log(np.sqrt(N) + 1)
                    gt_tensor[k][j, i, 16 + class_id:16 + class_id + 1] = maxv
                    if maxv >= gauss_thr:
                        gt_tensor[k][j, i, 13] = 1.0
                    elif maxv < gauss_thr:
                        gt_tensor[k][j, i, 13] = maxv
        gt_tensor[k][:, :, 14] = gt_label[15]

    def __creat_label(self, label_lists=[]):
        self.gt_tensor = [np.zeros((int(self.img_size // self.stride[i]), int(self.img_size // self.stride[i]),
                               self.num_classes + 16)) for i in range(3)]
        ratio = (1 - self.IOU_thresh)
        layer_thresh = [3*(self.stride[0]*2)/ratio, 3*(self.stride[2]*2)/ratio]#[200,400]#
        for gt_label in label_lists:
            bbox_xyxy = gt_label[:4]
            bbox_obb = gt_label[5:13]
            xmin, ymin, xmax, ymax = bbox_xyxy
            box_w = (xmax - xmin)
            box_h = (ymax - ymin)
            if (max(box_w, box_h) > 10 or (box_w*box_w) > 80) and box_w > 4 and box_h > 4:
                c_x = (xmax + xmin) / 2
                c_y = (ymax + ymin) / 2
                if gt_label[13] > 0.9:
                    a1 = a2 = a3 = a4 = 0
                else:
                    a1 = (bbox_obb[0] - bbox_xyxy[0]) / box_w
                    a2 = (bbox_obb[3] - bbox_xyxy[1]) / box_h
                    a3 = (bbox_xyxy[2] - bbox_obb[4]) / box_w
                    a4 = (bbox_xyxy[3] - bbox_obb[7]) / box_h
                class_id = int(gt_label[4]) #- 4
                len_w = (np.sqrt((bbox_obb[5] - bbox_obb[3]) ** 2 + (bbox_obb[2] - bbox_obb[4]) ** 2)
                         + np.sqrt((bbox_obb[7] - bbox_obb[1]) ** 2 + (bbox_obb[0] - bbox_obb[6]) ** 2)) / 2
                len_h = (np.sqrt((bbox_obb[1] - bbox_obb[3]) ** 2 + (bbox_obb[2] - bbox_obb[0]) ** 2) +
                         np.sqrt((bbox_obb[4] - bbox_obb[6]) ** 2 + (bbox_obb[5] - bbox_obb[7]) ** 2)) / 2
                c_x_r = (bbox_obb[0] + bbox_obb[2] + bbox_obb[4] + bbox_obb[6]) / 4
                c_y_r = (bbox_obb[1] + bbox_obb[3] + bbox_obb[5] + bbox_obb[7]) / 4
                angle = gt_label[14] * np.pi/180
                if len_w < len_h:
                    len_w, len_h = len_h, len_w
                length = max(box_w, box_h)
                if length <= layer_thresh[0]:
                    self.generate_label(0, self.gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                                        ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                                        gt_label, class_id)
                if length > layer_thresh[0] and length <= layer_thresh[1]:
                    self.generate_label(1, self.gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                                        ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                                        gt_label, class_id)
                if length > layer_thresh[1]:
                    self.generate_label(2, self.gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                                        ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                                        gt_label, class_id)
        label_sbbox, label_mbbox, label_lbbox = self.gt_tensor
        return label_sbbox, label_mbbox, label_lbbox


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_dataset = Construct_Dataset(anno_file_name="train_build", img_size=800)
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=False)
    for i, (imgs_rgb, imgs_dsm, label_sbbox, label_mbbox, label_lbbox) in enumerate(train_dataloader):
        continue