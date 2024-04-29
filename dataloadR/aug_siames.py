# coding=utf-8
import cv2
import random
import numpy as np
import imgaug.augmenters as iaa

class HSV(object):
    def __init__(self, hgain=0.015, sgain=0.7, vgain=0.4, p=0.75):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.p = p
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            x = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x).clip(None, 255).astype(np.uint8)
            np.clip(img_hsv[:, :, 0], None, 179, out=img_hsv[:, :, 0])  # inplace hue clip (0 - 179 deg)
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return img, bboxes

class equalizeHist(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img, bboxes

class Blur(object):
    def __init__(self, sigma=1.3, p=0.15):
        self.sigma = sigma
        self.p = p
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            blur_aug = iaa.GaussianBlur(sigma=(0, self.sigma))
            img = blur_aug.augment_image(img)
        return img, bboxes


class Grayscale(object):
    def __init__(self, grayscale=0.3, p=0.5):
        self.alpha = random.uniform(grayscale, 1.0)
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            gray_aug = iaa.Grayscale(alpha=(self.alpha, 1.0))
            img = gray_aug.augment_image(img)
        return img, bboxes


class Gamma(object):
    def __init__(self, intensity=0.2, p=0.3):
        self.intensity = intensity
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            gm = random.uniform(1 - self.intensity, 1 + self.intensity)
            img = np.uint8(np.power(img / float(np.max(img)), gm) * np.max(img))
        return img, bboxes

class Noise(object):
    def __init__(self, intensity=0.01, p=0.15):
        self.intensity = intensity
        self.p = p
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            noise_aug = iaa.AdditiveGaussianNoise(scale=(0, self.intensity * 255))
            img = noise_aug.augment_image(img)
        return img, bboxes

class Sharpen(object):
    def __init__(self, intensity=0.15, p=0.2):
        self.intensity = intensity
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            sharpen_aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(1 - self.intensity, 1 + self.intensity))
            img = sharpen_aug.augment_image(img)
        return img, bboxes

class Contrast(object):
    def __init__(self, intensity=0.15, p=0.3):
        self.intensity = intensity
        self.p = p
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            contrast_aug = iaa.contrast.LinearContrast((1 - self.intensity, 1 + self.intensity))
            img = contrast_aug.augment_image(img)
        return img, bboxes

class RandomVerticalFilp(object):
    def __init__(self, p=1):
        self.p = p
    def __call__(self, img_rgb, img_dsm, bboxes):
        if random.random() < self.p:
            h_img, _, _ = img_rgb.shape
            img_rgb = img_rgb[::-1, :, :]
            img_dsm = img_dsm[::-1, :, :]
            bboxes[:, [1, 3]] = h_img - bboxes[:, [3, 1]]
            bboxes[:, [6, 8, 10, 12]] = h_img - bboxes[:, [6, 8, 10, 12]]
            bboxes[:, [5, 6, 9, 10]] = bboxes[:, [9, 10, 5, 6]]
            bboxes[:, [-1]] = 180 - bboxes[:, [-1]]#-(90 + bboxes[:, [-1]])
        return img_rgb, img_dsm, bboxes

class RandomHorizontalFilp(object):
    def __init__(self, p=1):
        self.p = p
    def __call__(self, img_rgb, img_dsm, bboxes):
        if random.random() < self.p:
            _, w_img, _ = img_rgb.shape
            img_rgb = img_rgb[:, ::-1, :]
            img_dsm = img_dsm[:, ::-1, :]
            bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
            bboxes[:, [5, 7, 9, 11]] = w_img - bboxes[:, [5, 7, 9, 11]]
            bboxes[:, [7, 8, 11, 12]] = bboxes[:, [11, 12, 7, 8]]
            bboxes[:, [-1]] = 180 - bboxes[:, [-1]]#-np.where(bboxes[:, [-1]] <= 0, 90 + bboxes[:, [-1]], 90 - bboxes[:, [-1]])
        return img_rgb, img_dsm, bboxes

class RandomCrop(object):
    def __init__(self, p=0.3):
        self.p = p
    def __call__(self, img_rgb, img_dsm, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img_rgb.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]
            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = min(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))#
            crop_ymax = min(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))#
            img_rgb = img_rgb[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
            img_dsm = img_dsm[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
            ####xmin,ymin,xmax,ymax,c,x1,y1,x2,y2,x3,y3,x4,y4,r
            bboxes[:, [0, 2, 5, 7, 9, 11]] = bboxes[:, [0, 2, 5, 7, 9, 11]] - crop_xmin
            bboxes[:, [1, 3, 6, 8, 10, 12]] = bboxes[:, [1, 3, 6, 8, 10, 12]] - crop_ymin
        return img_rgb, img_dsm, bboxes

class RandomRot(object):
    def __init__(self, p=1):
        self.p = p
    def __call__(self, image_rgb, image_dsm, bboxes, degree=90):
        if random.random() < self.p:
            pn = np.random.randint(low=0, high=4, size=None, dtype='l')
            (h, w) = image_rgb.shape[:2]
            M = np.eye(3)
            M[:2] = cv2.getRotationMatrix2D(angle=degree*pn, center=(w / 2, h / 2), scale=1.0)
            height, width = image_rgb.shape[:2]
            new_img_rgb = cv2.warpAffine(image_rgb, M[:2], dsize=(width, height), borderValue=(128, 128, 128))
            new_img_dsm = cv2.warpAffine(image_dsm, M[:2], dsize=(width, height), borderValue=(128, 128, 128))
            # Transform label coordinates
            n = len(bboxes)
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = bboxes[:, [5, 6, 7, 8, 9, 10, 11, 12]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = xy[:, :2].reshape(n, 8)
            # create new boxes
            x = xy[:, [0, 2, 4, 6]].clip(0, width)
            y = xy[:, [1, 3, 5, 7]].clip(0, height)
            if pn == 0:
                bboxes[:, [5, 7, 9, 11]] = x
                bboxes[:, [6, 8, 10, 12]] = y
            elif pn == 1:
                bboxes[:, [11, 5, 7, 9]] = x
                bboxes[:, [12, 6, 8, 10]] = y
            elif pn == 2:
                bboxes[:, [9, 11, 5, 7]] = x
                bboxes[:, [10, 12, 6, 8]] = y
            elif pn ==3:
                bboxes[:, [7, 9, 11, 5]] = x
                bboxes[:, [8, 10, 12, 6]] = y
            bboxes[:, [-1]] = bboxes[:, [-1]] + pn * 90
            bboxes[:, [-1]] = np.where(bboxes[:, [-1]] > 180, bboxes[:, [-1]] - 180, bboxes[:, [-1]])
            bboxes[:, [-1]] = np.where(bboxes[:, [-1]] > 180, bboxes[:, [-1]] - 180, bboxes[:, [-1]])
            xy4 = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            bboxes[:, [0, 1, 2, 3]] = xy4
            return new_img_rgb, new_img_dsm, bboxes


class RandomAffine(object):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img_rgb, img_dsm, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img_rgb.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            img_rgb = cv2.warpAffine(img_rgb, M, (w_img, h_img), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            img_dsm = cv2.warpAffine(img_dsm, M, (w_img, h_img), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))

            bboxes[:, [0, 2, 5, 7, 9, 11]] = bboxes[:, [0, 2, 5, 7, 9, 11]] + tx
            bboxes[:, [1, 3, 6, 8, 10, 12]] = bboxes[:, [1, 3, 6, 8, 10, 12]] + ty
        return img_rgb, img_dsm, bboxes


class Resize(object):
    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img_rgb, img_dsm, bboxes):
        h_org , w_org , _= img_rgb.shape
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB).astype(np.float32)

        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized_rgb = cv2.resize(img_rgb, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC if resize_ratio > 1 else cv2.INTER_AREA)
        image_paded_rgb = np.full((self.h_target, self.w_target, 3), 128.0)

        image_resized_dsm = cv2.resize(img_dsm, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC if resize_ratio > 1 else cv2.INTER_AREA)
        image_paded_dsm = np.full((self.h_target, self.w_target, 3), 128.0)

        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded_rgb[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized_rgb
        image_rgb = image_paded_rgb #/ 255.0

        image_paded_dsm[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized_dsm
        image_dsm = image_paded_dsm #/ 255.0
        if self.correct_box:
            ################################x1-y4 trans
            bboxes[:, [0, 2, 5, 7, 9, 11]] = bboxes[:, [0, 2, 5, 7, 9, 11]] * resize_ratio + dw
            bboxes[:, [1, 3, 6, 8, 10, 12]] = bboxes[:, [1, 3, 6, 8, 10, 12]] * resize_ratio + dh
            return image_rgb, image_dsm, bboxes
        return image_rgb, image_dsm

class Mosaic(object):
    def __init__(self, output_size, scale_range=(0.4, 0.6), filter_scale=1 / 50):
        self.output_size = output_size
        self.scale_range = scale_range
        self.filter_scale = 1 / 50
        self.output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
        self.factor = random.random()
        self.factor = random.random()
        self.scale_x = scale_range[0] + self.factor * (scale_range[1] - scale_range[0])
        self.scale_y = scale_range[0] + self.factor * (scale_range[1] - scale_range[0])
        self.divid_point_x = int(self.scale_x * output_size[1])
        self.divid_point_y = int(self.scale_y * output_size[0])

        self.output_img = np.ones([output_size[0], output_size[1], 3], dtype=np.uint8)*128

    def __call__(self, img1, img2, img3, img4, bboxes1, bboxes2, bboxes3, bboxes4):
        #top-left
        img1 = cv2.resize(img1, (self.divid_point_x, self.divid_point_y), interpolation=cv2.INTER_AREA)
        self.output_img[:self.divid_point_y, :self.divid_point_x, :] = img1
        bboxes1[:, [0, 2, 5, 7, 9, 11]] = bboxes1[:, [0, 2, 5, 7, 9, 11]] * self.scale_x
        bboxes1[:, [1, 3, 6, 8, 10, 12]] = bboxes1[:, [1, 3, 6, 8, 10, 12]] * self.scale_y
        #top-right#######
        self.r = min(1 - self.scale_x, self.scale_y)
        self.ratio = min(self.output_size[1] - self.divid_point_x, self.divid_point_y)
        img2 = cv2.resize(img2, (self.ratio, self.ratio), interpolation=cv2.INTER_AREA)
        if self.ratio == self.divid_point_y:
            max_trans = self.output_size[1] - self.divid_point_x- self.ratio
            trans = int(random.random() * max_trans)
            self.output_img[:self.ratio, self.divid_point_x + trans:self.divid_point_x+self.ratio + trans, :] = img2
            bboxes2[:, [0, 2, 5, 7, 9, 11]] = bboxes2[:, [0, 2, 5, 7, 9, 11]] * self.r + self.divid_point_x + trans
            bboxes2[:, [1, 3, 6, 8, 10, 12]] = bboxes2[:, [1, 3, 6, 8, 10, 12]] * self.r
        else:
            max_trans = self.divid_point_y - self.ratio
            trans = int(random.random() * max_trans)
            self.output_img[trans:trans+self.ratio, self.divid_point_x:self.divid_point_x+self.ratio, :] = img2
            bboxes2[:, [0, 2, 5, 7, 9, 11]] = bboxes2[:, [0, 2, 5, 7, 9, 11]] * self.r + self.divid_point_x
            bboxes2[:, [1, 3, 6, 8, 10, 12]] = bboxes2[:, [1, 3, 6, 8, 10, 12]] * self.r + trans
        # bottom-left######
        self.r1 = min(self.scale_x, 1 - self.scale_y)
        self.ratio1 = min(self.divid_point_x, self.output_size[0] - self.divid_point_y)
        img3 = cv2.resize(img3, (self.ratio1, self.ratio1), interpolation=cv2.INTER_AREA)
        if self.ratio1 == self.divid_point_x:

            max_trans = self.output_size[0] - self.divid_point_y - self.ratio1
            trans = int(random.random() * max_trans)
            self.output_img[self.divid_point_y + trans: self.divid_point_y + trans + self.ratio1, :self.ratio1, :] = img3
            bboxes3[:, [0, 2, 5, 7, 9, 11]] = bboxes3[:, [0, 2, 5, 7, 9, 11]] * self.r1
            bboxes3[:, [1, 3, 6, 8, 10, 12]] = bboxes3[:, [1, 3, 6, 8, 10, 12]] * self.r1 + self.divid_point_y + trans
        else:
            max_trans = self.divid_point_x - self.ratio1
            trans = int(random.random() * max_trans)
            self.output_img[self.divid_point_y:self.divid_point_y + self.ratio1, trans:trans + self.ratio1, :] = img3
            bboxes3[:, [0, 2, 5, 7, 9, 11]] = bboxes3[:, [0, 2, 5, 7, 9, 11]] * self.r1 + trans
            bboxes3[:, [1, 3, 6, 8, 10, 12]] = bboxes3[:, [1, 3, 6, 8, 10, 12]] * self.r1 + self.divid_point_y
        # bottom-right
        img4 = cv2.resize(img4, (self.output_size[1] - self.divid_point_x, self.output_size[0] - self.divid_point_y), interpolation=cv2.INTER_AREA)
        self.output_img[self.divid_point_y:self.output_size[0], self.divid_point_x:self.output_size[1], :] = img4
        bboxes4[:, [0, 2, 5, 7, 9, 11]] = bboxes4[:, [0, 2, 5, 7, 9, 11]] * (1 - self.scale_x) + self.divid_point_x
        bboxes4[:, [1, 3, 6, 8, 10, 12]] = bboxes4[:, [1, 3, 6, 8, 10, 12]] * (1 - self.scale_y) + self.divid_point_y
        image = self.output_img
        new_bboxes = np.concatenate((bboxes1, bboxes2, bboxes3, bboxes4), axis=0)
        return image, new_bboxes

class Resize_trans(object):
    def __init__(self, target_shape, final_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.final_shape = final_shape
        self.correct_box = correct_box
    def __call__(self, img, bboxes):
        h_org , w_org , _= img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC if resize_ratio > 1 else cv2.INTER_AREA)
        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized

        image_paded2 = np.full((self.final_shape, self.final_shape, 3), 128.0)
        dw2 = int((self.final_shape - self.w_target) / 2)
        dh2 = int((self.final_shape - self.h_target) / 2)
        image_paded2[dh2:self.h_target + dh2, dw2:self.w_target + dw2, :] = image_paded
        image = image_paded2 / 255.0
        if self.correct_box:
            bboxes[:, [0, 2, 5, 7, 9, 11]] = bboxes[:, [0, 2, 5, 7, 9, 11]] * resize_ratio + dw + dw2
            bboxes[:, [1, 3, 6, 8, 10, 12]] = bboxes[:, [1, 3, 6, 8, 10, 12]] * resize_ratio + dh + dh2
            return image, bboxes
        return image

class Resizetest(object):
    def __init__(self, target_shape):
        self.h_target, self.w_target = target_shape
    def __call__(self, img):
        h_org , w_org , _= img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized = cv2.resize(img, (resize_w, resize_h))
        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
        image = image_paded / 255.0
        return image

class Mixup(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):
        img_org = img_org / 255.0
        img_mix = img_mix / 255.0
        if random.random() < self.p:
            lam = np.random.beta(1.5, 1.5)
            img = lam * img_org + (1 - lam) * img_mix
            bboxes_org = np.concatenate(
                [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=1)
            bboxes_mix = np.concatenate(
                [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=1)
            bboxes = np.concatenate([bboxes_org, bboxes_mix])
        else:
            img = img_org
            bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)

        return img, bboxes

class Mixup1(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):
        img_org = img_org / 255.0
        img_mix = img_mix / 255.0
        img = img_org
        bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)
        return img, bboxes

class Mixup_False(object):
    def __call__(self, img_org_rgb, img_org_dsm, bboxes_org):
        img_org_rgb = img_org_rgb / 255.0
        img_rgb = img_org_rgb
        img_org_dsm = img_org_dsm / 255.0
        img_dsm = img_org_dsm
        bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)
        return img_rgb, img_dsm, bboxes

class LabelSmooth(object):
    def __init__(self, delta=0.01):
        self.delta = delta
    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes