import logging
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import utils.gpu as gpu
from utils import cosine_lr_scheduler
from utils.log import Logger
import dataloadR.datasets_siames as data
from modelR.MuDet import MuDet
from modelR.loss.loss import Loss
from evalR.evaluatorGGHL import *
from torch.cuda import amp
import torch.backends.cudnn as cudnn
from copy import deepcopy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
class Trainer(object):
    def __init__(self,  weight_path, resume, gpu_id):
        init_seeds(0)
        self.__num_class = cfg.DATA["NUM"]

        self.device = gpu.select_device(gpu_id)
        print(self.device)
        self.cuda = self.device.type != 'cpu'
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        if self.multi_scale_train: print('Using multi scales training')
        else: print('train img size is {}'.format(cfg.TRAIN["TRAIN_IMG_SIZE"]))

        self.train_dataset = data.Construct_Dataset(anno_file_name=cfg.DATASET_NAME, img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                           shuffle=True,
                                           pin_memory=True)

        net_model = MuDet(weight_path=self.weight_path)
        self.model = net_model.to(self.device) ## Single GPU
        torch.backends.cudnn.benchmark = True

        # Optimizer
        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, torch.nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):  # weight (with decay)
                g1.append(v.weight)
        self.optimizer = optim.SGD(g0, lr=cfg.TRAIN["LR_INIT"], momentum=cfg.TRAIN["MOMENTUM"], nesterov=True)
        self.optimizer.add_param_group({'params': g1, 'weight_decay': cfg.TRAIN["WEIGHT_DECAY"]})  # add g1 with weight_decay
        self.optimizer.add_param_group({'params': g2})  # add g2 (biases)
        del g0, g1, g2
        self.criterion = Loss()

        self.__load_model_weights(self.weight_path, resume)
        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                           T_max=self.epochs*len(self.train_dataloader),
                                                           lr_init=cfg.TRAIN["LR_INIT"],
                                                           lr_min=cfg.TRAIN["LR_END"],
                                                           warmup=cfg.TRAIN["WARMUP_EPOCHS"] * len(self.train_dataloader))
        self.scaler = amp.GradScaler(enabled=self.cuda)

    def __load_model_weights(self, weight_path, resume):
        if resume:
            last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
            chkpt = torch.load(last_weight, map_location=self.device)
            self.model.load_state_dict(chkpt['model'])
            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
            del chkpt
        else:
            self.model.load_darknet_weights(weight_path) ## Single GPU

    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(os.path.split(self.weight_path)[0], "best.pt")
        last_weight = os.path.join(os.path.split(self.weight_path)[0], "last.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)
        if epoch > 0 and epoch % 5 == 0:
            torch.save(chkpt, os.path.join(os.path.split(self.weight_path)[0], 'backup_epoch%g.pt'%epoch))
        del chkpt

    def train(self):
        global writer
        logger.info(" Training start!  Img size:{:d},  Batchsize:{:d},  Number of workers:{:d}".format(
            cfg.TRAIN["TRAIN_IMG_SIZE"], cfg.TRAIN["BATCH_SIZE"], cfg.TRAIN["NUMBER_WORKERS"]))
        logger.info(" Train datasets number is : {}".format(len(self.train_dataset)))
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            self.model.train()
            mloss = torch.zeros(10)
            for i, (imgs_rgb, imgs_dsm, label_sbbox, label_mbbox, label_lbbox) in enumerate(self.train_dataloader):

                self.scheduler.step(len(self.train_dataloader)*epoch + i)
                imgs_rgb = imgs_rgb.to(self.device)
                imgs_dsm = imgs_dsm.to(self.device)
                with amp.autocast(enabled=self.cuda):
                    p, p_d = self.model(imgs_rgb, imgs_dsm)
                    label_sbbox = label_sbbox.to(self.device)
                    label_mbbox = label_mbbox.to(self.device)
                    label_lbbox = label_lbbox.to(self.device)
                    loss, loss_fg, loss_bg, loss_pos, loss_neg, loss_iou, loss_cls, loss_s, loss_r, loss_l = self.criterion(p, p_d, label_sbbox, label_mbbox, label_lbbox, epoch, i)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                loss_items = 10 * torch.tensor([loss_fg, loss_bg, loss_pos, loss_neg, loss_iou, loss_cls, loss_s, loss_r, loss_l, loss])
                mloss = (mloss * i + loss_items) / (i + 1)
                mAP = 0
                if i % 50 == 0:
                    logger.info(
                        "Epoch:[{:3}/{}] Batch:[{:3}/{}] Img_size:[{:3}] Loss:{:.4f} "
                        "Loss_fg:{:.4f} | Loss_bg:{:.4f} | Loss_pos:{:.4f} | Loss_neg:{:.4f} "
                        "| Loss_iou:{:.4f} | Loss_cls:{:.4f} | Loss_s:{:.4f} | Loss_r:{:.4f} | "
                        "Loss_l:{:.4f} | LR:{:g}".format(
                            epoch, self.epochs, i, len(self.train_dataloader) - 1, self.train_dataset.img_size,
                            mloss[9], mloss[0], mloss[1], mloss[2], mloss[3],
                            mloss[4], mloss[5], mloss[6], mloss[7], mloss[8 ],
                            self.optimizer.param_groups[0]['lr']
                        ))
                    writer.add_scalar('loss_fg', mloss[0], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_bg', mloss[1], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_pos', mloss[2], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_neg', mloss[3], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_iou', mloss[4], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_cls', mloss[5], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_s', mloss[6], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_r', mloss[7], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_l', mloss[8], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('train_loss', mloss[9], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)

                if self.multi_scale_train and (i+1) % 10 == 0:
                    self.train_dataset.img_size = random.choice(range(
                        cfg.TRAIN["MULTI_TRAIN_RANGE"][0], cfg.TRAIN["MULTI_TRAIN_RANGE"][1],
                        cfg.TRAIN["MULTI_TRAIN_RANGE"][2])) * 32

            if epoch >= 20 and epoch % 1 == 0 and cfg.TRAIN["EVAL_TYPE"] == 'VOC':
                logger.info("===== Validate =====".format(epoch, self.epochs))
                with torch.no_grad():
                    start = time.time()
                    APs, r, p, inference_time = Evaluator(deepcopy(self.model)).APs_voc()#deepcopy(self.model).cuda()
                    end = time.time()
                    logger.info("Test cost time:{:.4f}s".format(end - start))
                    for i in APs:
                        print("{} --> AP : {}".format(i, APs[i]))
                        mAP += APs[i]
                    mAP = mAP / self.__num_class
                    logger.info('mAP:{}'.format(mAP))
                    logger.info("inference time: {:.2f} ms".format(inference_time))
                    writer.add_scalar('test/VOCmAP', mAP)
            self.__save_model_weights(epoch, mAP)
            logger.info('Save weights Done')
            logger.info("mAP: {:.3f}".format(mAP))
            end = time.time()
            logger.info("Inference time: {:.4f}s".format(end - start))

        logger.info("Training finished.  Best_mAP: {:.3f}%".format(self.best_mAP))

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
    global logger, writer
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/darknet53.weights', help='weight file path')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--log_path', type=str, default='log/', help='log path')
    opt = parser.parse_args()
    writer = SummaryWriter(logdir=opt.log_path + '/event')
    logger = Logger(log_file_name=opt.log_path + '/log.txt', log_level=logging.DEBUG, logger_name='GGHLs').get_log()
    Trainer(weight_path=opt.weight_path, resume=opt.resume, gpu_id=opt.gpu_id).train()