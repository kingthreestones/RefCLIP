

import torch

import os
import random
import torch.optim.lr_scheduler as lr_scheduler
import math
import numpy as np


class EMA(object):
    '''
        apply expontential moving average to a model. This should have same function as the `tf.train.ExponentialMovingAverage` of tensorflow.
        usage:
            model = resnet()
            model.train()
            ema = EMA(model, 0.9999)
            ....
            for img, lb in dataloader:
                loss = ...
                loss.backward()
                optim.step()
                ema.update_params() # apply ema
            evaluate(model)  # evaluate with original model as usual
            ema.apply_shadow() # copy ema status to the model
            evaluate(model) # evaluate the model with ema paramters
            ema.restore() # resume the model parameters
        args:
            - model: the model that ema is applied
            - alpha: each parameter p should be computed as p_hat = alpha * p + (1. - alpha) * p_hat
            - buffer_ema: whether the model buffers should be computed with ema method or just get kept
        methods:
            - update_params(): apply ema to the model, usually call after the optimizer.step() is called
            - apply_shadow(): copy the ema processed parameters to the model
            - restore(): restore the original model parameters, this would cancel the operation of apply_shadow()
    '''
    def __init__(self, model, alpha, buffer_ema=True):
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = self.model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name]
                + (1 - decay) * state[name]
            )
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(
                    decay * self.shadow[name]
                    + (1 - decay) * state[name]
                )
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }


def setup_unique_version(__C):
    # if __C.RESUME:
    #     __C.VERSION = __C.RESUME_VERSION
    #     return
    while True:
        version = random.randint(0, 99999)
        # version = 77263
        if not (os.path.exists(os.path.join(__C.LOG_PATH ,str(version)))):
            __C.VERSION = str(version)
            break

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def batch_box_iou(box1, box2,threshold=0.5,iou_out=False):
    """
    :param box1:  N,4
    :param box2:  N,4
    :return: N
    """
    in_h = torch.min(box1[:,2], box2[:,2]) - torch.max(box1[:,0], box2[:,0])
    in_w = torch.min(box1[:,3], box2[:,3]) - torch.max(box1[:,1], box2[:,1])
    in_h=in_h.clamp(min=0.)
    in_w=in_w.clamp(min=0.)
    inter =in_h * in_w
    union = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1]) + \
            (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1]) - inter
    iou = inter / union
    if iou_out:
        return iou > threshold,iou
    else:
        return iou>threshold



def label2yolobox(labels, info_img, maxsize, lrflip=False):
    """
    Transform coco labels to yolo box labels
    Args:
        labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
            Each label consists of [class, x, y, w, h] where \
                class (float): class index.
                x, y, w, h (float) : coordinates of \
                    left-top points, width, and height of a bounding box.
                    Values range from 0 to width or height of the image.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
        maxsize (int): target image size after pre-processing
        lrflip (bool): horizontal flip flag

    Returns:
        labels:label data whose size is :math:`(N, 5)`.
            Each label consists of [class, xc, yc, w, h] where
                class (float): class index.
                xc, yc (float) : center of bbox whose values range from 0 to 1.
                w, h (float) : size of bbox whose values range from 0 to 1.
    """
    h, w, nh, nw, dx, dy,_ = info_img
    x1 = labels[:, 0] / w
    y1 = labels[:, 1] / h
    x2 = (labels[:, 0] + labels[:, 2]) / w
    y2 = (labels[:, 1] + labels[:, 3]) / h
    labels[:, 0] = (((x1 + x2) / 2) * nw + dx) / maxsize
    labels[:, 1] = (((y1 + y2) / 2) * nh + dy) / maxsize

    labels[:, 2] = (labels[:, 2] * (nw / w / maxsize))
    labels[:, 3] = (labels[:, 3] * (nh / h / maxsize))

    # labels[:, 2] *= nw / w / maxsize
    # labels[:, 3] *= nh / h / maxsize
    labels[:,:4]=np.clip(labels[:,:4],0.,0.99)
    if lrflip:
        labels[:, 0] = 1 - labels[:, 0]
    return labels


def yolobox2label(box, info_img):
    """
    Transform yolo box labels to yxyx box labels.
    Args:
        box (list): box data with the format of [yc, xc, w, h]
            in the coordinate system after pre-processing.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    Returns:
        label (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system of the input image.
    """
    # h, w, nh, nw, dx, dy,_ = info_img
    # y1, x1, y2, x2 = box
    # box_h = ((y2 - y1) / nh) * h
    # box_w = ((x2 - x1) / nw) * w
    # y1 = ((y1 - dy) / nh) * h
    # x1 = ((x1 - dx) / nw) * w
    # label = [y1, x1, y1 + box_h, x1 + box_w]
    h, w, nh, nw, dx, dy,_ = info_img
    x1, y1, x2, y2 = box[:4]
    box_h = ((y2 - y1) / nh) * h
    box_w = ((x2 - x1) / nw) * w
    y1 = ((y1 - dy) / nh) * h
    x1 = ((x1 - dx) / nw) * w
    label = [x1, y1,x1 + box_w, y1 + box_h]
    return np.concatenate([np.array(label),box[4:]])

def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

def get_lr_scheduler(__C,optimizer):
    if __C.SCHEDULER == 'step':
        t,T=__C.WARMUP,__C.EPOCHS
        def lr_func(epoch):
            coef = 1.
            if epoch<=t:
                coef=float(epoch)/float(t+1)
            else:
                for i,deps in enumerate(__C.DECAY_EPOCHS):
                    if epoch>=deps:
                        coef=__C.LR_DECAY_R**(i+1)
            return coef
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch:lr_func(epoch))
    elif __C.SCHEDULER == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=__C.EPOCHS)
    else:
        t,T=__C.WARMUP,__C.EPOCHS
        n_t=0.5
        lr_func = lambda epoch: (0.9 * epoch / t + __C.LR) if epoch < t else __C.LR if n_t * (
                    1 + math.cos(math.pi * (epoch - t) / (T - t))) < __C.LR else n_t * (
                    1 + math.cos(math.pi * (epoch - t) / (T - t)))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    return scheduler

def get_lr_scheduler(__C,optimizer,n_iter_per_epoch):
    num_steps = int(__C.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(__C.WARMUP * n_iter_per_epoch)
    if __C.SCHEDULER == 'step':
        #default step lr
        t,T=__C.WARMUP*n_iter_per_epoch,__C.EPOCHS*n_iter_per_epoch
        def lr_func(step):
            coef = 1.
            if step<=t:
                coef=float(step)/float(t+1)
            else:
                for i,deps in enumerate(__C.DECAY_EPOCHS):
                    if step>=deps*n_iter_per_epoch:
                        coef=__C.LR_DECAY_R**(i+1)
            return coef
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif __C.SCHEDULER == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=__C.EPOCHS*n_iter_per_epoch)
    else:
        t, T = __C.WARMUP * n_iter_per_epoch, __C.EPOCHS * n_iter_per_epoch
        n_t=0.5
        warm_step_lr=(__C.LR - __C.WARMUP_LR) / t
        lr_func = lambda step: ( step*warm_step_lr + __C.WARMUP_LR)/__C.LR if step < t \
            else (__C.MIN_LR + n_t * (__C.LR - __C.MIN_LR) * (1 + math.cos(math.pi * (step - t) / (T - t))))/__C.LR

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    return scheduler