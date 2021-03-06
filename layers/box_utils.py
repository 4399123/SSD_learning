# -*- coding: utf-8 -*-
import torch
import numpy as np
import torchvision


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

# 输入包括IoU阈值、真实边框位置、预选框、方差、真实边框类别
# 输出为每一个预选框的类别，保存在conf_t中，对应的真实边框位置，保存在loc_t中
def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """

    # 注意这里truth是最大最小值形式的,而prior是中心点与长宽形式
    # 求取真实框与预选框的IoU
    overlaps = jaccard(
        truths,
        point_form(priors)
    )

    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True) #求出GT与预测框中 IOU最大的框与IOU值
    # 将每一个真实框对应的最佳PriorBox的IoU设置为2
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True) #求出每个预测框跟 哪个个GT的IOU最大
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)


    # 将每一个truth对应的最佳box的overlap设置为2
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap

    # 保证每一个truth对应的最佳box,该box要对应到这个truth上,即使不是最大iou
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j


    # 每一个prior对应的真实框的位置
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]

    # 每一个prior对应的类别
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]

    # 如果一个PriorBox对应的最大IoU小于0.5，则视为负样本
    conf[best_truth_overlap < threshold] = 0  # label as background

    # 进一步计算定位的偏移真值
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    # 这个地方为什么要多此一举,本身是log(sum(exp(x)))
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(bboxes, scores, threshold=0.2, top_k=200):  #bboxes维度为[N,4],scores维度为[N,],均为tensor
    x1 = bboxes[:, 0]   #获得每一个框的左上角和右下角坐标
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas=(x2-x1)*(y2-y1)  #获得每个框的面积
    _,order=scores.sort(0,descending=True)  #按降序排列
    order=order[:top_k]     #取前top_k个
    keep=[]
    count=0
    while order.numel()>0:
        if order.numel()==1:
            break
        count += 1
        # print(order)
        i=order[0]
        keep.append(i)

        xx1=x1[order[1:]].clamp(min=x1[i].item())   #[N-1,]
        yy1=y1[order[1:]].clamp(min=y1[i].item())
        xx2=x2[order[1:]].clamp(max=x2[i].item())
        yy2=y2[order[1:]].clamp(max=y2[i].item())

        w=(xx2-xx1).clamp(min=0)
        h=(yy2-yy1).clamp(min=0)
        inter=w*h                        #相交的面积  [N-1,]

        overlap=inter/(areas[i]+areas[order[1:]]-inter)  #计算IOU   [N-1,]
        ids=(overlap<=threshold).nonzero().squeeze()   #返回一个包含输入 input 中非零元素索引的张量.输出张量中的每行包含 input 中非零元素的索引
        if ids.numel()==0:
            break
        order=order[ids+1]           #ids中索引为0的值在order中实际为1，后面所有的元素也一样，新的order是经过了一轮计算后留下来的bbox的索引
    print(torch.tensor(keep,dtype=torch.long))
    return torch.tensor(keep,dtype=torch.long),count

def PytorchNMS(bboxes, scores, threshold=0.2, top_k=200):  #bboxes维度为[N,4],scores维度为[N,],均为tensor

    _,order=scores.sort(0,descending=True)  #按降序排列
    order=order[:top_k]     #取前top_k个

    newbox=bboxes[order,:]
    newscore=scores[order]
    kkeep=torchvision.ops.nms(boxes=newbox,scores=newscore,iou_threshold=threshold)
    count=len(np.array(kkeep))
    keep=order[kkeep]

    return keep, count

def DIOUnms(bboxes, scores, threshold=0.2, top_k=200):  #bboxes维度为[N,4],scores维度为[N,],均为tensor
    x1 = bboxes[:, 0]   #获得每一个框的左上角和右下角坐标
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    center_x=x2-x1/2.0
    center_y=y2-y1/2.0

    areas=(x2-x1)*(y2-y1)  #获得每个框的面积
    _,order=scores.sort(0,descending=True)  #按降序排列
    order=order[:top_k]     #取前top_k个
    keep=[]
    count=0
    while order.numel()>0:
        if order.numel()==1:
            break
        count += 1
        i=order[0]
        keep.append(i)

        xx1=x1[order[1:]].clamp(min=x1[i].item())   #[N-1,]
        yy1=y1[order[1:]].clamp(min=y1[i].item())
        xx2=x2[order[1:]].clamp(max=x2[i].item())
        yy2=y2[order[1:]].clamp(max=y2[i].item())

        w=(xx2-xx1).clamp(min=0)
        h=(yy2-yy1).clamp(min=0)
        inter=w*h                        #相交的面积  [N-1,]

        overlap=inter/(areas[i]+areas[order[1:]]-inter)  #计算IOU   [N-1,]

        xxx1=[]
        xxx2=[]
        yyy1=[]
        yyy2=[]
        for j in range(len(np.array(xx1))):
            xxx1.append(min(x1[order[j+1]].item(), x1[i].item()))
            xxx2.append(min(x2[order[j+1]].item(), x2[i].item()))
            yyy1.append(max(y1[order[j+1]].item(), y1[i].item()))
            yyy2.append(max(y2[order[j+1]].item(), y2[i].item()))

        xxx1 = torch.Tensor(xxx1).clamp(min=0)
        xxx2 = torch.Tensor(xxx2).clamp(min=0)
        yyy1 = torch.Tensor(yyy1)
        yyy2 = torch.Tensor(yyy2)

        Cdistance=torch.pow(xxx2-xxx1,2)+torch.pow(yyy2-yyy1,2)
        Ddistance=torch.pow(center_x[i]-center_x[order[1:]],2)+torch.pow(center_y[i]-center_y[order[1:]],2)

        overlap=overlap-Ddistance/Cdistance

        ids=(overlap<=threshold).nonzero().squeeze()   #返回一个包含输入 input 中非零元素索引的张量.输出张量中的每行包含 input 中非零元素的索引
        if ids.numel()==0:
            break
        order=order[ids+1]           #ids中索引为0的值在order中实际为1，后面所有的元素也一样，新的order是经过了一轮计算后留下来的bbox的索引
    return torch.tensor(keep,dtype=torch.long),count