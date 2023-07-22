from __future__ import print_function
import argparse
import os
import random
from copy import deepcopy

import onnx
import onnxruntime as onnxruntime
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetSegmentation, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=8, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument(
    '--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default='E:/SANY/Program/Pytorch/PointNet/SPointNet/data/',
                    help="dataset path")
parser.add_argument('--class_choice', type=str, default='Airplane', help="class_choice")
# parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--feature_transform', default=True, help="use feature transform")
parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

train_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice])
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(train_dataset), len(test_dataset))
num_classes = train_dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

segmentation_model = PointNetSegmentation(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    segmentation_model.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(segmentation_model.parameters(), lr=0.01, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
segmentation_model.cuda()

num_batch = len(train_dataset) / opt.batchSize

try:
    onnx_model = onnx.load('E:/SANY/Program/Pytorch/PointNet/SPointNet/utils/seg/segmentation_model_Airplane_0.onnx')
    onnx.checker.check_model(onnx_model)
    print('ONNX export successful')
    ort_session = onnxruntime.InferenceSession
except Exception as e:
    print('ONNX export failure: %s' % e)

x = []
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

# Resume
start_epoch = 0
best_fitness = 0.0

last = opt.outf + '/segmentation_model_' + opt.class_choice + '_last.pt'
best = opt.outf + '/segmentation_model_' + opt.class_choice + '_best.pt'

if opt.resume:
    checkpoint_path = last
    checkpoint = torch.load(checkpoint_path)
    segmentation_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['lr_schedule_state_dic'])
    best_fitness = checkpoint['best_fitness']
    start_epoch = checkpoint['epoch'] + 1

for epoch in range(start_epoch, opt.nepoch):
    scheduler.step()  # 更新优化器的学习率,一般按照epoch为单位进行更新
    for i, data in enumerate(train_dataloader, 0):
        # 训练
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()  # 清空过往梯度
        segmentation_model = segmentation_model.train()
        pred, trans, trans_feat = segmentation_model(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0] - 1
        # print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()  # 反向传播，计算当前梯度
        optimizer.step()  # 根据梯度更新网络参数
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (
            epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize * 2500)))

    # 评估
    _, data = next(enumerate(test_dataloader, 0))
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    segmentation_model = segmentation_model.eval()
    pred, _, _ = segmentation_model(points)
    pred = pred.view(-1, num_classes)
    target = target.view(-1, 1)[:, 0] - 1
    val_loss = F.nll_loss(pred, target)
    pred_choice = pred.data.max(1)[1]
    val_correct = pred_choice.eq(target.data).cpu().sum()
    accuracy = val_correct.item() / float(opt.batchSize * 2500)

    if best_fitness < accuracy:
        best_fitness = accuracy

    ckpt = {'epoch': epoch,
            'best_fitness': best_fitness,
            'model_state_dict': deepcopy(segmentation_model.state_dict()),
            # state_dict只是一个Python字典对象，它将每一层映射到其参数张量(tensor)
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_schedule_state_dic': scheduler.state_dict()}
    torch.save(ckpt, last)
    if best_fitness == accuracy:
        torch.save(ckpt, best)

    print('[%d] %s loss: %f accuracy: %f' % (
        epoch, blue('test'), val_loss.item(), val_correct.item() / float(opt.batchSize * 2500)))

    # # torch.save(segmentation_model.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))
    # sm = torch.jit.script(segmentation_model)
    # sm.save('%s/segmentation_model_%s_%d.pt' % (opt.outf, opt.class_choice, epoch))
    #
    # dummy_input = torch.randn(opt.batchSize, 3, 2500).cuda()
    # onnx_name = opt.outf + '/segmentation_model_' + opt.class_choice + '_' + str(epoch) + '.onnx'
    # torch.onnx.export(segmentation_model, dummy_input, onnx_name, input_names=['InputPC'], output_names=['OutputSeg'])

# benchmark mIOU
shape_ious = []
for i, data in tqdm(enumerate(test_dataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    segmentation_model = segmentation_model.eval()
    pred, _, _ = segmentation_model(points)
    pred_choice = pred.data.max(2)[1]

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy() - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)  # np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))
