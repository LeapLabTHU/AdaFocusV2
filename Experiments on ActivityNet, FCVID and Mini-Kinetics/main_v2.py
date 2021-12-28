import os
import shutil
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from ops.dataset import TSNDataSet
from ops.transforms import *
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, cal_map

from opts import parser
from models.adafocus_v2 import AdaFocus

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    global args, best_acc1
    args = parser.parse_args()
    args.root_model = args.root_log
    if args.sample:
        args.input_order = input_order(args.num_segments)
    best_acc1 = 0.
    if args.glance_arch == 'res50' and args.glance_ckpt_path == '' and not args.evaluate:
        print('WARNING: res50 initialization checkpoint not specified')
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.data_dir)
    args.num_classes = num_class
    args.store_name = '_'.join(
        ['AdaFocusV2', args.dataset, args.glance_arch, 'segment%d' % args.num_segments, 'e{}'.format(args.epochs)])
    check_rootfolders()

    model = AdaFocus(args)
    start_epoch = 0

    scale_size = model.scale_size
    crop_size = model.crop_size
    input_mean = model.input_mean
    input_std = model.input_std
    train_augmentation = model.get_augmentation(flip=True)

    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    policies = model.module.get_optim_policies(args)
    # specify different optimizer to different training stage
    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for group in policies:
        try:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
        except:
            continue

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
    cudnn.benchmark = True

    # data loading
    normalize = GroupNormalize(input_mean, input_std)
    train_dataset = TSNDataSet(
        root_path=args.root_path, list_file=args.train_list, num_segments=args.num_segments, image_tmpl=prefix,
        transform=torchvision.transforms.Compose([
            train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize]),
        dense_sample=False,
        dataset=args.dataset,
        partial_fcvid_eval=False,
        partial_ratio=0.2,
        ada_reso_skip=False,
        reso_list=224,
        random_crop=False,
        center_crop=False,
        rescale_to=224,
        policy_input_offset=0,
        save_meta=False)

    val_dataset = TSNDataSet(
        root_path=args.root_path, list_file=args.val_list, num_segments=args.num_segments, image_tmpl=prefix,
        random_shift=False,
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize]),
        dense_sample=False,
        dataset=args.dataset,
        partial_fcvid_eval=False,
        partial_ratio=0.2,
        ada_reso_skip=False,
        reso_list=224,
        random_crop=False,
        center_crop=False,
        rescale_to=224,
        policy_input_offset=0,
        save_meta=False, )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=False)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    for epoch in range(start_epoch, args.epochs + 1):
        # time.sleep(1)
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training)

        if epoch > args.epochs - 10 or (epoch % 5 == 0):
            # evaluate the model on validation set
            acc1 = validate(val_loader, model, criterion, log_training)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if args.dataset == 'minik':
                output_best = 'Best Acc@1: %.3f\n' % best_acc1
            else:
                output_best = 'Best mAP@1: %.3f\n' % best_acc1
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()
            save_checkpoint({
                'state_dict': model.state_dict(),
            }, is_best)


def validate(val_loader, model, criterion, log=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # switch to evaluate mode
    model.eval()

    all_result = []
    all_targets = []
    end = time.time()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            _b = target.shape[0]
            all_targets.append(target)
            images = images.cuda()

            if args.sample:
                index = torch.tensor(args.input_order, dtype=torch.long).cuda()
                images = images.view(_b, args.num_segments, 3, args.input_size, args.input_size)
                images = torch.index_select(images, 1, index)  # sample policy
                images = images.view(_b, args.num_segments * 3, args.input_size, args.input_size)

            target = target[:, 0].cuda()
            p1 = model(images)
            cat_logits, cat_pred, global_logits, local_logits = p1
            loss = criterion(cat_logits, target.view(_b, -1).expand(_b, args.num_segments).reshape(-1))
            all_result.append(cat_pred)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(cat_pred, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

        mAP, _ = cal_map(torch.cat(all_result, 0).cpu(), torch.cat(all_targets, 0).cpu())
        output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}\n'
                  'mAP: {mAP}'.format(top1=top1, top5=top5, loss=losses, mAP=mAP))
        print(output)
        if log is not None:
            log.write(output + '\n')
            log.flush()
        if args.dataset == 'minik':
            return top1.avg
        else:
            return mAP


def train(train_loader, model, criterion, optimizer, epoch, log=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()

    end = time.time()

    all_targets = []

    for i, (images, target) in enumerate(train_loader):
        _b = target.shape[0]
        all_targets.append(target)
        data_time.update(time.time() - end)
        images = images.cuda()
        if args.sample:
            index = torch.tensor(args.input_order, dtype=torch.long).cuda()
            images = images.view(_b, args.num_segments, 3, args.input_size, args.input_size)
            images = torch.index_select(images, 1, index)  # sample policy
            images = images.view(_b, args.num_segments * 3, args.input_size, args.input_size)
        target = target[:, 0].cuda()

        optimizer.zero_grad()
        p1, p2 = model(images)
        outputs_1, pred_1, outputs_global_1, outputs_local_1 = p1
        outputs_2, pred_2, outputs_global_2, outputs_local_2 = p2
        target_e = target.view(_b, -1).expand(_b, args.num_segments).reshape(-1)
        loss_1 = criterion(outputs_1, target_e) + criterion(outputs_global_1, target_e) + criterion(outputs_local_1,
                                                                                                    target_e)
        loss_2 = criterion(outputs_2, target_e) + criterion(outputs_global_2, target_e) + criterion(outputs_local_2,
                                                                                                    target_e)
        loss = (loss_1 + loss_2) / 2

        loss.backward()
        optimizer.step()

        # Update evaluation metrics
        acc1, acc5 = accuracy(pred_1, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            log.write(output + '\n')
            log.flush()


def input_order(num_segments):
    S = [np.floor((num_segments - 1) / 2), 0, num_segments - 1]
    q = 2
    while len(S) < num_segments:
        interval = np.floor(np.linspace(0, num_segments - 1, q + 1))
        for i in range(0, len(interval) - 1):
            a = interval[i]
            b = interval[i + 1]
            ind = np.floor((a + b) / 2)
            if not ind in S:
                S.append(ind)
        q *= 2
    S = [int(s) for s in S]
    print('Input Order:', S)
    return S


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


if __name__ == '__main__':
    main()
