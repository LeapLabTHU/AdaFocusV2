import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm

from ops.dataset import TSNDataSet
from ops.transforms import *
from ops import dataset_config
from ops.early_exit import early_exit
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

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.data_dir)
    args.num_classes = num_class

    model = AdaFocus(args)

    scale_size = model.scale_size
    crop_size = model.crop_size
    input_mean = model.input_mean
    input_std = model.input_std
    train_augmentation = model.get_augmentation(flip=True)

    model = torch.nn.DataParallel(model).cuda()
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

    if args.dataset == 'minik':
        # sample 20k in training set
        train_set_index = torch.randperm(len(train_dataset))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(train_set_index[-20000:])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers, pin_memory=False, drop_last=True, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers, pin_memory=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=False)

    print('Generate Logits on Valset')
    val_logits, val_targets = generate_logits(val_loader, model)
    print('Generate Logits on Trainset')
    train_logits, train_targets = generate_logits(train_loader, model)
    data = {
        'train_logits': train_logits.cpu(),
        'train_targets': train_targets.cpu(),
        'val_logits': val_logits.cpu(),
        'val_targets': val_targets.cpu()
    }
    early_exit(data, args)


def generate_logits(val_loader, model):
    # switch to evaluate mode
    model.eval()

    all_targets = []
    all_logits = []
    with torch.no_grad():
        for i, (images, target) in tqdm(enumerate(val_loader)):
            _b = target.shape[0]
            all_targets.append(target)
            images = images.cuda()

            if args.sample:
                index = torch.tensor(args.input_order, dtype=torch.long).cuda()
                images = images.view(_b, args.num_segments, 3, args.input_size, args.input_size)
                images = torch.index_select(images, 1, index)  # sample policy
                images = images.view(_b, args.num_segments * 3, args.input_size, args.input_size)
            p1 = model(images)
            cat_logits, cat_pred, global_logits, local_logits = p1
            all_logits.append(cat_logits.view(_b, args.num_segments, -1).cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, 0)


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


if __name__ == '__main__':
    main()
