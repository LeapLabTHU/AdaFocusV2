import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from ops.transforms import GroupMultiScaleCrop, GroupRandomHorizontalFlip
from .resnet import resnet50
from .mobilenet_v2 import mobilenet_v2


def get_patches_frame(input_frames, actions, patch_size, image_size):
    input_frames = input_frames.view(-1, 3, image_size, image_size)  # [NT,C,H,W]
    theta = torch.zeros(input_frames.size(0), 2, 3).cuda()
    patch_coordinate = (actions * (image_size - patch_size))
    x1, x2, y1, y2 = patch_coordinate[:, 1], patch_coordinate[:, 1] + patch_size, \
                     patch_coordinate[:, 0], patch_coordinate[:, 0] + patch_size

    theta[:, 0, 0], theta[:, 1, 1] = patch_size / image_size, patch_size / image_size
    theta[:, 0, 2], theta[:, 1, 2] = -1 + (x1 + x2) / image_size, -1 + (y1 + y2) / image_size

    grid = F.affine_grid(theta.float(), torch.Size((input_frames.size(0), 3, patch_size, patch_size)))
    patches = F.grid_sample(input_frames, grid)  # [NT,C,H1,W1]
    return patches


class AdaFocus(nn.Module):
    def __init__(self, args):
        super(AdaFocus, self).__init__()
        self.num_segments = args.num_segments
        self.num_classes = args.num_classes
        if args.dataset == 'fcvid':
            assert args.num_classes == 239
        self.input_size = args.input_size
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.glance_arch = args.glance_arch
        if args.glance_arch == 'mbv2':
            print('Global CNN Backbone: mobilenet_v2')
            self.global_CNN = mobilenet_v2(pretrained=True)
            self.global_feature_dim = self.global_CNN.last_channel
            self.stn_state_dim = self.global_feature_dim * 7 * 7
        elif args.glance_arch == 'res50':
            print('Global CNN Backbone: resnet50 (glance size 96*96)')
            self.global_CNN = resnet50()
            if os.path.exists(args.glance_ckpt_path):
                state_dict = torch.load(args.glance_ckpt_path)
                self.global_CNN.load_state_dict(state_dict)
            self.global_feature_dim = self.global_CNN.fc.in_features
            self.stn_state_dim = self.global_feature_dim * 3 * 3
            self.glance_size = 96
        else:
            raise NotImplementedError

        self.global_CNN_fc = nn.Sequential(
            nn.Dropout(args.fc_dropout),
            nn.Linear(self.global_feature_dim, self.num_classes),
        )

        self.local_CNN = resnet50(pretrained=True)
        self.local_feature_dim = self.local_CNN.fc.in_features
        self.local_CNN_fc = nn.Sequential(
            nn.Dropout(args.fc_dropout),
            nn.Linear(self.local_feature_dim, self.num_classes),
        )

        self.stn_feature_dim = self.global_feature_dim
        self.policy_stn = PolicySTN(
            stn_state_dim=self.stn_state_dim,
            stn_feature_dim=self.global_feature_dim,
            num_segments=args.num_segments,
            hidden_dim=args.stn_hidden_dim,
        )
        self.cat_feature_dim = self.global_feature_dim + self.local_feature_dim
        self.classifier = PoolingClassifier(
            input_dim=self.cat_feature_dim,
            num_segments=self.num_segments,
            num_classes=args.num_classes,
            dropout=args.dropout
        )

    def forward(self, images):
        if self.glance_arch == 'res50':
            glance_images = torch.nn.functional.interpolate(images, (self.glance_size, self.glance_size),
                                                            mode='bilinear', align_corners=True)
            images = images.view(-1, 3, self.input_size, self.input_size)
            glance_images = glance_images.view(-1, 3, self.glance_size, self.glance_size)
            global_feat_map, global_feat = self.global_CNN.get_featmap(glance_images)
        else:
            images = images.view(-1, 3, self.input_size, self.input_size)
            global_feat_map, global_feat = self.global_CNN.get_featmap(images)

        global_logits = self.global_CNN_fc(global_feat)

        if self.training:
            actions_1 = self.policy_stn(global_feat_map.detach())
            actions_2 = torch.rand_like(actions_1)
            res = []
            for actions in [actions_1, actions_2]:
                patches = get_patches_frame(images, actions, self.patch_size, self.input_size)
                local_feat = self.local_CNN.get_featvec(patches)
                local_logits = self.local_CNN_fc(local_feat)
                cat_feat = torch.cat([global_feat, local_feat], dim=-1)
                cat_feat = cat_feat.view(-1, self.num_segments, self.cat_feature_dim)
                cat_logits, cat_pred = self.classifier(cat_feat)
                res.append((cat_logits, cat_pred, global_logits, local_logits))
            return res
        else:
            actions = self.policy_stn(global_feat_map)
            patches = get_patches_frame(images, actions, self.patch_size, self.input_size)
            local_feat = self.local_CNN.get_featvec(patches)
            local_logits = self.local_CNN_fc(local_feat)
            cat_feat = torch.cat([global_feat, local_feat], dim=-1)
            cat_feat = cat_feat.view(-1, self.num_segments, self.cat_feature_dim)
            cat_logits, cat_pred = self.classifier(cat_feat)
            return cat_logits, cat_pred, global_logits, local_logits

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    @property
    def crop_size(self):
        return self.input_size

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose(
                [GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]), GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])

    def get_optim_policies(self, args):
        return [{'params': self.policy_stn.parameters(), 'lr_mult': args.stn_lr_ratio, 'decay_mult': 1,
                 'name': "policy_stn"}] \
               + [{'params': self.global_CNN.parameters(), 'lr_mult': args.global_lr_ratio, 'decay_mult': 1,
                   'name': "global_CNN"}] \
               + [{'params': self.global_CNN_fc.parameters(), 'lr_mult': args.global_lr_ratio, 'decay_mult': 1,
                   'name': "global_CNN_fc"}] \
               + [{'params': self.local_CNN.parameters(), 'lr_mult': 1, 'decay_mult': 1, 'name': "local_CNN"}] \
               + [{'params': self.local_CNN_fc.parameters(), 'lr_mult': 1, 'decay_mult': 1, 'name': "local_CNN_fc"}] \
               + [{'params': self.classifier.parameters(), 'lr_mult': args.classifier_lr_ratio, 'decay_mult': 1,
                   'name': "pooling_classifier"}]


class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = torch.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1)), dim=1)
        return x.max(dim=1)[0]


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, num_neurons=4096):
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = [num_neurons]
        layers = []
        dim_input = input_dim
        for dim_output in self.num_neurons:
            layers.append(nn.Linear(dim_input, dim_output))
            layers.append(nn.BatchNorm1d(dim_output))
            layers.append(nn.ReLU())
            dim_input = dim_output
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class PoolingClassifier(nn.Module):
    def __init__(self, input_dim, num_segments, num_classes, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_pooling = MaxPooling()
        self.mlp = MultiLayerPerceptron(input_dim)
        self.num_segments = num_segments
        self.classifiers = nn.ModuleList()
        for m in range(self.num_segments):
            self.classifiers.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(4096, self.num_classes)
            ))

    def forward(self, x):
        _b = x.size(0)
        x = x.view(-1, self.input_dim)
        z = self.mlp(x).view(_b, self.num_segments, -1)
        logits = torch.zeros(_b, self.num_segments, self.num_classes).cuda()
        cur_z = z[:, 0]
        for frame_idx in range(0, self.num_segments):
            if frame_idx > 0:
                cur_z = self.max_pooling(z[:, frame_idx], cur_z)
            logits[:, frame_idx] = self.classifiers[frame_idx](cur_z)
        last_out = logits[:, -1, :].reshape(_b, -1)
        logits = logits.view(_b * self.num_segments, -1)
        return logits, last_out


class PolicySTN(nn.Module):
    def __init__(self, stn_feature_dim, stn_state_dim, hidden_dim, num_segments):
        super(PolicySTN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(stn_feature_dim, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(int(stn_state_dim * 64 / stn_feature_dim), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
        )
        self.num_segments = num_segments
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )

    def forward(self, features):
        feature = self.encoder(features)  # [NT, H]
        feature = feature.view(-1, self.num_segments, self.hidden_dim)  # [N, T, H]
        _b, _t, _f = feature.shape
        hx = torch.zeros(self.gru.num_layers, _b, self.hidden_dim).cuda()
        self.gru.flatten_parameters()
        out, _ = self.gru(feature, hx)  # [N, T, H]
        actions = self.fc(out.reshape(_b * _t, -1))
        return actions
