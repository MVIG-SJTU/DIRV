import datetime
import os
import argparse
import traceback
import thop

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from efficientdet.vcoco_dataset import VCOCO_Dataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.hico_det_dataset import HICO_DET_Dataset

from backbone import EfficientDetBackbone
from tensorboardX import SummaryWriter
import numpy as np
from utils.sync_batchnorm import patch_replication_callback, SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

from efficientdet.loss import FocalLoss, Union_Loss, Instance_Loss
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='vcoco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=3, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=8, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=int, default=1,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--freeze_object_detection', type=int, default=1,
                        help='freeze the object detection branch when training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adam', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--log_interval', type=int, default=5, help='Number of steps between logging')

    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=bool, default=False, help='whether visualize the predicted boxes of trainging, '
                                                                  'the output images will be in test/')
    parser.add_argument('--accumulate_batch', type=int, default=1, help='accumulate some batches before backward')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, dataset="vcoco", debug=False):
        super().__init__()
        self.criterion_union = Union_Loss(dataset=dataset)
        self.criterion_instance = Instance_Loss(dataset=dataset)
        self.model = model
        self.dataset = dataset
        self.debug = debug

    def forward(self, imgs, anns_inst, anns_union):
        _, union_act_cls, union_sub_reg, union_obj_reg, \
        inst_act_cls, inst_obj_cls, inst_bbox_reg, anchors = self.model(imgs)

        anns_union = anns_union.cuda()
        anns_inst = anns_inst.cuda()

        union_act_cls_loss, union_sub_reg_loss, union_obj_reg_loss, union_diff_reg_loss = \
            self.criterion_union(union_act_cls, union_sub_reg, union_obj_reg, anchors, anns_union)
        inst_act_cls_loss, inst_obj_cls_loss, inst_obj_reg_loss = \
            self.criterion_instance(inst_act_cls, inst_obj_cls, inst_bbox_reg, anchors, anns_inst)

        return union_act_cls_loss, union_sub_reg_loss, union_obj_reg_loss, union_diff_reg_loss, \
               inst_act_cls_loss, inst_obj_cls_loss, inst_obj_reg_loss


def freeze_backbone(m):
    classname = m.__class__.__name__
    for ntl in ['EfficientNet', 'BiFPN']:
        if ntl in classname:
            for param in m.parameters():
                param.requires_grad = False


def freeze_object_detection(m):
    for param in m.instance_branch.object_classifier.parameters():
        param.requires_grad = False
    for param in m.instance_branch.object_regressor.parameters():
        param.requires_grad = False


def freeze_bn_backbone(m):
    for module in m.backbone_net.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, SynchronizedBatchNorm2d):
            module.eval()
    for module in m.bifpn.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, SynchronizedBatchNorm2d):
            module.eval()

def freeze_bn_object_detection(m):
    for module in m.instance_branch.object_classifier.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, SynchronizedBatchNorm2d):
            module.eval()
    for module in m.instance_branch.object_regressor.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, SynchronizedBatchNorm2d):
            module.eval()

def train(opt):
    params = Params(f'projects/{opt.project}.yml')

    if opt.project == "vcoco":
        num_obj_class = 90
        num_union_action = 25
        num_inst_action = 51
    else:
        assert opt.project == "hico-det"
        num_obj_class = 90
        num_union_action = 117
        num_inst_action = 234

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers,
                       'pin_memory': False}

    val_params = {'batch_size': opt.batch_size * 2,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers,
                  'pin_memory': False}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

    train_transform = transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                          Augmenter(), Resizer(input_sizes[opt.compound_coef])])
    val_transform = transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[opt.compound_coef])])

    if opt.project == "vcoco":
        training_set = VCOCO_Dataset(root_dir="./datasets/vcoco", set=params.train_set, color_prob=1,
                               transform=train_transform)
        val_set = VCOCO_Dataset(root_dir="./datasets/vcoco", set=params.val_set,
                          transform=val_transform)
    else:
        training_set = HICO_DET_Dataset(root_dir="datasets/hico_20160224_det", set="train", color_prob=1, transform=train_transform)
        val_set = HICO_DET_Dataset(root_dir="datasets/hico_20160224_det", set="test", transform=val_transform)

    training_generator = DataLoader(training_set, **training_params)

    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=num_obj_class, num_union_classes=num_union_action,
                                 num_inst_classes=num_inst_action, compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    model.train()
    print("num_classes:", num_obj_class)
    print("num_union_classes:", num_union_action)
    print("instance_action_list", num_inst_action)
    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
            # last_epoch = int(os.path.basename(weights_path).split('_')[-2].split('.')[0]) + 1
            # last_step = last_epoch * len(training_generator)
        except:
            last_step = 0

        try:
            init_weights(model)
            print(weights_path)
            model_dict = model.state_dict()
            pretrained_dict = torch.load(weights_path,  map_location=torch.device('cpu'))
            new_pretrained_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    new_pretrained_dict[k] = v
                elif ("instance_branch.object_"+k) in model_dict:
                    new_pretrained_dict["instance_branch.object_"+k] = v
                    # print("instance_branch.object_"+k)
            ret = model.load_state_dict(new_pretrained_dict, strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        model.apply(freeze_backbone)
        freeze_bn_backbone(model)
        print('[Info] freezed backbone')

    if opt.freeze_object_detection:
        freeze_object_detection(model)
        freeze_bn_object_detection(model)
        # model.apply(freeze_object_detection)
        print('[Info] freezed object detection branch')


    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 8:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, dataset=opt.project, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)
                if opt.head_only:
                    print('[Info] freezed SyncBN backbone')
                    freeze_bn_backbone(model.module.model)
                if opt.freeze_object_detection:
                    print('[Info] freezed SyncBN object detection')
                    freeze_bn_object_detection(model.module.model)

    if opt.optim == 'adamw':
        # optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)
    elif opt.optim == "adam":
        # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)
    else:
        # optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, min_lr = 1e-7)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)

    num_iter_per_epoch = (len(training_generator) + opt.accumulate_batch - 1) // opt.accumulate_batch

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch + 1
            if epoch < last_epoch:
                continue

            if epoch in [120, 130]:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10

            epoch_loss = []
            for iter, data in enumerate(training_generator):
                try:
                    imgs = data['img']
                    annot = data['annot']
                    # torch.cuda.empty_cache()
                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        for key in annot:
                            annot[key] = annot[key].cuda()

                    union_act_cls_loss, union_sub_reg_loss, union_obj_reg_loss, union_diff_reg_loss, \
                    inst_act_cls_loss, inst_obj_cls_loss, inst_obj_reg_loss = model(imgs, annot["instance"], annot["interaction"])


                    union_act_cls_loss = union_act_cls_loss.mean()
                    union_sub_reg_loss = union_sub_reg_loss.mean()
                    union_obj_reg_loss = union_obj_reg_loss.mean()
                    union_diff_reg_loss = union_diff_reg_loss.mean()

                    inst_act_cls_loss = inst_act_cls_loss.mean()
                    inst_obj_cls_loss = inst_obj_cls_loss.mean()
                    inst_obj_reg_loss = inst_obj_reg_loss.mean()

                    union_loss = union_act_cls_loss + union_sub_reg_loss + union_obj_reg_loss + union_diff_reg_loss
                    instance_loss = inst_act_cls_loss + inst_obj_cls_loss + inst_obj_reg_loss

                    loss = union_loss + inst_act_cls_loss

                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    batch_loss = loss / opt.accumulate_batch
                    batch_loss.backward()
                    if (iter + 1) % opt.accumulate_batch == 0 or iter == len(training_generator) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        step += 1

                    loss = loss.item()
                    union_loss = union_loss.item()
                    instance_loss = instance_loss.item()

                    epoch_loss.append(float(loss))
                    current_lr = optimizer.param_groups[0]['lr']

                    if step % opt.log_interval == 0:
                        writer.add_scalars('Union Action Classification Loss', {'train': union_act_cls_loss}, step)
                        writer.add_scalars('Union Subject Regression Loss', {'train': union_sub_reg_loss}, step)
                        writer.add_scalars('Union Object Regression Loss', {'train': union_obj_reg_loss}, step)
                        writer.add_scalars('Union Diff Regression Loss', {'train': union_diff_reg_loss}, step)

                        writer.add_scalars('Instance Action Classification Loss', {'train': inst_act_cls_loss}, step)
                        writer.add_scalars('Instance Object Classification Loss', {'train': inst_obj_cls_loss}, step)
                        writer.add_scalars('Instance Regression Loss', {'train': inst_obj_reg_loss}, step)

                        writer.add_scalars('Total Loss', {'train': loss}, step)
                        writer.add_scalars('Union Loss', {'train': union_loss}, step)
                        writer.add_scalars('Instance Loss', {'train': instance_loss}, step)

                        # log learning_rate
                        writer.add_scalar('learning_rate', current_lr, step)

                    if iter % 20 == 0:
                        print(
                            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Union loss: {:.5f}. Instance loss: {:.5f}.  '
                            ' Total loss: {:.5f}. Learning rate: {:.5f}'.format(
                                step, epoch, opt.num_epochs, (iter + 1) // opt.accumulate_batch, num_iter_per_epoch, union_loss, instance_loss, loss, current_lr))

                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

            # scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                # model.eval()

                union_loss_ls = []
                instance_loss_ls = []

                union_act_cls_loss_ls = []
                union_obj_cls_loss_ls = []
                union_act_reg_loss_ls = []

                union_sub_reg_loss_ls = []
                union_obj_reg_loss_ls = []
                union_diff_reg_loss_ls = []

                inst_act_cls_loss_ls = []
                inst_obj_cls_loss_ls = []
                inst_obj_reg_loss_ls = []

                val_loss = []
                for iter, data in enumerate(val_generator):
                    if (iter + 1) % 50 == 0:
                        print("%d/%d" %(iter+1, len(val_generator)))
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']
                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            for key in annot:
                                annot[key] = annot[key].cuda()

                        union_act_cls_loss, union_sub_reg_loss, union_obj_reg_loss, union_diff_reg_loss, \
                        inst_act_cls_loss, inst_obj_cls_loss, inst_obj_reg_loss = model(imgs, annot["instance"], annot["interaction"])

                        union_act_cls_loss = union_act_cls_loss.mean()
                        union_sub_reg_loss = union_sub_reg_loss.mean()
                        union_obj_reg_loss = union_obj_reg_loss.mean()
                        union_diff_reg_loss = union_diff_reg_loss.mean()

                        inst_act_cls_loss = inst_act_cls_loss.mean()
                        inst_obj_cls_loss = inst_obj_cls_loss.mean()
                        inst_obj_reg_loss = inst_obj_reg_loss.mean()

                        union_loss = union_act_cls_loss + union_sub_reg_loss + union_obj_reg_loss + union_diff_reg_loss
                        instance_loss = inst_act_cls_loss + inst_obj_cls_loss + inst_obj_reg_loss

                        loss = union_loss + inst_act_cls_loss

                        if loss == 0 or not torch.isfinite(loss):
                            continue
                        val_loss.append(loss.item())

                        union_act_cls_loss_ls.append(union_act_cls_loss.item())
                        union_sub_reg_loss_ls.append(union_sub_reg_loss.item())
                        union_obj_reg_loss_ls.append(union_obj_reg_loss.item())
                        union_diff_reg_loss_ls.append(union_diff_reg_loss.item())
                        # union_obj_cls_loss_ls.append(union_obj_cls_loss.item())
                        # union_act_reg_loss_ls.append(union_act_reg_loss.item())

                        inst_act_cls_loss_ls.append(inst_act_cls_loss.item())
                        inst_obj_cls_loss_ls.append(inst_obj_cls_loss.item())
                        inst_obj_reg_loss_ls.append(inst_obj_reg_loss.item())

                        union_loss_ls.append(union_loss.item())
                        instance_loss_ls.append(instance_loss.item())

                union_loss = np.mean(union_loss_ls)
                instance_loss = np.mean(instance_loss_ls)

                union_act_cls_loss = np.mean(union_act_cls_loss_ls)
                union_sub_reg_loss = np.mean(union_sub_reg_loss_ls)
                union_obj_reg_loss = np.mean(union_obj_reg_loss_ls)
                union_diff_reg_loss = np.mean(union_diff_reg_loss_ls)

                inst_act_cls_loss = np.mean(inst_act_cls_loss_ls)
                inst_obj_cls_loss = np.mean(inst_obj_cls_loss_ls)
                inst_obj_reg_loss = np.mean(inst_obj_reg_loss_ls)

                loss = union_loss + inst_act_cls_loss

                print(
                    'Val. Epoch: {}/{}. Union loss: {:1.5f}. Instance loss: {:1.5f}. '
                    'Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, union_loss, instance_loss, loss))

                writer.add_scalars('Union Action Classification Loss', {'val': union_act_cls_loss}, step)
                writer.add_scalars('Union Subject Regression Loss', {'val': union_sub_reg_loss}, step)
                writer.add_scalars('Union Object Regression Loss', {'val': union_obj_reg_loss}, step)
                writer.add_scalars('Union Diff Regression Loss', {'val': union_diff_reg_loss}, step)

                writer.add_scalars('Instance Action Classification Loss', {'val': inst_act_cls_loss}, step)
                writer.add_scalars('Instance Object Classification Loss', {'val': inst_obj_cls_loss}, step)
                writer.add_scalars('Instance Regression Loss', {'val': inst_obj_reg_loss}, step)


                writer.add_scalars('Total Loss', {'val': loss}, step)
                writer.add_scalars('Union Loss', {'val': union_loss}, step)
                writer.add_scalars('Instance Loss', {'val': instance_loss}, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')

                # model.train()

            # scheduler.step()

                scheduler.step(np.mean(val_loss))
                if optimizer.param_groups[0]['lr'] < opt.lr / 100:
                    break 
                # Early stopping
                # if epoch - best_epoch > opt.es_patience > 0:
                #     print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, loss))
                #     break
    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    train(opt)
