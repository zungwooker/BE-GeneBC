from collections import Counter
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import json
import torch.nn.functional as F

import os
import torch.optim as optim

from data.util import get_dataset, IdxDataset, MixupModule
from module.loss import GeneralizedCELoss
from module.util import get_model
from module.util import get_backbone
from util import *

import warnings
warnings.filterwarnings(action='ignore')
import copy

def load_json(json_path: str):
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            try:
                json_file = json.load(file)
            except json.JSONDecodeError:
                raise RuntimeError("An error occurred while loading the existing json file.")
    else:
        raise RuntimeError(f".json does not exist.\nPath: {json_path}")
    
    return json_file

class LearnerOurs(object):
    def __init__(self, args):
        self.args = args

        data2model = {'cmnist': args.model,
                      'bar': "ResNet18",
                      'bffhq': "ResNet18",
                      'dogs_and_cats': "ResNet18",
                      'cifar10c': "ResNet18"
                     }

        data2batch_size = {'cmnist': 256,
                           'bar': 64,
                           'bffhq': 64,
                           'dogs_and_cats': 64,
                           'cifar10c': 256,
                          }
        
        data2preprocess = {'cmnist': None,
                           'bar': True,
                           'bffhq': True,
                           'dogs_and_cats':True,
                           'cifar10c': True
                          }
        
        data2num_class = {'cmnist': 10,
                          'bar': 6,
                          'bffhq': 2,
                          'dogs_and_cats':2,
                          'cifar10c': 10
                          }

        run_name = args.exp
        if args.tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(f'result/{args.tensorboard_dir}/{run_name}')

        self.model = data2model[args.dataset]
        self.batch_size = data2batch_size[args.dataset]

        print(f'model: {self.model} || dataset: {args.dataset}')
        print(f'working with experiment: {args.exp}...')
        self.log_dir = os.makedirs(os.path.join(args.log_dir, args.dataset, args.exp), exist_ok=True)
        self.device = torch.device(''.join(['cuda:', args.gpu_num]) if args.device == 'cuda' else 'cpu')
        self.args = args

        print(self.args)

        # logging directories
        self.log_dir = os.path.join(args.log_dir, args.dataset, args.exp)
        self.summary_dir =  os.path.join(args.log_dir, args.dataset, args.tensorboard_dir, args.exp)
        self.result_dir = os.path.join(self.log_dir, "result")
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.train_dataset = get_dataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
            dataset_split="train",
            transform_split="train",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
            preproc_dir = args.preproc_dir,
            mixup=args.mixup,
            only_tags=args.only_tags
        )
        self.train_g_dataset = get_dataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
            dataset_split="train",
            transform_split="train",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
            include_generated=True,
            preproc_dir=args.preproc_dir,
            mixup=args.mixup,
            half_generated=args.half_generated,
            only_no_tags=args.only_no_tags,
            no_tags_gene=args.no_tags_gene,
            only_no_tags_balanced = args.only_no_tags_balanced
        )
        self.valid_dataset = get_dataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
            dataset_split="valid",
            transform_split="valid",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
        )
        self.test_dataset = get_dataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
            dataset_split="test",
            transform_split="valid",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
        )

        if args.mixup:
            self.mixup_module = MixupModule(dataset=self.train_dataset, num_class=data2num_class[args.dataset])

        train_target_attr = []
        for data in self.train_dataset.data:
            train_target_attr.append(int(data.split('_')[-2]))
        train_target_attr = torch.LongTensor(train_target_attr)
        
        train_g_target_attr = []
        for data in self.train_g_dataset.data:
            train_g_target_attr.append(int(data.split('_')[-2]))
        train_g_target_attr = torch.LongTensor(train_g_target_attr)

        attr_dims = []
        attr_dims.append(torch.max(train_target_attr).item() + 1)
        self.num_classes = attr_dims[0]

        self.train_dataset = IdxDataset(self.train_dataset)
        self.train_g_dataset = IdxDataset(self.train_g_dataset)

        # make loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True
        )
        
        self.train_g_loader = DataLoader(
            self.train_g_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True
        )

        # For BCDs
        self.pretrain_loader = DataLoader(
            self.train_dataset, # Original+generated dataset
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=False
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # define model and optimizer
        self.model_b = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)
        self.model_d = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)

        self.optimizer_b = torch.optim.Adam(
                self.model_b.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        self.optimizer_d = torch.optim.Adam(
                self.model_d.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        # define loss
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        print(f'self.criterion: {self.criterion}')

        self.bias_criterion = GeneralizedCELoss(q=0.7)
        print(f'self.bias_criterion: {self.bias_criterion}')
        
        self.sample_loss_ema_b = EMA(torch.LongTensor(train_g_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha, device=self.device)
        self.sample_loss_ema_d = EMA(torch.LongTensor(train_g_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha, device=self.device)
        
        print(f'alpha : {self.sample_loss_ema_d.alpha}')
        self.best_valid_acc_b, self.best_test_acc_b = 0., 0.
        self.best_valid_acc_d, self.best_test_acc_d = 0., 0.
                    
        print('finished model initialization....')
        
    def wandb_switch(self, switch):
        if self.args.wandb and switch == 'start':
            wandb.init(
                project=self.args.projcode,
                name=f'{self.args.run_name} | seed: {self.args.seed}',
                config={
                },
                settings=wandb.Settings(start_method="fork")
            )
            wandb.define_metric("training/*", step_metric="Iter step")
            
            wandb.define_metric("train/*", step_metric="Iter step")
            wandb.define_metric("valid/*", step_metric="Iter step")
            wandb.define_metric("test/*", step_metric="Iter step")
                
        elif self.args.wandb and switch == 'finish':
            wandb.finish()

    # evaluation code for vanilla
    def evaluate(self, model, data_loader):
        model.eval()
        total_correct, total_num = 0, 0
        for data, attr, index in tqdm(data_loader, leave=False):
            label = attr[:, 0]
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = total_correct/float(total_num)
        model.train()
        return accs
    
    # evaluation code for disent
    def evaluate_disent(self,model_b, model_d, data_loader, model='label'):
        model_b.eval()
        model_d.eval()

        total_correct, total_num = 0, 0

        for data, attr, index in tqdm(data_loader, leave=False):
            label = attr[:, 0]
            # label = attr
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                if self.args.dataset == 'cmnist' and self.args.model == 'MLP':
                    z_l = model_d.extract(data)
                    z_b = model_b.extract(data)
                else:
                    z_l, z_b = [], []
                    hook_fn = self.model_d.avgpool.register_forward_hook(self.concat_dummy(z_l))
                    _ = self.model_d(data)
                    hook_fn.remove()
                    z_l = z_l[0]
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                z_origin = torch.cat((z_l, z_b), dim=1)
                if model == 'bias':
                    pred_label = model_b.fc(z_origin)
                else:
                    pred_label = model_d.fc(z_origin)
                pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = total_correct/float(total_num)
        model_b.train()
        model_d.train()
        return accs

    def save_best(self, step):
        model_path = os.path.join(self.result_dir, "best_model_d.th")
        state_dict = {
            'steps': step,
            'state_dict': self.model_d.state_dict(),
            'optimizer': self.optimizer_d.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        model_path = os.path.join(self.result_dir, "best_model_b.th")
        state_dict = {
            'steps': step,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        print(f'{step} model saved ...')

    def board_lff_loss(self, step, loss_b, loss_d):
        if self.args.tensorboard:
            self.writer.add_scalar(f"loss/loss_b_train", loss_b, step)
            self.writer.add_scalar(f"loss/loss_d_train", loss_d, step)
            
        if self.args.wandb:
            wandb.log({
                    "Iter step": step,
                    "train/loss_b_train": loss_b,
                    "train/loss_d_train": loss_d,
                })

    def board_disent_loss(self, step, loss_dis_conflict, loss_dis_align, loss_swap_conflict, loss_swap_align, lambda_swap):
        if self.args.tensorboard:
            self.writer.add_scalar(f"loss/loss_dis_conflict",  loss_dis_conflict, step)
            self.writer.add_scalar(f"loss/loss_dis_align",     loss_dis_align, step)
            self.writer.add_scalar(f"loss/loss_swap_conflict", loss_swap_conflict, step)
            self.writer.add_scalar(f"loss/loss_swap_align",    loss_swap_align, step)
            self.writer.add_scalar(f"loss/loss",               (loss_dis_conflict + loss_dis_align) + lambda_swap * (loss_swap_conflict + loss_swap_align), step)

    def board_lff_wx_conflict(self, step):
        loss_weight_total = None

        # conflict index
        conflict_index = torch.where(self.conflicting_index.squeeze(0) == 1)[0]
        label = self.label_index[conflict_index]

        # class-wise normalize
        loss_b = self.sample_loss_ema_b.parameter[conflict_index].clone().detach()
        loss_d = self.sample_loss_ema_d.parameter[conflict_index].clone().detach()

        if np.isnan(loss_b.mean().item()):
            raise NameError('loss_b_ema')
        if np.isnan(loss_d.mean().item()):
            raise NameError('loss_d_ema')

        label_cpu = label.cpu()

        for c in range(self.num_classes):
            class_index = np.where(label_cpu == c)[0]
            max_loss_b = self.sample_loss_ema_b.max_loss(c) + 1e-8
            max_loss_d = self.sample_loss_ema_d.max_loss(c)
            loss_b[class_index] /= max_loss_b
            loss_d[class_index] /= max_loss_d

        loss_weight = loss_b / (loss_b + loss_d + 1e-8)

        if loss_weight_total is None:
            loss_weight_total = loss_weight
        else:
            loss_weight_total = torch.cat((loss_weight_total, loss_weight), dim=0)

        loss_weight_total = loss_weight_total.mean()

        log_dict = {
                    "w(x)_mean/conflict_only": loss_weight_total
                }

        if self.args.tensorboard:
            for key, value in log_dict.items():
                self.writer.add_scalar(key, value, step)

    def board_lff_wx(self, step, loss_weight, ac_flag, aw_flag, cc_flag, cw_flag):
        log_dict = {
                    "w(x)_mean/align": loss_weight[aw_flag | ac_flag].mean(),
                    "w(x)_mean/conflict": loss_weight[cw_flag | cc_flag].mean(),
                }
        if self.args.tensorboard:
            for key, value in log_dict.items():
                self.writer.add_scalar(key, value, step)

    def board_lff_acc(self, step, inference=None):
        # check label network
        valid_accs_b = self.evaluate(self.model_b, self.valid_loader)
        test_accs_b = self.evaluate(self.model_b, self.test_loader)

        valid_accs_d = self.evaluate(self.model_d, self.valid_loader)
        test_accs_d = self.evaluate(self.model_d, self.test_loader)

        if inference:
            print(f'test acc: {test_accs_d.item()}')
            import sys
            sys.exit(0)

        if valid_accs_b >= self.best_valid_acc_b:
            self.best_valid_acc_b = valid_accs_b

        if test_accs_b >= self.best_test_acc_b:
            self.best_test_acc_b = test_accs_b

        if valid_accs_d >= self.best_valid_acc_d:
            self.best_valid_acc_d = valid_accs_d

        if test_accs_d >= self.best_test_acc_d:
            self.best_test_acc_d = test_accs_d
            self.save_best(step)

        if self.args.tensorboard:
            self.writer.add_scalar(f"acc/acc_b_valid", valid_accs_b, step)
            self.writer.add_scalar(f"acc/acc_b_test", test_accs_b, step)
            self.writer.add_scalar(f"acc/best_acc_b_valid", self.best_valid_acc_b, step)
            self.writer.add_scalar(f"acc/best_acc_b_test", self.best_test_acc_b, step)
            self.writer.add_scalar(f"acc/acc_d_valid", valid_accs_d, step)
            self.writer.add_scalar(f"acc/acc_d_test", test_accs_d, step)
            self.writer.add_scalar(f"acc/best_acc_d_valid", self.best_valid_acc_d, step)
            self.writer.add_scalar(f"acc/best_acc_d_test", self.best_test_acc_d, step)

        print(f'valid_b: {valid_accs_b} || test_b: {test_accs_b} ')
        print(f'valid_d: {valid_accs_d} || test_d: {test_accs_d} ')
        
        if self.args.wandb:
            wandb.log({
                    "Iter step": step,
                    "valid/accs_b": valid_accs_b,
                    "valid/accs_d": valid_accs_d,
                    "test/accs_b": test_accs_b,
                    "test/accs_d": test_accs_d,
                    "valid/best_acc_b": self.best_valid_acc_b,
                    "valid/best_acc_d": self.best_valid_acc_d,
                    "test/best_acc_b": self.best_test_acc_b,
                    "test/best_acc_d": self.best_test_acc_d,
                })
            
    def board_vanilla_acc(self, step, inference=None):
        # check label network
        valid_accs_d = self.evaluate(self.model_d, self.valid_loader)
        test_accs_d = self.evaluate(self.model_d, self.test_loader)

        if inference:
            print(f'test acc: {test_accs_d.item()}')
            import sys
            sys.exit(0)

        if valid_accs_d >= self.best_valid_acc_d:
            self.best_valid_acc_d = valid_accs_d

        if test_accs_d >= self.best_test_acc_d:
            self.best_test_acc_d = test_accs_d
            self.save_best(step)

        if self.args.tensorboard:
            self.writer.add_scalar(f"acc/best_acc_b_valid", self.best_valid_acc_b, step)
            self.writer.add_scalar(f"acc/best_acc_b_test", self.best_test_acc_b, step)
            self.writer.add_scalar(f"acc/acc_d_valid", valid_accs_d, step)
            self.writer.add_scalar(f"acc/acc_d_test", test_accs_d, step)
            self.writer.add_scalar(f"acc/best_acc_d_valid", self.best_valid_acc_d, step)
            self.writer.add_scalar(f"acc/best_acc_d_test", self.best_test_acc_d, step)

        print(f'valid_d: {valid_accs_d} || test_d: {test_accs_d} ')
        
        if self.args.wandb:
            wandb.log({
                    "Iter step": step,
                    "valid/accs_d": valid_accs_d,
                    "test/accs_d": test_accs_d,
                    "valid/best_acc_b": self.best_valid_acc_b,
                    "valid/best_acc_d": self.best_valid_acc_d,
                    "test/best_acc_b": self.best_test_acc_b,
                    "test/best_acc_d": self.best_test_acc_d,
                })

    def board_pretrain_best_acc(self, i, model_b, best_valid_acc_b, step):
        # check label network
        valid_accs_b = self.evaluate(model_b, self.valid_loader)

        print(f'best: {best_valid_acc_b}, curr: {valid_accs_b}')

        if valid_accs_b > best_valid_acc_b:
            best_valid_acc_b = valid_accs_b

            ######### copy parameters #########
            self.best_model_b = copy.deepcopy(model_b)
            print(f'early model {i}th saved...')

        log_dict = {
            f"{i}_pretrain_best_valid_acc": best_valid_acc_b,
        }

        if self.args.tensorboard:
            for key, value in log_dict.items():
                self.writer.add_scalar(key, value, step)

        return best_valid_acc_b
 
    def board_disent_acc(self, step, inference=None):
        # check label network
        valid_accs_d = self.evaluate_disent(self.model_b, self.model_d, self.valid_loader, model='label')
        test_accs_d = self.evaluate_disent(self.model_b, self.model_d, self.test_loader, model='label')
        
        valid_accs_b = self.evaluate_disent(self.model_b, self.model_d, self.valid_loader, model='bias')
        test_accs_b = self.evaluate_disent(self.model_b, self.model_d, self.test_loader, model='bias')
        
        if inference:
            print(f'test acc: {test_accs_d.item()}')
            import sys
            sys.exit(0)

        if valid_accs_b >= self.best_valid_acc_b:
            self.best_valid_acc_b = valid_accs_b

        if test_accs_b >= self.best_test_acc_b:
            self.best_test_acc_b = test_accs_b

        if valid_accs_d > self.best_valid_acc_d:
            self.best_valid_acc_d = valid_accs_d

        if test_accs_d >= self.best_test_acc_d:
            self.best_test_acc_d = test_accs_d
            self.save_best(step)

        if self.args.tensorboard:
            self.writer.add_scalar(f"acc/acc_b_valid", valid_accs_b, step)
            self.writer.add_scalar(f"acc/acc_b_test", test_accs_b, step)
            self.writer.add_scalar(f"acc/acc_d_valid", valid_accs_d, step)
            self.writer.add_scalar(f"acc/acc_d_test", test_accs_d, step)
            self.writer.add_scalar(f"acc/best_acc_b_valid", self.best_valid_acc_b, step)
            self.writer.add_scalar(f"acc/best_acc_b_test", self.best_test_acc_b, step)
            self.writer.add_scalar(f"acc/best_acc_d_valid", self.best_valid_acc_d, step)
            self.writer.add_scalar(f"acc/best_acc_d_test", self.best_test_acc_d, step)

        print(f'valid_b: {valid_accs_b} || test_b: {test_accs_b} ')
        print(f'valid_d: {valid_accs_d} || test_d: {test_accs_d} ')
        
        if self.args.wandb:
            wandb.log({
                    "Iter step": step,
                    "valid/accs_b": valid_accs_b,
                    "valid/accs_d": valid_accs_d,
                    "test/accs_b": test_accs_b,
                    "test/accs_d": test_accs_d,
                    "valid/best_acc_b": self.best_valid_acc_b,
                    "valid/best_acc_d": self.best_valid_acc_d,
                    "test/best_acc_b": self.best_test_acc_b,
                    "test/best_acc_d": self.best_test_acc_d,
                })

    def concat_dummy(self, z):
        def hook(model, input, output):
            z.append(output.squeeze())
            return torch.cat((output, torch.zeros_like(output)), dim=1)
        return hook

    def pretrain_b_ensemble_best(self, args):
        train_iter = iter(self.train_loader)
        train_num = len(self.train_dataset.dataset)
            
        epoch, cnt = 0, 0
        index_dict, label_dict, gt_prob_dict = {}, {}, {}

        for i in range(self.args.num_bias_models):
            best_valid_acc_b = 0
            print(f'{i}th model working ...')
            del self.model_b
            self.best_model_b = None
            self.model_b = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained, first_stage=True).to(self.device)
            self.optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            for step in tqdm(range(self.args.biased_model_train_iter)):
                try:
                    index, data, attr, _ = next(train_iter)
                except:
                    train_iter = iter(self.train_loader)
                    index, data, attr, _ = next(train_iter)

                data = data.to(self.device)
                attr = attr.to(self.device)
                label = attr[:, args.target_attr_idx]

                logit_b = self.model_b(data)
                loss_b_update = self.bias_criterion(logit_b, label)
                loss = loss_b_update.mean()

                self.optimizer_b.zero_grad()
                loss.backward()
                self.optimizer_b.step()

                cnt += len(index)
                if cnt >= train_num:
                    print(f'finished epoch: {epoch}')
                    epoch += 1
                    cnt = len(index)

                if step % args.valid_freq == 0:
                    best_valid_acc_b = self.board_pretrain_best_acc(i, self.model_b, best_valid_acc_b, step)

            label_list, bias_list, pred_list, index_list, gt_prob_list, align_flag_list = [], [], [], [], [], []
            self.best_model_b.eval()
            
            for index, data, attr, _ in self.pretrain_loader:
                index = index.to(self.device)
                data = data.to(self.device)
                attr = attr.to(self.device)
                label = attr[:, args.target_attr_idx]
                bias_label = attr[:, args.bias_attr_idx]

                logit_b = self.best_model_b(data)
                prob = torch.softmax(logit_b, dim=-1)
                gt_prob = torch.gather(prob, index=label.unsqueeze(1), dim=1).squeeze(1)

                label_list += label.tolist()
                index_list += index.tolist()
                gt_prob_list += gt_prob.tolist()
                align_flag_list += (label == bias_label).tolist()

            index_list = torch.tensor(index_list)
            label_list = torch.tensor(label_list)
            gt_prob_list = torch.tensor(gt_prob_list)
            align_flag_list = torch.tensor(align_flag_list)

            align_mask = ((gt_prob_list > args.biased_model_softmax_threshold) & (align_flag_list == True)).long()
            conflict_mask = ((gt_prob_list > args.biased_model_softmax_threshold) & (align_flag_list == False)).long()
            mask = (gt_prob_list > args.biased_model_softmax_threshold).long()

            exceed_align = index_list[align_mask.nonzero().squeeze(1)]
            exceed_conflict = index_list[conflict_mask.nonzero().squeeze(1)]
            exceed_mask = index_list[mask.nonzero().squeeze(1)]

            model_index = i
            index_dict[f'{model_index}_exceed_align'] = exceed_align
            index_dict[f'{model_index}_exceed_conflict'] = exceed_conflict
            index_dict[f'{model_index}_exceed_mask'] = exceed_mask
            label_dict[model_index] = label_list
            gt_prob_dict[model_index] = gt_prob_list

            log_dict = {
                f"{model_index}_exceed_align": len(exceed_align),
                f"{model_index}_exceed_conflict": len(exceed_conflict),
                f"{model_index}_exceed_mask": len(exceed_mask),
            }
            if args.tensorboard:
                for key, value in log_dict.items():
                    self.writer.add_scalar(key, value, step)

        exceed_mask = [(gt_prob_dict[i] > args.biased_model_softmax_threshold).long() for i in
                        range(self.args.num_bias_models)]
        exceed_mask_align = [
            ((gt_prob_dict[i] > args.biased_model_softmax_threshold) & (align_flag_list == True)).long() for i in
            range(self.args.num_bias_models)]
        exceed_mask_conflict = [
            ((gt_prob_dict[i] > args.biased_model_softmax_threshold) & (align_flag_list == False)).long() for i in
            range(self.args.num_bias_models)]

        mask_sum = torch.stack(exceed_mask).sum(dim=0)
        mask_sum_align = torch.stack(exceed_mask_align).sum(dim=0)
        mask_sum_conflict = torch.stack(exceed_mask_conflict).sum(dim=0)

        total_exceed_mask = index_list[(mask_sum >= self.args.agreement).long().nonzero().squeeze(1)]
        total_exceed_align = index_list[(mask_sum_align >= self.args.agreement).long().nonzero().squeeze(1)]
        total_exceed_conflict = index_list[(mask_sum_conflict >= self.args.agreement).long().nonzero().squeeze(1)]

        print(f'exceed mask length: {total_exceed_mask.size(0)}')
        curr_index_label = torch.index_select(label_dict[0].unsqueeze(1).to(self.device), 0,
                                              torch.tensor(total_exceed_mask).long().to(self.device))
        curr_align_index_label = torch.index_select(label_dict[0].unsqueeze(1).to(self.device), 0,
                                                    torch.tensor(total_exceed_align).long().to(self.device))
        curr_conflict_index_label = torch.index_select(label_dict[0].unsqueeze(1).to(self.device), 0,
                                                       torch.tensor(total_exceed_conflict).long().to(self.device))
        log_dict = {
            f"total_exceed_align": len(total_exceed_align),
            f"total_exceed_conflict": len(total_exceed_conflict),
            f"total_exceed_mask": len(total_exceed_mask),
        }

        total_exceed_mask = torch.tensor(total_exceed_mask)

        for key, value in log_dict.items():
            print(f"* {key}: {value}")
        print(f"* EXCEED DATA COUNT: {Counter(curr_index_label.squeeze(1).tolist())}")
        print(f"* EXCEED DATA (ALIGN) COUNT: {Counter(curr_align_index_label.squeeze(1).tolist())}")
        print(f"* EXCEED DATA (CONFLICT) COUNT: {Counter(curr_conflict_index_label.squeeze(1).tolist())}")
        
        if args.tensorboard:
            for key, value in log_dict.items():
                self.writer.add_scalar(key, value, step)
        return total_exceed_mask

    # Done.
    def train_vanilla(self, args): 
        print("training vanilla ours...")
        train_iter = iter(self.train_g_loader) # Use all images(origin & generated)

        for step in tqdm(range(args.num_steps)):
            try:
                index, data, attr, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, _ = next(train_iter)

            data = data.to(self.device)
            attr = attr.to(self.device)
            label = attr[:, args.target_attr_idx]

            logit_d = self.model_d(data)
            loss_d_update = self.criterion(logit_d, label)
            loss = loss_d_update.mean()

            self.optimizer_d.zero_grad()
            loss.backward()
            self.optimizer_d.step()

            if step % args.valid_freq == 0:
                self.board_vanilla_acc(step)
 
    # Implementing..
    def train_vanilla_mixup(self, args, origin_only=False):
        print("training vanilla ours mixup...")
        # MIXUP
        if origin_only:
            train_iter = iter(self.train_loader)
        else:
            train_iter = iter(self.train_g_loader)

        for step in tqdm(range(args.num_steps)):
            try:
                index, data, attr, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, _ = next(train_iter)

            lam = np.random.beta(2, 2)
            p = np.random.rand()

            mixed_x_list = []
            mixed_y_list = []

            for i in range(data.size(0)):
                mixed_x, mixed_y = self.mixup_module.mixup(index=index[i], p=p, lam=lam)
                mixed_x_list.append(mixed_x.unsqueeze(0))
                mixed_y_list.append(mixed_y.unsqueeze(0))

            data = torch.cat(mixed_x_list, dim=0).to(self.device)
            label_soft = torch.cat(mixed_y_list, dim=0).to(self.device)

            logit_d = self.model_d(data)

            log_probs = F.log_softmax(logit_d, dim=-1)
            per_sample_losses = -log_probs * label_soft
            loss = per_sample_losses.sum(dim=-1).mean()

            self.optimizer_d.zero_grad()
            loss.backward()
            self.optimizer_d.step()
            
            if step % args.valid_freq == 0:
                self.board_vanilla_acc(step)

    # Done.
    def train_lff(self, args):
        print('Training LfF ours...')

        num_updated = 0
        train_iter = iter(self.train_loader) # Original dataset
        train_num = len(self.train_dataset.dataset) # num of Original dataset
        train_g_iter = iter(self.train_g_loader) # Original + Generated dataset

        del self.model_b
        self.model_b = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)
        
        self.optimizer_b = torch.optim.Adam(
                self.model_b.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        
        if args.use_lr_decay:
            self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_step,gamma=args.lr_gamma)
            self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_d, step_size=args.lr_decay_step,gamma=args.lr_gamma)

        for step in tqdm(range(args.num_steps)):
            # train main model
            try:
                index, data, attr, _ = next(train_g_iter)
            except:
                train_g_iter = iter(self.train_g_loader)
                index, data, attr, _ = next(train_g_iter)
                
            data = data.to(self.device)
            attr = attr.to(self.device)
            index = index.to(self.device)
            label = attr[:, args.target_attr_idx]

            with torch.no_grad():
                logit_b = self.model_b(data)
            logit_d = self.model_d(data)

            loss_b = self.criterion(logit_b, label).cpu().detach()
            loss_d = self.criterion(logit_d, label).cpu().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d')

            # EMA sample loss
            self.sample_loss_ema_b.update(loss_b, index)
            self.sample_loss_ema_d.update(loss_d, index)

            # class-wise normalize
            loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
            loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b_ema')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d_ema')

            label_cpu = label.cpu()

            for c in range(self.num_classes):
                class_index = np.where(label_cpu == c)[0]
                max_loss_b = self.sample_loss_ema_b.max_loss(c) + 1e-8
                max_loss_d = self.sample_loss_ema_d.max_loss(c)
                loss_b[class_index] /= max_loss_b
                loss_d[class_index] /= max_loss_d

            # re-weighting based on loss value / generalized CE for biased model
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)

            if np.isnan(loss_weight.mean().item()):
                raise NameError('loss_weight')
            
            # Biased model(model_b) is trained on original dataset.
            try:
                index_b, data_b, attr_b, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index_b, data_b, attr_b, _ = next(train_iter)
                
            data_b = data_b.to(self.device)
            attr_b = attr_b.to(self.device)
            index_b = index_b.to(self.device)
            label_b = attr_b[:, args.target_attr_idx]
                
            logit_b_update = self.model_b(data_b)
            
            loss_b_update = self.bias_criterion(logit_b_update, label_b)
            loss_d_update = self.criterion(logit_d, label) * loss_weight.to(self.device)

            if np.isnan(loss_b_update.mean().item()):
                raise NameError('loss_b_update')

            if np.isnan(loss_d_update.mean().item()):
                raise NameError('loss_d_update')

            loss = loss_b_update.mean() + loss_d_update.mean()
            num_updated += loss_weight.mean().item() * data.size(0)

            self.optimizer_b.zero_grad()
            self.optimizer_d.zero_grad()
            loss.backward()
            self.optimizer_b.step()
            self.optimizer_d.step()
            
            if args.use_lr_decay:
                self.scheduler_b.step()
                self.scheduler_l.step()

            if args.use_lr_decay and step % args.lr_decay_step == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer_b lr: {self.optimizer_b.param_groups[-1]['lr']}")
                print(f"self.optimizer_d lr: {self.optimizer_d.param_groups[-1]['lr']}")
                    
            if step % args.valid_freq == 0:
                self.board_lff_acc(step)

                if args.use_lr_decay and args.tensorboard:
                    self.writer.add_scalar(f"loss/learning rate", self.optimizer_d.param_groups[-1]['lr'], step)

    def train_lff_be_ours_all(self, args):
        print('Training LfF with BiasEnsemble ...')

        num_updated = 0
        train_iter = iter(self.train_g_loader)
        train_num = len(self.train_g_dataset.dataset)

        mask_index = torch.zeros(train_num, 1)
        self.conflicting_index = torch.zeros(train_num, 1)
        self.label_index = torch.zeros(train_num).long().to(self.device)

        epoch, cnt = 0, 0

        #### BiasEnsemble ####
        pseudo_align_flag = self.pretrain_b_ensemble_best(args)
        mask_index[pseudo_align_flag] = 1

        del self.model_b
        self.model_b = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)

        self.optimizer_b = torch.optim.Adam(
                self.model_b.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        if args.use_lr_decay:
            self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_step,gamma=args.lr_gamma)
            self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_d, step_size=args.lr_decay_step,gamma=args.lr_gamma)

        for step in tqdm(range(args.num_steps)):
            # train main model
            try:
                index, data, attr, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, _ = next(train_iter)

            data = data.to(self.device)
            attr = attr.to(self.device)
            index = index.to(self.device)
            label = attr[:, args.target_attr_idx]
            bias_label = attr[:, args.bias_attr_idx]

            flag_conflict = (label != bias_label)
            flag_conflict_index = index[flag_conflict]
            self.conflicting_index[flag_conflict_index] = 1
            self.label_index[index] = label

            logit_b = self.model_b(data)
            logit_d = self.model_d(data)

            loss_b = self.criterion(logit_b, label).cpu().detach()
            loss_d = self.criterion(logit_d, label).cpu().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d')

            # EMA sample loss
            self.sample_loss_ema_b.update(loss_b, index)
            self.sample_loss_ema_d.update(loss_d, index)

            # class-wise normalize
            loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
            loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b_ema')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d_ema')

            label_cpu = label.cpu()

            for c in range(self.num_classes):
                class_index = np.where(label_cpu == c)[0]
                max_loss_b = self.sample_loss_ema_b.max_loss(c) + 1e-8
                max_loss_d = self.sample_loss_ema_d.max_loss(c)
                loss_b[class_index] /= max_loss_b
                loss_d[class_index] /= max_loss_d

            # re-weighting based on loss value / generalized CE for biased model
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)
            pred = logit_d.data.max(1, keepdim=True)[1].squeeze(1)


            if np.isnan(loss_weight.mean().item()):
                raise NameError('loss_weight')

            curr_align_flag = torch.index_select(mask_index.to(self.device), 0, index)
            curr_align_flag = (curr_align_flag.squeeze(1) == 1)

            loss_b_update = self.criterion(logit_b[curr_align_flag], label[curr_align_flag])
            loss_d_update = self.criterion(logit_d, label) * loss_weight.to(self.device)

            if np.isnan(loss_b_update.mean().item()):
                raise NameError('loss_b_update')

            if np.isnan(loss_d_update.mean().item()):
                raise NameError('loss_d_update')

            loss = loss_b_update.mean() + loss_d_update.mean()
            num_updated += loss_weight.mean().item() * data.size(0)

            self.optimizer_b.zero_grad()
            self.optimizer_d.zero_grad()
            loss.backward()
            self.optimizer_b.step()
            self.optimizer_d.step()

            if args.use_lr_decay:
                self.scheduler_b.step()
                self.scheduler_l.step()

            if args.use_lr_decay and step % args.lr_decay_step == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer_b lr: {self.optimizer_b.param_groups[-1]['lr']}")
                print(f"self.optimizer_d lr: {self.optimizer_d.param_groups[-1]['lr']}")

            if step % args.valid_freq == 0:
                self.board_lff_acc(step)

                if args.use_lr_decay and args.tensorboard:
                    self.writer.add_scalar(f"loss/learning rate", self.optimizer_d.param_groups[-1]['lr'], step)
    
    # Done.             
    def train_lff_be(self, args):
        print('Training LfF with BiasEnsemble ours...')

        num_updated = 0
        train_iter = iter(self.train_loader) # Original dataset
        train_g_iter = iter(self.train_g_loader) # Original + Generated dataset
        train_num = len(self.train_dataset.dataset) # num of Original + Generated dataset

        # Mask for original dataset for BCDs.
        mask_index = torch.zeros(train_num, 1)
        self.conflicting_index = torch.zeros(train_num, 1)
        self.label_index = torch.zeros(train_num).long().to(self.device)
        
        #### BiasEnsemble ####
        # BCDs learns original dataset.
        pseudo_align_flag = self.pretrain_b_ensemble_best(args)
        mask_index[pseudo_align_flag] = 1
        
        del self.model_b
        self.model_b = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)
        
        self.optimizer_b = torch.optim.Adam(
                self.model_b.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        
        if args.use_lr_decay:
            self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_step,gamma=args.lr_gamma)
            self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_d, step_size=args.lr_decay_step,gamma=args.lr_gamma)

        for step in tqdm(range(args.num_steps)):
            # train main model
            try:
                index, data, attr, _ = next(train_g_iter)
            except:
                train_g_iter = iter(self.train_g_loader)
                index, data, attr, _ = next(train_g_iter)
                
            data = data.to(self.device)
            attr = attr.to(self.device)
            index = index.to(self.device)
            label = attr[:, args.target_attr_idx]

            with torch.no_grad():
                logit_b = self.model_b(data)
            logit_d = self.model_d(data)

            loss_b = self.criterion(logit_b, label).cpu().detach()
            loss_d = self.criterion(logit_d, label).cpu().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d')

            # EMA sample loss
            self.sample_loss_ema_b.update(loss_b, index)
            self.sample_loss_ema_d.update(loss_d, index)

            # class-wise normalize
            loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
            loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b_ema')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d_ema')

            label_cpu = label.cpu()

            for c in range(self.num_classes):
                class_index = np.where(label_cpu == c)[0]
                max_loss_b = self.sample_loss_ema_b.max_loss(c) + 1e-8
                max_loss_d = self.sample_loss_ema_d.max_loss(c)
                loss_b[class_index] /= max_loss_b
                loss_d[class_index] /= max_loss_d

            # re-weighting based on loss value / generalized CE for biased model
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)
            pred = logit_d.data.max(1, keepdim=True)[1].squeeze(1)

            if np.isnan(loss_weight.mean().item()):
                raise NameError('loss_weight')
            
            # Biased model(model_b) is trained on original dataset.
            try:
                index_b, data_b, attr_b, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index_b, data_b, attr_b, _ = next(train_iter)
                
            data_b = data_b.to(self.device)
            attr_b = attr_b.to(self.device)
            index_b = index_b.to(self.device)
            label_b = attr_b[:, args.target_attr_idx]
                
            logit_b_update = self.model_b(data_b)

            curr_align_flag = torch.index_select(mask_index.to(self.device), 0, index_b)
            curr_align_flag = (curr_align_flag.squeeze(1) == 1)
            
            loss_b_update = self.criterion(logit_b_update[curr_align_flag], label_b[curr_align_flag])
            loss_d_update = self.criterion(logit_d, label) * loss_weight.to(self.device)

            if np.isnan(loss_b_update.mean().item()):
                raise NameError('loss_b_update')

            if np.isnan(loss_d_update.mean().item()):
                raise NameError('loss_d_update')

            loss = loss_b_update.mean() + loss_d_update.mean()
            num_updated += loss_weight.mean().item() * data.size(0)

            self.optimizer_b.zero_grad()
            self.optimizer_d.zero_grad()
            loss.backward()
            self.optimizer_b.step()
            self.optimizer_d.step()
            
            if args.use_lr_decay:
                self.scheduler_b.step()
                self.scheduler_l.step()

            if args.use_lr_decay and step % args.lr_decay_step == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer_b lr: {self.optimizer_b.param_groups[-1]['lr']}")
                print(f"self.optimizer_d lr: {self.optimizer_d.param_groups[-1]['lr']}")
                    
            if step % args.valid_freq == 0:
                self.board_lff_acc(step)

                if args.use_lr_decay and args.tensorboard:
                    self.writer.add_scalar(f"loss/learning rate", self.optimizer_d.param_groups[-1]['lr'], step)

    def train_lff_ours_mixup(self, args):
        print('Training LfF ...')

        num_updated = 0
        train_iter = iter(self.train_loader) # Original dataset
        train_num = len(self.train_dataset.dataset) # num of Original dataset
        
        del self.model_b
        self.model_b = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)
        
        self.optimizer_b = torch.optim.Adam(
                self.model_b.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        
        if args.use_lr_decay:
            self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_step,gamma=args.lr_gamma)
            self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_d, step_size=args.lr_decay_step,gamma=args.lr_gamma)

        for step in tqdm(range(args.num_steps)):
            # train main model
            try:
                index, data, attr, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, _ = next(train_iter)

            lam = np.random.beta(2, 2)
            p = np.random.rand()

            mixed_x_list = []
            mixed_y_list = []

            for i in range(data.size(0)):
                mixed_x, mixed_y = self.mixup_module.mixup(index=index[i], p=p, lam=lam)
                mixed_x_list.append(mixed_x.unsqueeze(0))
                mixed_y_list.append(mixed_y.unsqueeze(0))

            data = torch.cat(mixed_x_list, dim=0).to(self.device)
            label_soft = torch.cat(mixed_y_list, dim=0).to(self.device)
                
            data = data.to(self.device)
            attr = attr.to(self.device)
            index = index.to(self.device)
            label = attr[:, args.target_attr_idx]

            with torch.no_grad():
                logit_b = self.model_b(data)
            logit_d = self.model_d(data)

            log_probs_b = F.log_softmax(logit_b, dim=-1)
            per_sample_losses_b = -log_probs_b * label_soft
            loss_b = per_sample_losses_b.sum(-1).cpu().detach()

            log_probs_d = F.log_softmax(logit_d, dim=-1)
            per_sample_losses_d = -log_probs_d * label_soft
            loss_d = per_sample_losses_d.sum(-1).cpu().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d')

            if args.ema:
                # EMA sample loss
                self.sample_loss_ema_b.update(loss_b, index)
                self.sample_loss_ema_d.update(loss_d, index)

                # class-wise normalize
                loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
                loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()

                if np.isnan(loss_b.mean().item()):
                    raise NameError('loss_b_ema')
                if np.isnan(loss_d.mean().item()):
                    raise NameError('loss_d_ema')

                label_cpu = label.cpu()

                for c in range(self.num_classes):
                    class_index = np.where(label_cpu == c)[0]
                    max_loss_b = self.sample_loss_ema_b.max_loss(c) + 1e-8
                    max_loss_d = self.sample_loss_ema_d.max_loss(c)
                    loss_b[class_index] /= max_loss_b
                    loss_d[class_index] /= max_loss_d

            # re-weighting based on loss value / generalized CE for biased model
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)

            if np.isnan(loss_weight.mean().item()):
                raise NameError('loss_weight')
            
            # Biased model(model_b) is trained on original dataset.
            try:
                index_b, data_b, attr_b, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index_b, data_b, attr_b, _ = next(train_iter)
                
            data_b = data_b.to(self.device)
            attr_b = attr_b.to(self.device)
            index_b = index_b.to(self.device)
            label_b = attr_b[:, args.target_attr_idx]
                
            logit_b_update = self.model_b(data_b)
            
            loss_b_update = self.bias_criterion(logit_b_update, label_b)

            log_probs_d = F.log_softmax(logit_d, dim=-1)
            per_sample_losses_d = -log_probs_d * label_soft
            loss_d_update = per_sample_losses_d.sum(-1) * loss_weight.to(self.device)

            if np.isnan(loss_b_update.mean().item()):
                raise NameError('loss_b_update')

            if np.isnan(loss_d_update.mean().item()):
                raise NameError('loss_d_update')

            loss = loss_b_update.mean() + loss_d_update.mean()
            num_updated += loss_weight.mean().item() * data.size(0)

            self.optimizer_b.zero_grad()
            self.optimizer_d.zero_grad()
            loss.backward()
            self.optimizer_b.step()
            self.optimizer_d.step()
            
            if args.use_lr_decay:
                self.scheduler_b.step()
                self.scheduler_l.step()

            if args.use_lr_decay and step % args.lr_decay_step == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer_b lr: {self.optimizer_b.param_groups[-1]['lr']}")
                print(f"self.optimizer_d lr: {self.optimizer_d.param_groups[-1]['lr']}")
                    
            if step % args.valid_freq == 0:
                self.board_lff_acc(step)

                if args.use_lr_decay and args.tensorboard:
                    self.writer.add_scalar(f"loss/learning rate", self.optimizer_d.param_groups[-1]['lr'], step)

    def train_lff_ours_all(self, args):
        print('Training LfF ...')

        num_updated = 0
        train_g_iter = iter(self.train_g_loader) # Original + Generated dataset
        
        del self.model_b
        self.model_b = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)
        
        self.optimizer_b = torch.optim.Adam(
                self.model_b.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        
        if args.use_lr_decay:
            self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_step,gamma=args.lr_gamma)
            self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_d, step_size=args.lr_decay_step,gamma=args.lr_gamma)

        for step in tqdm(range(args.num_steps)):
            # train main model
            try:
                index, data, attr, _ = next(train_g_iter)
            except:
                train_g_iter = iter(self.train_g_loader)
                index, data, attr, _ = next(train_g_iter)
                
            data = data.to(self.device)
            attr = attr.to(self.device)
            index = index.to(self.device)
            label = attr[:, args.target_attr_idx]

            logit_b = self.model_b(data)
            logit_d = self.model_d(data)

            loss_b = self.criterion(logit_b, label).cpu().detach()
            loss_d = self.criterion(logit_d, label).cpu().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d')

            # EMA sample loss
            self.sample_loss_ema_b.update(loss_b, index)
            self.sample_loss_ema_d.update(loss_d, index)

            # class-wise normalize
            loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
            loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b_ema')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d_ema')

            label_cpu = label.cpu()

            for c in range(self.num_classes):
                class_index = np.where(label_cpu == c)[0]
                max_loss_b = self.sample_loss_ema_b.max_loss(c) + 1e-8
                max_loss_d = self.sample_loss_ema_d.max_loss(c)
                loss_b[class_index] /= max_loss_b
                loss_d[class_index] /= max_loss_d

            # re-weighting based on loss value / generalized CE for biased model
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)

            if np.isnan(loss_weight.mean().item()):
                raise NameError('loss_weight')
            
            loss_b_update = self.bias_criterion(logit_b, label)
            loss_d_update = self.criterion(logit_d, label) * loss_weight.to(self.device)

            if np.isnan(loss_b_update.mean().item()):
                raise NameError('loss_b_update')

            if np.isnan(loss_d_update.mean().item()):
                raise NameError('loss_d_update')

            loss = loss_b_update.mean() + loss_d_update.mean()
            num_updated += loss_weight.mean().item() * data.size(0)

            self.optimizer_b.zero_grad()
            self.optimizer_d.zero_grad()
            loss.backward()
            self.optimizer_b.step()
            self.optimizer_d.step()
            
            if args.use_lr_decay:
                self.scheduler_b.step()
                self.scheduler_l.step()

            if args.use_lr_decay and step % args.lr_decay_step == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer_b lr: {self.optimizer_b.param_groups[-1]['lr']}")
                print(f"self.optimizer_d lr: {self.optimizer_d.param_groups[-1]['lr']}")
                    
            if step % args.valid_freq == 0:
                self.board_lff_acc(step)

                if args.use_lr_decay and args.tensorboard:
                    self.writer.add_scalar(f"loss/learning rate", self.optimizer_d.param_groups[-1]['lr'], step)
                    
    def train_disent_be_ours(self, args):
        print('Training DisEnt with BiasEnsemble ...')
        train_num = len(self.train_dataset)

        # self.model_d   : model for predicting intrinsic attributes ((E_i,C_i) in the main paper)
        # self.model_d.fc: fc layer for predicting intrinsic attributes (C_i in the main paper)
        # self.model_b   : model for predicting bias attributes ((E_b, C_b) in the main paper)
        # self.model_b.fc: fc layer for predicting bias attributes (C_b in the main paper)

        #################
        # define models
        #################
        if args.dataset == 'cmnist' and args.model == 'MLP':
            model_name = 'mlp_DISENTANGLE'
        else:
            model_name = 'resnet_DISENTANGLE'

        print(f'criterion: {self.criterion}')
        print(f'bias criterion: {self.bias_criterion}')

        train_iter = iter(self.train_loader) # Original dataset
        train_g_iter = iter(self.train_g_loader) # Original + Our dataset
        train_num = len(self.train_dataset.dataset)

        self.conflicting_index = torch.zeros(train_num, 1)
        self.label_index = torch.zeros(train_num).long().to(self.device)

        mask_index = torch.zeros(train_num, 1)

        #### BiasEnsemble ####
        pseudo_align_flag = self.pretrain_b_ensemble_best(args) # For only original dataset

        del self.model_b
        self.model_b = get_model(model_name, self.num_classes).to(self.device)
        self.model_d = get_model(model_name, self.num_classes).to(self.device)

        ##################
        # define optimizer
        ##################

        self.optimizer_d = torch.optim.Adam(
            self.model_d.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        self.optimizer_b = torch.optim.Adam(
            self.model_b.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        if args.use_lr_decay:
            self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_step,
                                                         gamma=args.lr_gamma)
            self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_d, step_size=args.lr_decay_step,
                                                         gamma=args.lr_gamma)

        mask_index[pseudo_align_flag] = 1

        for step in tqdm(range(args.num_steps)):
            # Load a batch from original + ours
            try:
                index, data, attr, image_path = next(train_g_iter)
            except:
                train_g_iter = iter(self.train_g_loader)
                index, data, attr, image_path = next(train_g_iter)

            data = data.to(self.device)
            attr = attr.to(self.device)
            index = index.to(self.device)
            label = attr[:, args.target_attr_idx].to(self.device)

            # Feature extraction
            # Prediction by concatenating zero vectors (dummy vectors).
            # We do not use the prediction here.
            if args.dataset == 'cmnist' and args.model == 'MLP':
                z_l = self.model_d.extract(data)
                with torch.no_grad():
                    z_b = self.model_b.extract(data)
            else:
                with torch.no_grad():
                    z_b = []
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]

                z_l = []
                hook_fn = self.model_d.avgpool.register_forward_hook(self.concat_dummy(z_l))
                _ = self.model_d(data)
                hook_fn.remove()

                z_l = z_l[0]

            # z=[z_l, z_b]
            # Gradients of z_b are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
            z_conflict = torch.cat((z_l, z_b.detach()), dim=1)
            z_align = torch.cat((z_l.detach(), z_b), dim=1)

            # Prediction using z=[z_l, z_b]
            pred_conflict = self.model_d.fc(z_conflict)
            with torch.no_grad():
                pred_align = self.model_b.fc(z_align)

            loss_dis_conflict = self.criterion(pred_conflict, label).detach()
            loss_dis_align = self.criterion(pred_align, label).detach()

            # EMA sample loss
            self.sample_loss_ema_d.update(loss_dis_conflict, index)
            self.sample_loss_ema_b.update(loss_dis_align, index)

            # class-wise normalize
            loss_dis_conflict = self.sample_loss_ema_d.parameter[index].clone().detach()
            loss_dis_align = self.sample_loss_ema_b.parameter[index].clone().detach()

            loss_dis_conflict = loss_dis_conflict.to(self.device)
            loss_dis_align = loss_dis_align.to(self.device)

            for c in range(self.num_classes):
                class_index = torch.where(label == c)[0].to(self.device)
                max_loss_conflict = self.sample_loss_ema_d.max_loss(c)
                max_loss_align = self.sample_loss_ema_b.max_loss(c)
                loss_dis_conflict[class_index] /= max_loss_conflict
                loss_dis_align[class_index] /= max_loss_align

            loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)  # Eq.1 (reweighting module) in the main paper
            loss_dis_conflict = self.criterion(pred_conflict, label) * loss_weight.to(self.device)  # Eq.2 W(z)CE(C_i(z),y)

            # For B.Nets load a batch from original dataset
            try:
                index_o, data_o, attr_o, image_path_o = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index_o, data_o, attr_o, image_path_o = next(train_iter)

            data_o = data_o.to(self.device)
            attr_o = attr_o.to(self.device)
            index_o = index_o.to(self.device)
            label_o = attr_o[:, args.target_attr_idx].to(self.device)

            if args.dataset == 'cmnist' and args.model == 'MLP':
                with torch.no_grad():
                    z_l_o = self.model_d.extract(data_o)
                z_b_o = self.model_b.extract(data_o)
            else:
                z_b_o = []
                hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b_o))
                _ = self.model_b(data_o)
                hook_fn.remove()

                z_b_o = z_b_o[0]
                with torch.no_grad():
                    z_l_o = []
                    hook_fn = self.model_d.avgpool.register_forward_hook(self.concat_dummy(z_l_o))
                    _ = self.model_d(data_o)
                    hook_fn.remove()

                    z_l_o = z_l_o[0]

            z_align_o = torch.cat((z_l_o.detach(), z_b_o), dim=1)
            curr_align_flag = torch.index_select(mask_index.to(self.device), 0, index_o)
            curr_align_flag = (curr_align_flag.squeeze(1) == 1)
            pred_align_o = self.model_b.fc(z_align_o)
            loss_dis_align = self.criterion(pred_align_o[curr_align_flag], label_o[curr_align_flag])

            # feature-level augmentation : augmentation after certain iteration (after representation is disentangled at a certain level)
            if step > args.curr_step:
                indices = np.random.permutation(z_b.size(0))
                z_b_swap = z_b[indices]  # z tilde

                indices_o = np.random.permutation(z_b_o.size(0))
                z_b_swap_o = z_b_o[indices_o]  # z tilde
                label_swap_o = label_o[indices_o]  # y tilde
                curr_align_flag = curr_align_flag[indices_o]

                # Prediction using z_swap=[z_l, z_b tilde]
                # Again, gradients of z_b tilde are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
                z_mix_conflict = torch.cat((z_l, z_b_swap.detach()), dim=1)
                z_mix_align = torch.cat((z_l_o.detach(), z_b_swap_o), dim=1)

                # Prediction using z_swap
                pred_mix_conflict = self.model_d.fc(z_mix_conflict)
                pred_mix_align = self.model_b.fc(z_mix_align)

                loss_swap_conflict = self.criterion(pred_mix_conflict, label) * loss_weight.to(self.device)  # Eq.3 W(z)CE(C_i(z_swap),y)
                loss_swap_align = self.criterion(pred_mix_align[curr_align_flag], label_swap_o[curr_align_flag])
                lambda_swap = self.args.lambda_swap  # Eq.3 lambda_swap_b
            else:
                # before feature-level augmentation
                loss_swap_conflict = torch.tensor([0]).float()
                loss_swap_align = torch.tensor([0]).float()
                lambda_swap = 0

            loss_dis = loss_dis_conflict.mean() + args.lambda_dis_align * loss_dis_align.mean()  # Eq.2 L_dis
            loss_swap = loss_swap_conflict.mean() + args.lambda_swap_align * loss_swap_align.mean()  # Eq.3 L_swap
            loss = loss_dis + lambda_swap * loss_swap  # Eq.4 Total objective

            self.optimizer_d.zero_grad()
            self.optimizer_b.zero_grad()
            loss.backward()
            self.optimizer_d.step()
            self.optimizer_b.step()

            if step >= args.curr_step and args.use_lr_decay:
                self.scheduler_b.step()
                self.scheduler_l.step()

            if args.use_lr_decay and step % args.lr_decay_step == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer_b lr: {self.optimizer_b.param_groups[-1]['lr']}")
                print(f"self.optimizer_d lr: {self.optimizer_d.param_groups[-1]['lr']}")

            if step % args.valid_freq == 0:
                self.board_disent_acc(step)
                
    def train_disent_ours(self, args):
        print('Training DisEnt with BiasEnsemble ...')
        train_num = len(self.train_dataset)

        # self.model_d   : model for predicting intrinsic attributes ((E_i,C_i) in the main paper)
        # self.model_d.fc: fc layer for predicting intrinsic attributes (C_i in the main paper)
        # self.model_b   : model for predicting bias attributes ((E_b, C_b) in the main paper)
        # self.model_b.fc: fc layer for predicting bias attributes (C_b in the main paper)

        #################
        # define models
        #################
        if args.dataset == 'cmnist' and args.model == 'MLP':
            model_name = 'mlp_DISENTANGLE'
        else:
            model_name = 'resnet_DISENTANGLE'

        print(f'criterion: {self.criterion}')
        print(f'bias criterion: {self.bias_criterion}')

        train_iter = iter(self.train_loader) # Original dataset
        train_g_iter = iter(self.train_g_loader) # Original + Our dataset
        train_num = len(self.train_dataset.dataset)

        self.conflicting_index = torch.zeros(train_num, 1)
        self.label_index = torch.zeros(train_num).long().to(self.device)

        mask_index = torch.zeros(train_num, 1)

        del self.model_b
        self.model_b = get_model(model_name, self.num_classes).to(self.device)
        self.model_d = get_model(model_name, self.num_classes).to(self.device)

        ##################
        # define optimizer
        ##################

        self.optimizer_d = torch.optim.Adam(
            self.model_d.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        self.optimizer_b = torch.optim.Adam(
            self.model_b.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        if args.use_lr_decay:
            self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_step,
                                                         gamma=args.lr_gamma)
            self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_d, step_size=args.lr_decay_step,
                                                         gamma=args.lr_gamma)

        for step in tqdm(range(args.num_steps)):
            # Load a batch from original + ours
            try:
                index, data, attr, image_path = next(train_g_iter)
            except:
                train_g_iter = iter(self.train_g_loader)
                index, data, attr, image_path = next(train_g_iter)

            data = data.to(self.device)
            attr = attr.to(self.device)
            index = index.to(self.device)
            label = attr[:, args.target_attr_idx].to(self.device)

            # Feature extraction
            # Prediction by concatenating zero vectors (dummy vectors).
            # We do not use the prediction here.
            if args.dataset == 'cmnist' and args.model == 'MLP':
                z_l = self.model_d.extract(data)
                with torch.no_grad():
                    z_b = self.model_b.extract(data)
            else:
                with torch.no_grad():
                    z_b = []
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]

                z_l = []
                hook_fn = self.model_d.avgpool.register_forward_hook(self.concat_dummy(z_l))
                _ = self.model_d(data)
                hook_fn.remove()

                z_l = z_l[0]

            # z=[z_l, z_b]
            # Gradients of z_b are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
            z_conflict = torch.cat((z_l, z_b.detach()), dim=1)
            z_align = torch.cat((z_l.detach(), z_b), dim=1)

            # Prediction using z=[z_l, z_b]
            pred_conflict = self.model_d.fc(z_conflict)
            with torch.no_grad():
                pred_align = self.model_b.fc(z_align)

            loss_dis_conflict = self.criterion(pred_conflict, label).detach()
            loss_dis_align = self.criterion(pred_align, label).detach()

            # EMA sample loss
            self.sample_loss_ema_d.update(loss_dis_conflict, index)
            self.sample_loss_ema_b.update(loss_dis_align, index)

            # class-wise normalize
            loss_dis_conflict = self.sample_loss_ema_d.parameter[index].clone().detach()
            loss_dis_align = self.sample_loss_ema_b.parameter[index].clone().detach()

            loss_dis_conflict = loss_dis_conflict.to(self.device)
            loss_dis_align = loss_dis_align.to(self.device)

            for c in range(self.num_classes):
                class_index = torch.where(label == c)[0].to(self.device)
                max_loss_conflict = self.sample_loss_ema_d.max_loss(c)
                max_loss_align = self.sample_loss_ema_b.max_loss(c)
                loss_dis_conflict[class_index] /= max_loss_conflict
                loss_dis_align[class_index] /= max_loss_align

            loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)  # Eq.1 (reweighting module) in the main paper
            loss_dis_conflict = self.criterion(pred_conflict, label) * loss_weight.to(self.device)  # Eq.2 W(z)CE(C_i(z),y)

            # For B.Nets load a batch from original dataset
            try:
                index_o, data_o, attr_o, image_path_o = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index_o, data_o, attr_o, image_path_o = next(train_iter)

            data_o = data_o.to(self.device)
            attr_o = attr_o.to(self.device)
            index_o = index_o.to(self.device)
            label_o = attr_o[:, args.target_attr_idx].to(self.device)

            if args.dataset == 'cmnist' and args.model == 'MLP':
                with torch.no_grad():
                    z_l_o = self.model_d.extract(data_o)
                z_b_o = self.model_b.extract(data_o)
            else:
                z_b_o = []
                hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b_o))
                _ = self.model_b(data_o)
                hook_fn.remove()

                z_b_o = z_b_o[0]
                with torch.no_grad():
                    z_l_o = []
                    hook_fn = self.model_d.avgpool.register_forward_hook(self.concat_dummy(z_l_o))
                    _ = self.model_d(data_o)
                    hook_fn.remove()

                    z_l_o = z_l_o[0]

            z_align_o = torch.cat((z_l_o.detach(), z_b_o), dim=1)
            pred_align_o = self.model_b.fc(z_align_o)
            loss_dis_align = self.criterion(pred_align_o, label_o)

            # feature-level augmentation : augmentation after certain iteration (after representation is disentangled at a certain level)
            if step > args.curr_step:
                indices = np.random.permutation(z_b.size(0))
                z_b_swap = z_b[indices]  # z tilde

                indices_o = np.random.permutation(z_b_o.size(0))
                z_b_swap_o = z_b_o[indices_o]  # z tilde
                label_swap_o = label_o[indices_o]  # y tilde

                # Prediction using z_swap=[z_l, z_b tilde]
                # Again, gradients of z_b tilde are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
                z_mix_conflict = torch.cat((z_l, z_b_swap.detach()), dim=1)
                z_mix_align = torch.cat((z_l_o.detach(), z_b_swap_o), dim=1)

                # Prediction using z_swap
                pred_mix_conflict = self.model_d.fc(z_mix_conflict)
                pred_mix_align = self.model_b.fc(z_mix_align)

                loss_swap_conflict = self.criterion(pred_mix_conflict, label) * loss_weight.to(self.device)  # Eq.3 W(z)CE(C_i(z_swap),y)
                loss_swap_align = self.criterion(pred_mix_align, label_swap_o)
                lambda_swap = self.args.lambda_swap  # Eq.3 lambda_swap_b
            else:
                # before feature-level augmentation
                loss_swap_conflict = torch.tensor([0]).float()
                loss_swap_align = torch.tensor([0]).float()
                lambda_swap = 0

            loss_dis = loss_dis_conflict.mean() + args.lambda_dis_align * loss_dis_align.mean()  # Eq.2 L_dis
            loss_swap = loss_swap_conflict.mean() + args.lambda_swap_align * loss_swap_align.mean()  # Eq.3 L_swap
            loss = loss_dis + lambda_swap * loss_swap  # Eq.4 Total objective

            self.optimizer_d.zero_grad()
            self.optimizer_b.zero_grad()
            loss.backward()
            self.optimizer_d.step()
            self.optimizer_b.step()

            if step >= args.curr_step and args.use_lr_decay:
                self.scheduler_b.step()
                self.scheduler_l.step()

            if args.use_lr_decay and step % args.lr_decay_step == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer_b lr: {self.optimizer_b.param_groups[-1]['lr']}")
                print(f"self.optimizer_d lr: {self.optimizer_d.param_groups[-1]['lr']}")

            if step % args.valid_freq == 0:
                self.board_disent_acc(step)

    def test_lff_be(self, args):
        if args.dataset == 'cmnist':
            self.model_b = get_backbone("MLP", self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)
            self.model_d = get_backbone("MLP", self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)
        else:
            self.model_b = get_backbone("ResNet18", self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)
            self.model_d = get_backbone("ResNet18", self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)

        self.model_d.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_d.th'))['state_dict'])
        self.model_b.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_b.th'))['state_dict'])
        self.board_lff_acc(step=0, inference=True)

    def test_disent_be(self, args):
        if args.dataset == 'cmnist':
            self.model_d = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
        else:
            self.model_d = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)

        self.model_d.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_d.th'))['state_dict'])
        self.model_b.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_b.th'))['state_dict'])
        self.board_disent_acc(step=0, inference=True)

