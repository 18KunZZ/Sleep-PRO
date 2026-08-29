import os
import json
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from thop import profile

import time

from utils import *
from loader import EEGDataLoader
from models.main_model import MainModel
from loss import SupSwAVLoss


class_margins = {
    0: 0.2,  # 清醒期
    1: 0.02, # N1期
    2: 0.1,  # N2期
    3: 0.2, # N3期（深度睡眠，类内差异小）
    4: 0.1  # REM期
}


def print_module_params(model, indent=0):
    """
    按模块层级递归打印参数量

    Args:
        model: PyTorch模型
        indent: 缩进级别（用于层级显示）
    """
    indent_str = "  " * indent
    total_params = 0

    # 遍历所有子模块
    for name, child in model.named_children():
        # 计算当前子模块的参数量
        child_params = sum(p.numel() for p in child.parameters() if p.requires_grad)
        total_params += child_params

        # 如果子模块还有子模块，递归打印
        if len(list(child.children())) > 0:
            print(f"{indent_str}- {name}: {child_params:,} parameters")
            print_module_params(child, indent + 1)
        else:
            # 叶子节点（基础模块）直接打印
            print(f"{indent_str}  - {name}: {child_params:,} parameters")

    # 打印当前层级的总参数量
    if indent == 0:  # 顶层
        print(f"{indent_str}Total: {total_params:,} parameters")


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold

        self.cfg = config
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        # self.criterion = SupConLoss(temperature=self.tp_cfg['temperature'])
        self.criterion = SupSwAVLoss(temperature=self.tp_cfg['temperature'])

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.tp_cfg['lr'], weight_decay=self.tp_cfg['weight_decay'])

        self.ckpt_path = os.path.join('checkpoints', config['name'])

        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])



    def build_model(self):
        model = MainModel(self.cfg)
        model.to(self.device)  # 先移动到 GPU/CPU

        # ==========================================
        # 🧮 [新增] FLOPs 和 参数量计算代码
        # ==========================================
        try:
            print("\n" + "=" * 40)
            print("[INFO] Calculating FLOPs and Params...")

            # 1. 构造一个假的输入 (Batch=1, Channel=1, Length=3000)
            # 请根据你 config['dataset']['input_size'] 确认长度是 3000 还是其他
            input_len = self.cfg['dataset'].get('input_size', 3000)
            dummy_input = torch.randn(1, 1, input_len).to(self.device)

            # 2. 使用 thop 计算
            # inputs 参数必须是 tuple，所以要加括号和逗号 (dummy_input, )
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)

            # 3. 打印结果 (自动换算为 M / G)
            print(f"----------------------------------------")
            print(f"输入尺寸: (1, 1, {input_len})")
            print(f"参数量 (Params): {params / 1e6:.4f} M (百万)")
            print(f"计算量 (FLOPs) : {flops / 1e9:.4f} G (十亿)")
            print(f"----------------------------------------")
            print("=" * 40 + "\n")

        except Exception as e:
            print(f"[WARNING] FLOPs 计算失败 (可能是层结构复杂导致): {e}")
        # ==========================================


        print('[INFO] Number of params of models: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model

    def build_dataloader(self):
        dataloader_args = {'batch_size': self.tp_cfg['batch_size'], 'shuffle': True, 'num_workers': 4*len(self.args.gpu.split(",")), 'pin_memory': True}
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, **dataloader_args)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, **dataloader_args)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader}


    def train_one_epoch(self):
        self.model.train()
        train_loss = 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            loss = 0
            labels = labels.view(-1).to(self.device)

            # 1. 预处理：将list中的每个numpy视图转为Tensor，并添加batch维度
            # inputs_list：MultiCrop生成的list，每个元素是[C, L]（C=1通道，L=不同长度）
            tensor_list = []
            for view in inputs:
                # 1. 转移设备；
                tensor = view.to(self.device)  # 无需.float()（view已是float类型）
                tensor_list.append(tensor)

            # 2. 分批前向：每个视图单独过模型，获取特征
            view_features = []
            for tensor in tensor_list:
                # 模型输出：[1, C, T]（C=通道数，T=模型输出的时间维度）
                feat = self.model(tensor)[0]  # 取pretrain模式的输出（如c5）
                view_features.append(feat)

            outputs = torch.cat(view_features, dim=0)

            # inputs = torch.concat(inputs, dim=0).to(self.device)
            #
            # outputs = self.models(inputs)[0]
            # print(outputs.shape)

            # 5. 重塑为SupConLoss要求的格式：[batch_size, n_views, C]
            B = labels.size(0)
            V = 6  # 比如 5
            split_sizes = [B] * V  # [B, B, B, B, B]

            views = torch.split(outputs, split_sizes, dim=0)  # 返回一个 tuple，长度 V，每个 [B, D]

            # 拼成 [B, V, D]
            features = torch.cat([v.unsqueeze(1) for v in views], dim=1)

            # 6. 计算损失（此时格式完全正确）
            loss += self.criterion(features, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            self.train_iter += 1

            progress_bar(i, len(self.loader_dict['train']), 'Lr: %.4e | Loss: %.3f' %(get_lr(self.optimizer), train_loss / (i + 1)))

            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                val_loss = self.evaluate(mode='val')
                self.early_stopping(None, val_loss, self.model)
                self.model.train()
                if self.early_stopping.early_stop:
                    break

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        eval_loss = 0

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):

            loss = 0
            inputs = inputs.to(self.device)

            if inputs.dim() == 4:
                inputs = inputs.squeeze(-1)

            labels = labels.view(-1).to(self.device)

            outputs = self.model(inputs)[0]

            # features = outputs.unsqueeze(1).repeat(1, 2, 1)
            features = outputs.unsqueeze(1)  # [B, 1, D]
            # ProtoConLoss 要求至少 2 views，为简便建议只跳过验证阶段损失，或传 f1 + f1 模拟
            features = torch.cat([features, features, features, features, features, features], dim=1)  # [B, 6, D]

            loss += self.criterion(features, labels)

            eval_loss += loss.item()

            progress_bar(i, len(self.loader_dict[mode]), 'Lr: %.4e | Loss: %.3f' %(get_lr(self.optimizer), eval_loss / (i + 1)))

        return eval_loss


    def run(self):

        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch()
            if self.early_stopping.early_stop:
                break

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')

    for fold in range(1, config['dataset']['num_splits'] + 1):
        trainer = OneFoldTrainer(args, fold, config)
        trainer.run()


if __name__ == "__main__":
    main()

