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
    0: 0.2,  # Wake stage
    1: 0.02, # N1 stage
    2: 0.1,  # N2 stage
    3: 0.2, # N3 stage (deep sleep with low intra-class variation)
    4: 0.1  # REM stage
}


def print_module_params(model, indent=0):
    """
    Recursively print parameter counts by module hierarchy.

    Args:
        model: PyTorch model.
        indent: Indentation level used to display the hierarchy.
    """
    indent_str = "  " * indent
    total_params = 0

    # Iterate over all child modules
    for name, child in model.named_children():
        # Count parameters in the current child module
        child_params = sum(p.numel() for p in child.parameters() if p.requires_grad)
        total_params += child_params

        # Recurse when the child module contains nested modules
        if len(list(child.children())) > 0:
            print(f"{indent_str}- {name}: {child_params:,} parameters")
            print_module_params(child, indent + 1)
        else:
            # Print leaf modules directly
            print(f"{indent_str}  - {name}: {child_params:,} parameters")

    # Print the total parameter count at the top level
    if indent == 0:  # Top level
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
        model.to(self.device)  # Move the model to the selected device first

        # ==========================================
        # FLOPs and parameter-count calculation
        # ==========================================
        try:
            print("\n" + "=" * 40)
            print("[INFO] Calculating FLOPs and Params...")

            # 1. Create a dummy input (batch=1, channel=1, length=3000)
            # Confirm the actual length from config['dataset']['input_size']
            input_len = self.cfg['dataset'].get('input_size', 3000)
            dummy_input = torch.randn(1, 1, input_len).to(self.device)

            # 2. Calculate metrics with thop
            # inputs must be a tuple, so retain the comma in (dummy_input,)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)

            # 3. Print results converted to millions and billions
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

            # 1. Preprocess each NumPy view in the list as a tensor
            # MultiCrop produces [C, L] views, where C=1 and L may vary
            tensor_list = []
            for view in inputs:
                # Move the view to the selected device
                tensor = view.to(self.device)  # No .float() call is needed
                tensor_list.append(tensor)

            # 2. Run each view through the model separately to extract features
            view_features = []
            for tensor in tensor_list:
                # Model output: [1, C, T], with channel and temporal dimensions
                feat = self.model(tensor)[0]  # Use the pretraining output, such as c5
                view_features.append(feat)

            outputs = torch.cat(view_features, dim=0)

            # inputs = torch.concat(inputs, dim=0).to(self.device)
            #
            # outputs = self.models(inputs)[0]
            # print(outputs.shape)

            # 5. Reshape to the [batch_size, n_views, C] format expected by SupConLoss
            B = labels.size(0)
            V = 6  # Number of views
            split_sizes = [B] * V  # [B, B, B, B, B]

            views = torch.split(outputs, split_sizes, dim=0)  # V tensors shaped [B, D]

            # Concatenate into [B, V, D]
            features = torch.cat([v.unsqueeze(1) for v in views], dim=1)

            # 6. Compute the loss using the required feature format
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
            # ProtoConLoss needs at least two views; duplicate features for validation
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
