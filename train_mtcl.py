import os
import json
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
# import umap
from sklearn.preprocessing import LabelEncoder

from utils import *
from loader import EEGDataLoader
from models.main_model import MainModel


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
        self.ds_cfg = config['dataset']
        self.fp_cfg = config['feature_pyramid']
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.train_iter = 0

        # ========== 路径隔离：保存目录必须带模式后缀 ==========
        # 读取用 config['name']（原始配置名），保存用隔离后的名字
        if self.tp_cfg['mode'] == 'scratch':
            save_dir_name = config['name']
        else:
            save_dir_name = f"{config['name']}_{self.tp_cfg['mode']}"

        self.ckpt_path = os.path.join('checkpoints', save_dir_name)
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        # ========================================================

        self.early_stopping = EarlyStopping(
            patience=self.es_cfg['patience'],
            verbose=True,
            ckpt_path=self.ckpt_path,
            ckpt_name=self.ckpt_name,
            mode=self.es_cfg['mode']
        )

        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        self.criterion = nn.CrossEntropyLoss()
        self.activate_train_mode()
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.tp_cfg['lr'],
            weight_decay=self.tp_cfg['weight_decay']
        )

        self.ckpt_path = os.path.join('checkpoints', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path,
                                            ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])

    def build_model(self):
        model = MainModel(self.cfg)

        # ==========================================
        # FLOPs 计算（保持原有）
        # ==========================================
        try:
            from torchinfo import summary
            real_input_len = 15000
            print("\n" + "=" * 50)
            print("[INFO] Calculating FLOPs with torchinfo...")
            model_stats = summary(model,
                                  input_size=(1, 1, real_input_len),
                                  col_names=["input_size", "output_size", "num_params", "mult_adds"],
                                  verbose=0)
            total_mult_adds = model_stats.total_mult_adds
            total_params = model_stats.total_params
            print(f"[Statistics] Input Shape: (1, 1, {real_input_len})")
            print(f"[Statistics] Params: {total_params / 1e6:.4f} M")
            print(f"[Statistics] FLOPs (Mult-Adds): {total_mult_adds / 1e9:.4f} G")
            print("=" * 50 + "\n")
        except ImportError:
            print("[WARN] 请安装 torchinfo: pip install torchinfo")
        except Exception as e:
            print(f"[WARN] torchinfo 计算失败: {e}")
        # ==========================================

        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))

        # ========== 加载预训练权重（完全硬编码路径） ==========
        if self.tp_cfg['mode'] != 'scratch':
            print('[INFO] Model loaded for finetune')

            # ✅ 预训练路径完全硬编码，与当前 config['name'] 无关
            load_name = "SleePyCo-Transformer_SL-01_numScales-1_MASS_pretrain"
            load_path = os.path.join('checkpoints', load_name, f'ckpt_fold-{self.fold:02d}.pth')

            print(f'[INFO] Loading pretrain weights from : {load_path}')
            print(f'[INFO] Saving finetune checkpoints to: {self.ckpt_path}')

            if not os.path.exists(load_path):
                raise FileNotFoundError(f"[ERROR] Pretrain checkpoint not found: {load_path}")

            model.load_state_dict(torch.load(load_path), strict=False)

            backbone_params = sum(p.numel() for p in model.module.feature.parameters() if p.requires_grad)
            print('[INFO] Number of params of backbone: ', backbone_params)
        # ========================================================

        # 按模块打印参数量
        print("[INFO] 模型按模块参数量统计：")
        print_module_params(model)

        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model

    def build_dataloader(self):
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=True,
                                  num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False,
                                num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True)
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False,
                                 num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}

    def activate_train_mode(self):
        self.model.train()
        if self.tp_cfg['mode'] == 'freezefinetune':
            print('[INFO] Freeze backone')
            self.model.module.feature.train(False)
            for p in self.model.module.feature.parameters():
                p.requires_grad = False

            print('[INFO] Unfreeze conv_c5')
            self.model.module.feature.conv_c5.train(True)
            for p in self.model.module.feature.conv_c5.parameters(): p.requires_grad = True

            if self.fp_cfg['num_scales'] > 1:
                print('[INFO] Unfreeze conv_c4')
                self.model.module.feature.conv_c4.train(True)
                for p in self.model.module.feature.conv_c4.parameters(): p.requires_grad = True

            if self.fp_cfg['num_scales'] > 2:
                print('[INFO] Unfreeze conv_c3')
                self.model.module.feature.conv_c3.train(True)
                for p in self.model.module.feature.conv_c3.parameters(): p.requires_grad = True

    def train_one_epoch(self, epoch):
        correct, total, train_loss = 0, 0, 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            loss = 0
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            outputs = self.model(inputs)
            outputs_sum = torch.zeros_like(outputs[0])

            for j in range(len(outputs)):
                loss += self.criterion(outputs[j], labels)
                outputs_sum += outputs[j]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()
            self.train_iter += 1

            progress_bar(i, len(self.loader_dict['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (i + 1), 100. * correct / total, correct, total))

            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                val_acc, val_loss = self.evaluate(mode='val')
                self.early_stopping(val_acc, val_loss, self.model)
                self.activate_train_mode()
                if self.early_stopping.early_stop:
                    break

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.cfg['classifier']['num_classes']))

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            loss = 0
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            outputs = self.model(inputs)
            outputs_sum = torch.zeros_like(outputs[0])

            for j in range(len(outputs)):
                loss += self.criterion(outputs[j], labels)
                outputs_sum += outputs[j]

            eval_loss += loss.item()
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()

            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs_sum.cpu().numpy()])

            progress_bar(i, len(self.loader_dict[mode]), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (eval_loss / (i + 1), 100. * correct / total, correct, total))

        if mode == 'val':
            return 100. * correct / total, eval_loss
        elif mode == 'test':
            return y_true, y_pred
        else:
            raise NotImplementedError

    def run(self):
        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch(epoch)
            if self.early_stopping.early_stop:
                break

        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        y_true, y_pred = self.evaluate(mode='test')
        print('')

        return y_true, y_pred


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

    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    for fold in range(1, config['dataset']['num_splits'] + 1):
        trainer = OneFoldTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])

        summarize_result(config, fold, Y_true, Y_pred)


if __name__ == "__main__":
    main()
