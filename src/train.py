import time
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utils.config import Config
from utils.utils_func import *
from models.model import GCN



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        torch.manual_seed(999)
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
 
        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr

        # download and shuffle the data
        self.dataset = load_data(self.base_path)
        num_node_features = self.dataset.num_node_features
        cls_num = self.dataset.num_classes

        # split train / val / test
        train_len = int(len(self.dataset) * 0.7)
        val_len = int(len(self.dataset) * 0.15)


        if self.mode == 'train':
            self.dataset = {
                'train': self.dataset[:train_len],
                'val': self.dataset[train_len:train_len+val_len],
                'test': self.dataset[train_len+val_len:]
                }
            self.dataloaders = {
                s: DataLoader(d, self.batch_size, shuffle=True) if s == 'train' else DataLoader(d, self.batch_size, shuffle=False)
                for s, d in self.dataset.items()}
        else:
            self.dataset = {
                'test': self.dataset[train_len+val_len:]
                }
            self.dataloaders = {s: DataLoader(d, self.batch_size, shuffle=False) for s, d in self.dataset.items() if s == 'test'}

        # model, optimizer, loss
        self.model = GCN(self.config, num_node_features, cls_num).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        if self.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
            total_steps = len(self.dataloaders['train']) * self.epochs
            pct_start = 10 / total_steps
            final_div_factor = self.lr / 25 / 1e-6
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=pct_start, final_div_factor=final_div_factor)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        else:
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])    
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()


    def train(self):
        early_stop = 0
        best_val_acc = 0 if not self.continuous else self.loss_data['best_val_acc']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']
        self.loss_data = {
            'train_history': {'loss': [], 'acc': []}, \
            'val_history': {'loss': [], 'acc': []}
            }

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)

            for phase in ['train', 'val']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                total_loss, total_acc = 0, 0
                for i, data in enumerate(self.dataloaders[phase]):
                    batch_size = data.y.size(0)
                    data = data.to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        output, _ = self.model(data)
                        loss = self.criterion(output, data.y)
                        
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()

                        acc = torch.sum(torch.argmax(output, dim=-1).detach().cpu() == data.y.detach().cpu()) / batch_size

                    total_loss += loss.item()*batch_size
                    total_acc += acc.item()*batch_size

                    if i % 10 == 0:
                        print('Epoch {}: {}/{} step loss: {} acc: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item(), acc.item()))

                epoch_loss = total_loss/len(self.dataloaders[phase].dataset)
                epoch_acc = total_acc/len(self.dataloaders[phase].dataset)

                print('{} loss: {:4f}, acc: {:4f}\n'.format(phase, epoch_loss, epoch_acc))
                
                if phase == 'train':
                    self.loss_data['train_history']['loss'].append(epoch_loss)
                    self.loss_data['train_history']['acc'].append(epoch_acc)
                elif phase == 'val':
                    self.loss_data['val_history']['loss'].append(epoch_loss)
                    self.loss_data['val_history']['acc'].append(epoch_acc)
                    # save best model
                    early_stop += 1
                    if  epoch_acc > best_val_acc:
                        early_stop = 0
                        best_val_acc = epoch_acc
                        best_epoch = best_epoch_info + epoch + 1
                        save_checkpoint(self.model_path, self.model, self.optimizer)

            print("time: {} s\n".format(time.time() - start))
            print('\n'*2)

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break
        
        print('best val acc: {:4f}, best epoch: {:d}\n'.format(best_val_acc, best_epoch))
        self.loss_data['best_val_acc'] = best_val_acc
        self.loss_data['best_epoch'] = best_epoch
        return self.loss_data


    def test(self, phase):
        with torch.no_grad():
            self.model.eval()
            total_loss, total_acc = 0, 0
            all_features, all_label, all_pred = [], [], []
            
            for data in tqdm(self.dataloaders[phase]):
                batch_size = data.y.size(0)
                data = data.to(self.device)

                output, feature = self.model(data)
                loss = self.criterion(output, data.y)
                pred = torch.argmax(output, dim=-1).detach().cpu()
                acc = torch.sum(pred == data.y.detach().cpu()) / batch_size

                all_features.append(feature.detach().cpu())
                all_label.append(data.y.detach().cpu())
                all_pred.append(pred)

                total_loss += loss.item()*batch_size
                total_acc += acc.item()*batch_size

            print('{} set: loss: {}, acc: {}'.format(phase, total_loss/len(self.dataloaders[phase].dataset), total_acc/len(self.dataloaders[phase].dataset)))

        # feature visualization
        all_features = torch.cat(all_features, dim=0).numpy()
        all_label = torch.cat(all_label, dim=0).numpy()
        all_pred = torch.cat(all_pred, dim=0).numpy()
        
        tsne = TSNE()
        x_test_2D = tsne.fit_transform(all_features)
        x_test_2D = (x_test_2D - x_test_2D.min())/(x_test_2D.max() - x_test_2D.min())

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plt.setp(ax, xticks=[], yticks=[])
        
        ax[0].scatter(x_test_2D[:, 0], x_test_2D[:, 1], s=10, cmap='tab10', c=all_label)
        ax[0].set_title("GT visualization")

        ax[1].scatter(x_test_2D[:, 0], x_test_2D[:, 1], s=10, cmap='tab10', c=all_pred)
        ax[1].set_title("Pred visualization")

        fig.tight_layout()
        plt.savefig(self.config.base_path+'result/'+ self.config.visualize_file_name + '.png')