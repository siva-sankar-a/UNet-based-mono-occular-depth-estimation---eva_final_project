import torch.nn as nn                        # Import neural net module from pytorch
import torch.nn.functional as F              # Import functional interface from pytorch
import torch

import numpy as np

from kornia.losses import SSIM

from tqdm import tqdm

class Train(object):

    def __init__(self, model, device, train_loader, optimizer, writer, scheduler=None):
        super().__init__()
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
    
    def step(self, epoch, regularization=None, weight_decay=0.01):
        self.model.train()
        train_loss = 0
        correct = 0
        pbar = tqdm(self.train_loader)
        train_len = len(self.train_loader.dataset)

        for batch_idx, (data, target) in enumerate(pbar):
            
            # Move data to cpu/gpu based on input
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            
            # Loss computation
            batch_loss = F.nll_loss(output, target)
            train_loss += batch_loss  # sum up batch loss
            

            # Regularization
            if regularization == 'L1' or regularization == 'L1 and L2':
                l1_loss = nn.L1Loss(reduction='sum')
                regularization_loss = 0
                for param in self.model.parameters():
                    regularization_loss += l1_loss(param, target=torch.zeros_like(param))
                train_loss += weight_decay * regularization_loss # regularization loss
            
            # Predictions
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Backward pass
            batch_loss.backward()
            
            # Gradient descent
            self.optimizer.step()
            if self.scheduler:
                try:
                    self.scheduler.step()
                except Exception as ex:
                    pass
                # Logging - updating progress bar and summary writer
                pbar.set_description(desc= f'TRAIN : epoch={epoch} train_loss={(train_loss / train_len):.5f} correct/total={correct}/{train_len} lr={(self.scheduler.get_last_lr()[-1]):.2f} accuracy={(100. * correct / train_len):.2f}')
            else:
                pbar.set_description(desc= f'TRAIN : epoch={epoch} train_loss={(train_loss / train_len):.5f} correct/total={correct}/{train_len} accuracy={(100. * correct / train_len):.2f}')
            self.writer.add_scalar('train/batch_loss', batch_loss, epoch * train_len + batch_idx)
        
        train_loss /= train_len
        train_accuracy = 100. * correct / train_len
        self.writer.add_scalar('loss', train_loss, epoch)
        self.writer.add_scalar('accuracy', train_accuracy, epoch)
        self.writer.add_scalar('lr', self.scheduler.get_last_lr()[-1], epoch)
        return {'train_loss': train_loss, 'train_accuracy': train_accuracy }

class TrainExtended(Train):

    def __init__(self, model, device, train_loader, optimizer, writer, scheduler=None):
        super().__init__()
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
    
    def step(self, epoch):
        self.model.train()
        self.dataset.set_train()
        train_loss = 0
        pbar = tqdm(self.train_loader)
        train_len = len(self.train_loader.dataset)
        n_train_batches = train_len / self.train_loader.batch_size

        ssim_loss = SSIM(window_size=11, reduction='mean')
        
        train_loss, cm_loss, d_loss, acc1, acc2 = 0, 0, 0, 0, 0

        for batch_idx, batch in enumerate(pbar):

            fg_bg_images = batch['fg_bg']
            bg_images = batch['bg']
            mask_images = batch['mask']
            depth_images = batch['depth']
            mask_minimal_images = batch['mask_minimal']
            depth_minimal_images = batch['depth_minimal']

            fg_bg_stacked = torch.cat([fg_bg_images, bg_images], axis=1)

            # Move data to cpu/gpu based on input
            fg_bg_stacked = fg_bg_stacked.to(device)

            optimizer.zero_grad()

            target_mask = torch.squeeze(mask_minimal_images, 1).to(device)
            target_mask = (target_mask > 0.5).float()
            
            target_depth_map = depth_minimal_images.to(device)

            # Forward pass
            output_mask, output_depth_map = model(fg_bg_stacked)

            # Loss computation

            camel_mask_loss = F.binary_cross_entropy_with_logits(output_mask, target_mask, reduction='mean')
            
            # Loss computation
            c_camel_mask_loss, c_depth_map_ssim, c_depth_map_l1 = 1e1, 1e2, 1e-1

            dssim_loss = ssim_loss(output_depth_map.unsqueeze(1), target_depth_map)

            depth_l1_loss = torch.abs(output_depth_map.unsqueeze(1) - target_depth_map).mean()
            depth_map_loss =  c_depth_map_ssim * dssim_loss + c_depth_map_l1 * depth_l1_loss

            batch_loss = (c_camel_mask_loss * camel_mask_loss) + \
                        depth_map_loss

            train_loss += (batch_loss / n_train_batches).item()
            cm_loss += (camel_mask_loss / n_train_batches).item()
            d_loss += (depth_map_loss / n_train_batches).item()
            
            # Backward pass)
            batch_loss.backward()
            
            # Gradient descent
            optimizer.step()
            
            acc1 += (torch.round(torch.sigmoid(output_mask)).eq(target_mask).float().mean() / n_train_batches).item()
            acc2 += ((1 - dssim_loss) / n_train_batches).item()


            # Step scheduler if scheduler is present
            if scheduler:
                scheduler.step()
            
            # Logging - updating progress bar and summary writer
            pbar.set_description(desc= f'TRAIN : epoch={epoch} dsssim: {dssim_loss:5f} ' +
                                    f'mask_acc: {100 * acc1:.2f} depth_acc: {100 * acc2:.2f} ' +
                                    f'loss: {train_loss:.5f} cm_loss={cm_loss:.5f} d_loss={d_loss:.5f}')

            writer.add_scalar('train/batch_loss', batch_loss, epoch * train_len + batch_idx)
            break
        writer.add_scalar('loss', train_loss, epoch)

        return train_loss, np.mean([acc1, acc2])