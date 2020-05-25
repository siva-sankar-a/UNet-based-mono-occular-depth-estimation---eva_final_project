import torch.nn as nn                        # Import neural net module from pytorch
import torch.nn.functional as F              # Import functional interface from pytorch
import torch

from tqdm import tqdm

import numpy as np

from kornia.losses import SSIM

class Test(object):

    def __init__(self, model, device, test_loader, writer):
        super().__init__()
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.writer = writer
    
    def step(self, epoch, regularization=None, weight_decay=0.01):
        self.model.eval()
        test_loss = 0
        correct = 0
        pbar = tqdm(self.test_loader)
        test_len = len(self.test_loader.dataset)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):

                # Move data to cpu/gpu based on input
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # Loss computation
                batch_loss = F.nll_loss(output, target, reduction='sum').item()
                test_loss += batch_loss

                # Predictions
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                # Logging - updating progress bar and summary writer
                pbar.set_description(desc= f'TEST :  epoch={epoch} test_loss={(test_loss / test_len):.5f} correct/total={correct}/{test_len} accuracy=\033[1m{(100. * correct / test_len):.2f}\033[0m')
                self.writer.add_scalar('test/batch_loss', batch_loss, epoch * test_len + batch_idx)

        test_loss /= test_len
        test_accuracy = 100. * correct / test_len
        self.writer.add_scalar('loss', test_loss, epoch)
        self.writer.add_scalar('accuracy', test_accuracy, epoch)
        return {'test_loss': test_loss, 'test_accuracy': test_accuracy }

class TestExtended(Test):

    def __init__(self, model, device, test_loader, writer):
        super().__init__(model, device, test_loader, writer)
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.writer = writer
    
    def step(self, epoch):
        self.model.eval()
        pbar = tqdm(self.test_loader)
        test_len = len(self.test_loader.dataset)
        n_test_batches = test_len / self.test_loader.batch_size

        ssim_loss = SSIM(window_size=11, reduction='mean')
        
        test_loss, cm_loss, d_loss, acc1, acc2 = 0, 0, 0, 0, 0

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

            target_mask = torch.squeeze(mask_minimal_images, 1).to(device)
            target_mask = (target_mask > 0.5).float()
            
            target_depth_map = depth_minimal_images.to(device)

            # Forward pass
            output_mask, output_depth_map = model(fg_bg_stacked)

            # Loss computation
            camel_mask_loss = F.binary_cross_entropy_with_logits(output_mask, target_mask, reduction='mean')

            c_camel_mask_loss, c_depth_map_ssim, c_depth_map_l1 = 1e1, 1e2, 1e-1

            dssim_loss = ssim_loss(output_depth_map.unsqueeze(1), target_depth_map)

            depth_l1_loss = torch.abs(output_depth_map.unsqueeze(1) - target_depth_map).mean()
            depth_map_loss = c_depth_map_ssim * dssim_loss + c_depth_map_l1 * depth_l1_loss

            batch_loss = (c_camel_mask_loss * camel_mask_loss) + \
                        depth_map_loss

            test_loss += (batch_loss / n_test_batches).item()
            cm_loss += (camel_mask_loss / n_test_batches).item()
            d_loss += (depth_map_loss / n_test_batches).item()
            
            acc1 += (torch.round(torch.sigmoid(output_mask)).eq(target_mask).float().mean() / n_test_batches).item()
            acc2 += ((1 - dssim_loss) / n_test_batches).item()
            
            # Logging - updating progress bar and summary writer
            pbar.set_description(desc= f'TEST : epoch={epoch} dssim: {dssim_loss:.5f} ' +
                                    f'mask_acc: {100 * acc1:.2f} depth_acc: {100 * acc2:.2f} ' +
                                    f'loss: {test_loss:.5f} cm_loss={cm_loss:.5f} d_loss={d_loss:.5f}')

            writer.add_scalar('test/batch_loss', batch_loss, epoch * test_len + batch_idx)
            break
        writer.add_scalar('loss', test_loss, epoch)
        return test_loss, np.mean([acc1, acc2])