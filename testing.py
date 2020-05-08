import torch.nn as nn                        # Import neural net module from pytorch
import torch.nn.functional as F              # Import functional interface from pytorch
import torch

from tqdm import tqdm

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