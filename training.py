import torch.nn as nn                        # Import neural net module from pytorch
import torch.nn.functional as F              # Import functional interface from pytorch

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