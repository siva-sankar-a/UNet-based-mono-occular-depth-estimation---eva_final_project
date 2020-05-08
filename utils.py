import torch
from torchsummary import summary             # Import summary with pytorch
from torchviz import make_dot
from torch.autograd import Variable

def textual_summary_cifar10(model):
    summary(model, input_size=(3, 32, 32))

def graphical_summary_cifar10(model, use_cuda=True, save=True):
    random_input = torch.randn(1, 3, 32, 32).cuda() if use_cuda else torch.randn(1, 3, 32, 32) 
    model.eval()
    y = model(Variable(random_input))
    dot_graph = make_dot(y)
    if save:
        dot_graph.format = 'svg'
        dot_graph.render(f'model_architecture')
    return dot_graph

def textual_summary_tiny_imagenet(model):
    summary(model, input_size=(3, 64, 64))

def graphical_summary_tiny_imagenet(model, use_cuda=True, save=True):
    random_input = torch.randn(1, 3, 64, 64).cuda() if use_cuda else torch.randn(1, 3, 64, 64) 
    model.eval()
    y = model(Variable(random_input))
    dot_graph = make_dot(y)
    if save:
        dot_graph.format = 'svg'
        dot_graph.render(f'model_architecture')
    return dot_graph