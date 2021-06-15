import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import numpy as np
import asyncio

import properties as prop
from PIL import Image
import style_transformer as st_tr

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()#это константа. Убираем ее из дерева вычеслений
        self.loss = F.mse_loss(self.target, self.target )#to initialize with something

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input



def gram_matrix(input):
    batch_size , h, w, f_map_num = input.size()  # batch size(=1)
    # b=number of feature maps
    # (h,w)=dimensions of a feature map (N=h*w)

    features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(batch_size * h * w * f_map_num)



class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)# to initialize with something

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input



class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1) # torch.tensor(mean).view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1) # torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std



def normalize_and_cut_cnn(cnn, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                cnn_normalization_mean=torch.tensor([0.485, 0.456, 0.406]),
                cnn_normalization_std=torch.tensor([0.229, 0.224, 0.225]),
                content_layers=['conv_4'],
                style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = cnn.features.to(device).eval()

    # normalization module
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0 # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            #Переопределим relu уровень
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise ValueError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)


    # now we remove the unnecessary layers
    all_loss_layers = np.unique(np.concatenate([content_layers, style_layers]))
    is_passed = np.zeros([len(all_loss_layers)])
    num_of_layers = 0

    for name_of_layer, _ in model.named_modules():
        # checking if we need one more layer in model
        if not (is_passed == 1).all():
            # we need one more layer
            num_of_layers += 1
            if name_of_layer in all_loss_layers:
                is_passed[np.where(all_loss_layers == name_of_layer)] = 1 
        else:
            break
        
    model = model[:num_of_layers-1]
    return model


async def get_model_and_losses_from_cut_cnn(modified_cnn, style_img, content_img,
                                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                    content_layers=['conv_4'],
                                    style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    modified_cnn = modified_cnn.to(device).eval()

    model = nn.Sequential()
    content_losses = []
    style_losses = []

    i = 0  # increment every time we see a conv
    for name, layer in modified_cnn.named_children():
        model.add_module(name, layer)

        if isinstance(layer, nn.Conv2d):
            i += 1

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    return model, style_losses, content_losses


async def get_style_model_and_losses_from_full_cnn(input_cnn, style_img, content_img,
                                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                    cnn_normalization_mean=torch.tensor([0.485, 0.456, 0.406]),
                                    cnn_normalization_std=torch.tensor([0.229, 0.224, 0.225]),
                                    content_layers=['conv_4'],
                                    style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(input_cnn)
    cnn = cnn.features.to(device).eval()

    # normalization module
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            #Переопределим relu уровень
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise ValueError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    #выбрасываем все уровни после последнего style loss или content loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses



'''
model = normalize_and_cut_cnn(models.vgg19(pretrained=True))
torch.save(model, "cut_vgg19.pth")
'''

