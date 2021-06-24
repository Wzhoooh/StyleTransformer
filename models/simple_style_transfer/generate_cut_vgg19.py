import torch
import torch.nn as nn

import torchvision.models as models

import numpy as np

import cut_vgg19_code


def normalize_and_cut_cnn(cnn, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                cnn_normalization_mean=torch.tensor([0.485, 0.456, 0.406]),
                cnn_normalization_std=torch.tensor([0.229, 0.224, 0.225]),
                content_layers=['conv_4'],
                style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = cnn.features.to(device).eval()

    # normalization module
    normalization = cut_vgg19_code.Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential()
    model.add_module("norm", normalization)

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

    for name_of_layer, _ in model.named_children():
        # checking if we need one more layer in model
        if not (is_passed == 1).all():
            # we need one more layer
            num_of_layers += 1
            if name_of_layer in all_loss_layers:
                is_passed[np.where(all_loss_layers == name_of_layer)] = 1 
        else:
            break
        
    model = model[:num_of_layers]
    return model

model = normalize_and_cut_cnn(models.vgg19(pretrained=True))
torch.save(model.state_dict(), "cut_vgg19.pth")
