import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import copy

import result

class ImageTransformer(object):
    def __init__(self, max_img_side_size):
        self.__max_img_side_size = max_img_side_size

    def __call__(self, content_image, style_image, height=None, width=None):
        warnings = []

        if height == None: # taking size of content_image
            _, height = content_image.size

        if width == None:
            width, _ = content_image.size

        # compression of image if its side size > max_img_side_size
        max_size = max([height, width])
        compression_coef = 0.0
        if max_size > self.__max_img_side_size:
            compression_coef = self.__max_img_side_size / max_size
            height = int(height * compression_coef)
            width = int(width * compression_coef)

        if height > self.__max_img_side_size or width > self.__max_img_side_size or height <= 0 or width <= 0:
            raise ValueError("uncorrect size of image")

        loader = transforms.Compose([
            transforms.Resize(size=[height, width]), # нормируем размер изображения
            transforms.CenterCrop(size=[height, width]),
            transforms.ToTensor()]) # превращаем в удобный формат

        return (loader(content_image).unsqueeze(0).to(torch.float),
            loader(style_image).unsqueeze(0).to(torch.float))

    @property
    def max_side_size(self):
        return self.__max_side_size
        
        

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



class StyleTransformer(object):
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 cnn=models.vgg19(pretrained=True),
                 cnn_normalization_mean=torch.tensor([0.485, 0.456, 0.406]),
                 cnn_normalization_std=torch.tensor([0.229, 0.224, 0.225]),
                 content_layers=['conv_4'],
                 style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
        self._device = device
        self._cnn = cnn.features.to(device).eval()
        self._cnn_normalization_mean = cnn_normalization_mean.to(device)
        self._cnn_normalization_std = cnn_normalization_std.to(device)
        self._content_layers = content_layers
        self._style_layers = style_layers


    def __call__(self, content_image: torch.Tensor, style_image: torch.Tensor, 
                  num_steps=100, style_weight=100000, content_weight=1):
        if self.__check_images_sizes(content_image, style_image) == False:
            raise ValueError("uncorrect size of image")

        content_image = content_image.to(self._device)
        style_image = style_image.to(self._device)
        input_image = content_image.clone().to(self._device)
        return self.__run_style_transfer(content_image, style_image, input_image, num_steps, 
                           style_weight, content_weight)
        

    def __check_images_sizes(self, content_image: torch.Tensor, style_image: torch.Tensor):
        return content_image.shape == style_image.shape
        

    def __get_style_model_and_losses(self, style_img, content_img):
        cnn = copy.deepcopy(self._cnn)

        # normalization module
        normalization = Normalization(self._cnn_normalization_mean, self._cnn_normalization_std).to(self._device)

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

            if name in self._content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self._style_layers:
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


    def __run_style_transfer(self, content_img, style_img, input_img, num_steps,
                        style_weight, content_weight):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.__get_style_model_and_losses(style_img, content_img)
        optimizer = optim.LBFGS([input_img.requires_grad_()])

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values 
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                #взвешивание ощибки
                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img


    @property
    def device(self):
        return self._device
    @property
    def cnn(self):
        return self._cnn
    @property
    def cnn_normalization_mean(self):
        return self._cnn_normalization_mean
    @property
    def cnn_normalization_std(self):
        return self._cnn_normalization_std
    @property
    def content_layers(self):
        return self._content_layers
    @property
    def style_layers(self):
        return self._style_layers

