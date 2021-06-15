import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import copy

import asyncio

import models

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
                 cut_cnn_path="cut_vgg19.pth",
                 cnn=None, #models.vgg19(pretrained=True),
                 cnn_normalization_mean=torch.tensor([0.485, 0.456, 0.406]),
                 cnn_normalization_std=torch.tensor([0.229, 0.224, 0.225]),
                 content_layers=['conv_4'],
                 style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
        self._device = device
        self._cut_cnn_path = cut_cnn_path
        self._cnn = cnn
        self._cnn_normalization_mean = cnn_normalization_mean.to(device)
        self._cnn_normalization_std = cnn_normalization_std.to(device)
        self._content_layers = content_layers
        self._style_layers = style_layers


    async def __call__(self, content_image: torch.Tensor, style_image: torch.Tensor, 
                  num_steps=100, style_weight=100000, content_weight=1):
        if self.__check_images_sizes(content_image, style_image) == False:
            raise ValueError("uncorrect size of image")

        content_image = content_image.to(self._device)
        style_image = style_image.to(self._device)
        input_image = content_image.clone().to(self._device)
        result = await self.__run_style_transfer(content_image, style_image, input_image, num_steps, 
                           style_weight, content_weight)
        return result
        

    def __check_images_sizes(self, content_image: torch.Tensor, style_image: torch.Tensor):
        return content_image.shape == style_image.shape
        

    async def __run_style_transfer(self, content_img, style_img, input_img, num_steps,
                        style_weight, content_weight):
        """Run the style transfer."""
        print('Building the style transfer model..')
        
        model = nn.Sequential()
        style_losses = []
        content_losses = []

        if self._cnn != None:
            # we download cnn into ram (acceptable only if we have many ram)
            print("Loading cnn from internet")
            model, style_losses, content_losses = await models.get_style_model_and_losses_from_full_cnn(self._cnn, style_img, content_img)
        else:
            # we load cuted cnn from file
            print("Loading cuted cnn from disk")
            model = torch.load(self._cut_cnn_path)
            model, style_losses, content_losses = await models.get_model_and_losses_from_cut_cnn(model, style_img, content_img)

        optimizer = optim.LBFGS([input_img.requires_grad_()], max_iter=1)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
            await asyncio.sleep(0) # asynchronicity :-)

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
                if run[0] % 10 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            await asyncio.sleep(0) # asynchronicity :-)
            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)
        print("Style transferring done")        

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




