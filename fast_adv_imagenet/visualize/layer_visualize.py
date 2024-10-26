import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
from fast_adv.models.cifar10.model_attention import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger


def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (32, 32))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


class FeatureVisualization():
    def __init__(self,img_path,models,selected_layer):
        self.img_path=img_path
        self.selected_layer=selected_layer
        self.pretrained_model = models#.vgg16(pretrained=True).features

    def process_image(self):
        img=cv2.imread(self.img_path)
        img=preprocess_image(img)
        return img

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input=self.process_image()
        print(input.shape)
        x=input
        for index,layer in enumerate(self.pretrained_model):
            x=layer(x)
            print(index,layer)
            if (index == self.selected_layer):
                return x

    def get_single_feature(self):
        features=self.get_feature()
        print(features.shape)

        feature=features[:,0,:,:]
        print(feature.shape)

        feature=feature.view(feature.shape[1],feature.shape[2])
        print(feature.shape)

        return feature

    def save_feature_to_img(self,resize_dim):
        #to numpy
        feature=self.get_single_feature()

        feature=feature.data.numpy()

        #use sigmod to [0,1]
        feature= 1.0/(1+np.exp(-1*feature))

        # to [0,255]
        feature=np.round(feature*255)
        #print(feature[0])
        if resize_im:
            cv2im = cv2.resize(feature, (32, 32))

        cv2.imwrite('./layer.jpg',feature)




if __name__=='__main__':
    # get class
    image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
    DEVICE = torch.device('cpu')
    m = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=0.3)
    model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
    model_dict = model.state_dict()
    model_file='/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2_2AT_cifar10_ep_29_val_acc0.8870.pth'
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
    model.load_state_dict(state_dict)
    # model_dict = torch.load('/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2_1Attention_cifar10_ep_33_val_acc0.8890.pth')
    #model_dict = torch.load(
     #   '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2_2AT_cifar10_ep_29_val_acc0.8870.pth')
    #model.load_state_dict(model_dict)
    print(model)
    myClass=FeatureVisualization('./input_images/home.jpg',model,5)
    #print (myClass.pretrained_model)

    myClass.save_feature_to_img(32)
