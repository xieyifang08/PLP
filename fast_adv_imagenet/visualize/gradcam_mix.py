"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import cv2
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torchvision import models
from fast_adv.models.cifar10.model_mixed_attention import wide_resnet
#from fast_adv.models.cifar10.model_attention import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN
from misc_functions import get_example_params, save_class_activation_images

def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        pil_im.thumbnail((224, 224))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        #im_as_arr[channel] -= mean[channel]
        #im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def get_example_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    example_list = (('/media/unknown/Data/PLP/fast_adv/visualize/smooth_and_adv/0_grey.png', 1),
                    ('/media/unknown/Data/PLP/fast_adv/visualize/smooth_and_adv/0.png', 1),
                    ('/media/unknown/Data/PLP/fast_adv/visualize/smooth_and_adv/7/ddn_0.png', 1))
    img_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    # Read image
    original_image = Image.open(img_path).convert('RGB')
    # Process image
    prep_img = preprocess_image(original_image)
    # Define model
    m = wide_resnet(num_classes=10, depth=28, widen_factor=10,dropRate=0.3)
    image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
    device ='cpu'# torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')  # torch.device('cpu')
    model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(device)
    weight_norm = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2Norm_cifar10_ep_184_val_acc0.9515.pth'
    weight_AT = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2AT_cifar10_ep_13_val_acc0.8770.pth'
    weight_ALP = '/media/unknown/Data/PLP/fast_adv/defenses/weights/AT+ALP/cifar10acc0.8699999809265136_50.pth'
    weight_smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2random_smooth_cifar10_ep_120_val_acc0.8510.pth'
    weight_025smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25random_smooth_cifar10_ep_146_val_acc0.8070.pth'
    weight_05smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/shape_0.5_random/cifar10acc0.6944999784231186_50.pth'

    weight_025conv_mixatten = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25MixedAttention_mixed_attention_cifar10_ep_50_val_acc0.8720.pth'

    weight_025conv_mixatten_ALP = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25Mixed+ALP_cifar10_ep_85_val_acc0.8650.pth'

    weight=weight_025conv_mixatten_ALP
    model_dict = torch.load(weight)
    model.load_state_dict(model_dict)
    model.eval()
    #pretrained_model = models.alexnet(pretrained=True)
    return (img_path,
            prep_img,
            target_class,
            weight,
            model)


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (32, 32)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    output_cam2 = []
    for idx in class_idx:
        one=np.ones_like(weight_softmax[idx])

        cam2 = one.dot(feature_conv.reshape((nc, h * w)))
        cam2 = cam2.reshape(h, w)
        cam_img2 = (cam2 - cam2.min()) / (cam2.max() - cam2.min())  # normalize
        cam_img2 = np.uint8(255 * cam_img2)
        output_cam2.append(cv2.resize(cam_img2, size_upsample))

        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam,output_cam2







if __name__ == '__main__':
    # Get params
    target_example = 2
    (img_path, prep_img, target_class, weight,model) =\
        get_example_params(target_example)

    noise = torch.randn_like(prep_img, device='cpu') * 0.02
    #prep_img = prep_img + noise

    logits = model.forward(prep_img)
    h_x = F.softmax(logits, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    pre_class=logits.argmax(1).numpy()
    print('label:',target_class,'pre:',pre_class)
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], idx[i]))

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[80].data.detach().numpy())

    #noise = torch.randn_like(prep_img, device='cpu') * 8
    #prep_img = prep_img + noise

    features_blobs = model.feature_map(prep_img).detach().numpy()
    # generate class activation mapping for the top1 prediction
    CAMs,Feature = returnCAM(features_blobs, weight_softmax, [pre_class])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % target_class)
    clean_path='/media/unknown/Data/PLP/fast_adv/visualize/smooth_and_adv/7/ddn_0.png'
    img = cv2.imread(clean_path)
    #img=cv2.imread(img_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    heatmap2 = cv2.applyColorMap(cv2.resize(Feature[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.8 + img * 0.5
    result2 = heatmap2 * 0.8 + img * 0.5
    #name = 'CAM' + str(weight) + str(target_class) + '.png'
    #print(name)
    CAM_PATH='./smooth_and_adv/0_7_mixCAM.jpg'
    Feature_PATH='./smooth_and_adv/0_7_mixFeature.jpg'
    #path='CAM' + str(weight) + str(target_class) + '.png'
    cv2.imwrite(CAM_PATH, result)
    cv2.imwrite(Feature_PATH, result2)
