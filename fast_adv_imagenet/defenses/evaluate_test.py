
import argparse
import glob
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from progressbar import *
from torchvision import transforms
import sys
#from fast_adv.models.cifar10 import wide_resnet
from fast_adv.models.cifar10.model_mixed_attention import wide_resnet2
from fast_adv.models.cifar10.model_attention import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN
import scipy.misc
import numpy as np


sys.path.append("..")



class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        image = self.transformer(scipy.misc.imread(image_path))#
        #b=Image.fromarray(jpeg_compression(image_path))
        #image = self.transformer(b)
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename': os.path.basename(image_path)
        }
        return sample

def load_data_for_defense(img_size=32,batch_size=1):
    input_data = [
        #'../data/cifar10/white/Deepfool/*/*.png']
        #'../data/cifar10/white/CW/*/*.png']
        #'../data/cifar10/grey/1_PGD/*/*.png']
        '../data/cifar10/white/1_AT_PGD/*/*.png']
        #'../data/cifar10/white/1_AT_PGD_NEW/*/*.png']


        #'../data/cifar10/white/0.5_MIX_PGD/*/*.png']


    all_imgs = []
    all_labels = []
    #for input_dir in jpg_data:
    for input_dir in input_data:
        one_imgs = glob.glob(input_dir)  # (os.path.join(input_dir, './*/*.jpg'))
        one_labels = [int(img_path.split('/')[-2]) for img_path in one_imgs]
        all_imgs.extend(one_imgs)
        all_labels.extend(one_labels)
    print(len(all_labels))
    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    # print(all_labels)
    transformer = transforms.Compose([

        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                     std=[0.5, 0.5, 0.5]),
    ])
    datasets = {
        'dev_data': ImageSet(dev_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=False) for ds in datasets.keys()
        }
    return dataloaders
    #
    # path=os.path.join(input_dir, '/*/*.jpg')
    # print(path)





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='test_FGSM', help='path to dataset')
   # parser.add_argument('--input_dir', default='jpg_test_PGD',
                     #   help='Input directory with images.', type=str)
    parser.add_argument('--output_file', default='output.csv',
                        help='Output file to save labels', type=str)
    parser.add_argument('--target_model', default='densenet161',
                        help='cnn model, e.g. , densenet121, densenet161', type=str)
    parser.add_argument('--gpu_id', default=0, nargs='+',
                        help='gpu ids to use, e.g. 0 1 2 3', type=int)
    parser.add_argument('--batch_size', default=128,
                        help='batch size, e.g. 16, 32, 64...', type=int)
    return parser.parse_args()


if __name__ == '__main__':
        args = parse_args()
        gpu_ids = args.gpu_id
        if isinstance(gpu_ids, int):

            gpu_ids = [gpu_ids]
        batch_size = args.batch_size
        target_model = args.target_model
        #inputDir = args.input_dir
        outputFile = args.output_file
    ################## Defense #######################
        m = wide_resnet(num_classes=10, depth=28, widen_factor=10,
                            dropRate=0.3)  # densenet_cifar(num_classes=110)##densenet121(num_classes=110)#wide_resnet(num_classes=110, depth=28, widen_factor=10, dropRate=0.3) ######
        m2 = wide_resnet2(num_classes=10, depth=28, widen_factor=10,
                        dropRate=0.3)  # densenet_cifar(num_classes=110)##densenet121(num_classes=110)#wide_resnet(num_classes=110, depth=28, widen_factor=10, dropRate=0.3) ######

        # Loading data for ...densenet161(num_classes=110)
        image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
        device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')  # torch.device('cpu')
        model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(device)
        model2 = NormalizedModel(model=m2, mean=image_mean, std=image_std).to(device)
        #model = model.to(device)
        print('loading data for defense using %s ....' % target_model)

        test_loader = load_data_for_defense()['dev_data']
        weight_norm = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2Norm_cifar10_ep_184_val_acc0.9515.pth'
        weight_AT = '/media/unknown/Data/PLP/fast_adv/defenses/weights/cifar10_AT/cifar10acc0.8709999859333039_45.pth'
        weight_ALP = '/media/unknown/Data/PLP/fast_adv/defenses/weights/AT+ALP/cifar10acc0.8699999809265136_50.pth'

        weight_conv_mixatten = '/media/unknown/Data/PLP/fast_adv/defenses/weights/cifar10_mixed_Attention/cifar10acc0.8759999752044678_100.pth'
        weight_025conv_mixatten='/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25MixedAttention_mixed_attention_cifar10_ep_50_val_acc0.8720.pth'
        weight_05conv_mixatten = '/media/unknown/Data/PLP/fast_adv/defenses/weights/shape_0.5_cifar10_mixed_Attention/cifar10acc0.8434999763965607_130.pth'
        weight_1conv_mixatten = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/1MixedAttention_mixed_attention_cifar10_ep_25_val_acc0.7080.pth'

        weight_shape_alp='/media/unknown/Data/PLP/fast_adv/defenses/weights/best/shape_ALP_cifar10_ep_79_val_acc0.7625.pth'
        weight_attention = '/media/unknown/Data/PLP/fast_adv/defenses/weights/cifar10_Attention/cifar10acc0.8729999780654907_120.pth'

        weight_025conv_mixatten_ALP = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25Mixed+ALP_cifar10_ep_85_val_acc0.8650.pth'

        weight_smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/2random_smooth_cifar10_ep_120_val_acc0.8510.pth'
        weight_05smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/shape_0.5_random/cifar10acc0.6944999784231186_50.pth'
        weight_025smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/0.25random_smooth_cifar10_ep_146_val_acc0.8070.pth'
        weight_1smooth = '/media/unknown/Data/PLP/fast_adv/defenses/weights/best/1random_smooth_cifar10_ep_107_val_acc0.5380.pth'
        print('loading weights from : ', weight_AT)
        model_dict = torch.load(weight_AT)
        model.load_state_dict(model_dict)
        model.eval()
        model_dict2 = torch.load(weight_025conv_mixatten_ALP)
        model2.load_state_dict(model_dict2)
        model2.eval()
        test_accs = AverageMeter()
        test_losses = AverageMeter()
        widgets = ['test :', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)
        with torch.no_grad():
            for batch_data in pbar(test_loader):
                images, labels = batch_data['image'].to(device), batch_data['label_idx'].to(device)
                noise = torch.randn_like(images, device='cuda') * 0.2
                image_shape = images + noise
                #image_shape = torch.renorm(image_shape - images, p=2, dim=0, maxnorm=1) + images
                #logits,_ = model.forward_attention(images.detach(), image_shape.detach())
                logits= model(images.detach())
                logits2 = model2(images.detach())
                if logits.argmax(1) !=labels and logits2.argmax(1) ==labels:
                    i=0
                    #print(images.size())
                    ddn2 = np.transpose(images[0].detach().cpu().numpy(), (1, 2, 0))
                    # deepfool2 = np.transpose(deepfool[t].cpu().numpy(), (1, 2, 0))
                    name = '/ddn_' + str(i) + '.png'
                    i+=1
                    path='/media/unknown/Data/PLP/fast_adv/visualize/smooth_and_adv'
                    out_path = os.path.join(path, str(labels[0].cpu().numpy()))
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    # print(out_path)
                    out = out_path + name
                    scipy.misc.imsave(out, ddn2)
                #logits = model(image_shape.detach())
                test_accs.append((logits.argmax(1) == labels).float().mean().item())

        print('\nTest accuracy ',test_accs.avg)
