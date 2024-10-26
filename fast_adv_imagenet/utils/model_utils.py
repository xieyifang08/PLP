import sys

sys.path.append("..")
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
import timm
import logging


from fast_adv_imagenet.models.move_weak.fpn4observe import Fpn4observe
from fast_adv_imagenet.models.wide_resnet.resnet import wide_resnet101_2
from fast_adv_imagenet.models.self_attention.resnet import wide_resnet101_2 as wide_resnet101_2_self_attention
from fast_adv_imagenet.models.move_weak.resnet_move import wide_resnet101_2 as wide_resnet101_2_move
from fast_adv_imagenet.models.move_weak.resnet_afp import wide_resnet101_2 as wide_resnet101_2_afp
from fast_adv_imagenet.models.move_weak.resnet_dfp import wide_resnet101_2 as wide_resnet101_2_dfp
from fast_adv_imagenet.models.move_weak.resnet_anl import wide_resnet101_2 as wide_resnet101_2_anl
from fast_adv_imagenet.models.move_weak.resnet_anl_block12_onlyNonLocal import wide_resnet101_2 as wide_resnet101_2_anl_block12
from fast_adv_imagenet.models.move_weak.resnet_anl_block23_onlyNonLocal import wide_resnet101_2 as wide_resnet101_2_anl_block23
from fast_adv_imagenet.models.move_weak.resnet_anl_block34_onlyNonLocal import wide_resnet101_2 as wide_resnet101_2_anl_block34
from fast_adv_imagenet.models.move_weak.resnet_anl_block4fc_onlyNonLocal import wide_resnet101_2 as wide_resnet101_2_anl_block4fc

from fast_adv_imagenet.models.move_weak.resnet_dfp_replace_conv1 import wide_resnet101_2 as wide_resnet101_2_dfp
from fast_adv_imagenet.models.move_weak.resnet_dfp_replace_conv1_ap import wide_resnet101_2 as imagenet100_wide_resnet101_dfp_fp

from fast_adv_imagenet.models.move_weak.resnet_dfp_replace_conv1_k7 import wide_resnet101_2 as wide_resnet101_2_dfp_replace_conv1_k7
from fast_adv_imagenet.models.move_weak.resnet_mwe import wide_resnet101_2 as wide_resnet101_2_mwe
logging.basicConfig(level=logging.INFO)

def load_model(model_name, pretrained=True, weight_file="", **kwargs):
    model = None
    if 'alexnet'.__eq__(model_name):
        logging.info("load model alexnet")
        m = models.alexnet(pretrained=False)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        if pretrained:
            weight = "../defenses/weights/fast_adv_imagenet/imagenet_adv_train/alexnet/best_imagenet_ep_29_val_acc0.0010.pth"
            loaded_state_dict = torch.load(weight)
            model.load_state_dict(loaded_state_dict)


    elif 'alexnet_4att_move'.__eq__(model_name):
        logging.info("load model Fpn4observe")
        m = Fpn4observe(**kwargs)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pretrained_model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        model = pretrained_model

    elif 'vgg19'.__eq__(model_name):
        logging.info("load model vgg19")
        m = models.vgg19(pretrained=pretrained)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pretrained_model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        model = pretrained_model

    elif 'resnet50'.__eq__(model_name):
        logging.info("load model resnet50")
        m = models.resnet50(pretrained=pretrained)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pretrained_model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        model = pretrained_model

    elif 'resnet152'.__eq__(model_name):
        logging.info("load model resnet152")
        m = models.resnet152(pretrained=pretrained)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pretrained_model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        model = pretrained_model

    elif 'resnet152_ddn_jpeg'.__eq__(model_name):
        logging.info("load model resnet152_ddn_jpeg")
        m = models.resnet152(pretrained=pretrained)
        weight = './weights/jpeg_ddn_resnet152/jpeg_ddn_resnet152.pth'
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pretrained_model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        loaded_state_dict = torch.load(weight)
        pretrained_model.load_state_dict(loaded_state_dict)
        model = pretrained_model

    elif 'wide_resnet101_2_dnn'.__eq__(model_name):
        logging.info("load model wide_resnet101_2_dnn")
        m = models.wide_resnet101_2(pretrained=pretrained)
        weight = '../defenses/weights/wide_resnet101_at/cifar10acc0.9232047872340425_20.pth'
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pretrained_model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        loaded_state_dict = torch.load(weight)
        pretrained_model.load_state_dict(loaded_state_dict)
        model = pretrained_model

    elif 'wide_resnet101_2'.__eq__(model_name):
        logging.info("load model wide_resnet101_2")
        m = wide_resnet101_2(pretrained=False, **kwargs)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    # elif 'imagenet100_wide_resnet101_dfp'.__eq__(model_name):
    #     logging.info("load model imagenet100 wide_resnet101_dfp")
    #     m = wide_resnet101_2_dfp(**kwargs)
    #     image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    #     image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    #     model = NormalizedModel(model=m, mean=image_mean, std=image_std)
    #     logging.info("pretrained is{}".format(pretrained))
    #     if pretrained:
    #         weight = "../defenses/weights/best/best_imagenet100_wrn_dfp_at_ep_3_val_acc0.7094.pth"
    #         loaded_state_dict = torch.load(weight)
    #         model.load_state_dict(loaded_state_dict)

    elif 'imagenet100_wide_resnet101_dfp'.__eq__(model_name):
        logging.info("load model imagenet100_wide_resnet101_dfp")
        m = wide_resnet101_2_dfp(**kwargs)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        if pretrained:
            # weight = "../defenses/weights/best/best_imagenet100_wrn_dfp_at_ep_3_val_acc0.7094.pth"
            weight = "../defenses/weights/imagenet100_dfp_replace_conv1_rerun_at/best_imagenet100_ep_19_val_acc0.7971.pth"
            loaded_state_dict = torch.load(weight)
            model.load_state_dict(loaded_state_dict)

    elif 'imagenet100_wide_resnet101_dfp_fp'.__eq__(model_name):
        logging.info("load model imagenet100 wide_resnet101_dfp fp")
        m = imagenet100_wide_resnet101_dfp_fp(**kwargs)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'imagenet100_wide_resnet101_dfp_replace_conv1_k7'.__eq__(model_name):
        logging.info("load model imagenet100 wide_resnet101_dfp replace_conv1_k7")
        m = wide_resnet101_2_dfp_replace_conv1_k7(**kwargs)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'imagenet100_wide_resnet101_anl'.__eq__(model_name):
        logging.info("load model imagenet100_wide_resnet101_anl")
        m = wide_resnet101_2_anl(**kwargs)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        logging.info("pretrained is{}".format(pretrained))
        if pretrained:
            weight = "../defenses/weights/best/best_imagenet100_wrn_anl_at_ep_19_val_acc0.7420.pth"
            loaded_state_dict = torch.load(weight)
            model.load_state_dict(loaded_state_dict)

    elif 'imagenet100_wide_resnet101_anl_block12'.__eq__(model_name):
        logging.info("load model imagenet100_wide_resnet101_anl block12")
        m = wide_resnet101_2_anl_block12(**kwargs)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'imagenet100_wide_resnet101_anl_block23'.__eq__(model_name):
        logging.info("load model imagenet100_wide_resnet101_anl block23")
        m = wide_resnet101_2_anl_block23(**kwargs)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'imagenet100_wide_resnet101_anl_block34'.__eq__(model_name):
        logging.info("load model imagenet100_wide_resnet101_anl block34")
        m = wide_resnet101_2_anl_block34(**kwargs)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'imagenet100_wide_resnet101_anl_block4fc'.__eq__(model_name):
        logging.info("load model imagenet100_wide_resnet101_anl block4fc")
        m = wide_resnet101_2_anl_block4fc(**kwargs)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'imagenet100_wide_resnet101_mwe'.__eq__(model_name):
        logging.info("load model  wide_resnet101 mwe")
        m = wide_resnet101_2_mwe(**kwargs)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        logging.info("pretrained is{}".format(pretrained))
        if pretrained:
            weight = "../defenses/weights/imagenet100_mwe_at/best_imagenet100_ep_16_val_acc0.7448.pth"
            loaded_state_dict = torch.load(weight)
            model.load_state_dict(loaded_state_dict)

    elif 'wide_resnet101_imagenet100_backbone'.__eq__(model_name):
        logging.info("load model wide_resnet101_imagenet100_backbone")
        m = wide_resnet101_2(pretrained=False, **kwargs)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        if pretrained:
            # weight = "../defenses/weights/best/best_imagenet100_wrn_clean_ep_76_val_acc0.7773.pth"
            weight = "../defenses/weights/best/best_imagenet100_wrn_at_ep_19_val_acc0.7715.pth"
            loaded_state_dict = torch.load(weight)
            model.load_state_dict(loaded_state_dict)

    elif 'wide_resnet101_2_AT'.__eq__(model_name):
        logging.info("load model wide_resnet101_2 AT")
        m = models.wide_resnet101_2(pretrained=pretrained)
        weight = '../defenses/weights/wide_resnet101_size112_AT_finetune/best_AT_ep_8_val_acc0.7179.pth'
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pretrained_model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        loaded_state_dict = torch.load(weight)
        pretrained_model.load_state_dict(loaded_state_dict)
        model = pretrained_model

    elif 'wide_resnet101_2_MiniData_AT'.__eq__(model_name):
        logging.info("load model wide_resnet101_2 MiniData AT")
        m = models.wide_resnet101_2(pretrained=pretrained)
        # weight = '../defenses/weights/wide_resnet101_minidata_AT_finetune/cifar10acc0.8641333468755086_10.pth'
        weight = '../defenses/weights/wide_resnet101_selfAtt_minidata_AT_finetune_2/cifar10acc0.8500000143051147_20.pth'
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pretrained_model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        loaded_state_dict = torch.load(weight)
        pretrained_model.load_state_dict(loaded_state_dict)
        model = pretrained_model

    elif 'wide_resnet101_2_train'.__eq__(model_name):
        logging.info("load model wide_resnet101_2 train myself")
        m = wide_resnet101_2(pretrained=False)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'wide_resnet101_2_move'.__eq__(model_name):
        logging.info("load model wide_resnet101_2 move attention")
        m = wide_resnet101_2_move(pretrained=False)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)

    elif 'wide_resnet101_2_afp'.__eq__(model_name):
        logging.info("load model wide_resnet101_2 afp")
        m = wide_resnet101_2_afp(**kwargs)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pretrained_model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        weight = "../defenses/weights/wide_resnet101_imagenet100_backbone/backbone_imagenet100_ep_99_val_acc0.7872.pth"
        loaded_state_dict = torch.load(weight)
        pretrained_model.load_state_dict(loaded_state_dict)
        model = pretrained_model

    elif 'wide_resnet101_2_move_ft'.__eq__(model_name):
        logging.info("load model wide_resnet101_2 move attention ft")
        m = wide_resnet101_2_move(pretrained=False)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        weight = '..//defenses/weights/wide_resnet101_move_AT_finetune_b20/best_AT_ep_0_val_acc0.5457.pth'
        loaded_state_dict = torch.load(weight)
        model.load_state_dict(loaded_state_dict)

    elif 'wide_resnet101_2_self_attention'.__eq__(model_name):
        logging.info("load model wide_resnet101_2_self_attention train myself")
        m = wide_resnet101_2_self_attention(pretrained=False)
        # weight = '../defenses/weights/wide_resnet101_self_attention_finetune_tianchi_15/cifar10acc0.7500000120699406_150.pth'
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        loaded_state_dict = torch.load(weight_file)
        model.load_state_dict(loaded_state_dict)

    elif 'wide_resnet101_2_attloss'.__eq__(model_name):
        logging.info("load model wide_resnet101_2_attloss")
        m = models.wide_resnet101_2(pretrained=False)
        # weight = '../defenses/weights/jpeg_ddn_wide_resnet101/jpeg_ddn_wide_resnet101.pth'
        weight = '../defenses/weights/wide_resnet101_attention_loss/cifar10acc0.932845744680851_10.pth'
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        loaded_state_dict = torch.load(weight)
        model.load_state_dict(loaded_state_dict)

    elif 'wide_resnet101_2_dnn_jpeg'.__eq__(model_name):
        logging.info("load model wide_resnet101_2_dnn_jpeg")
        m = models.wide_resnet101_2(pretrained=False)
        # weight = '../defenses/weights/jpeg_ddn_wide_resnet101/jpeg_ddn_wide_resnet101.pth'
        weight = '../defenses/weights/wide_resnet101_at_jpeg/cifar10acc0.9251994680851063_20.pth'
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        loaded_state_dict = torch.load(weight)
        model.load_state_dict(loaded_state_dict)

    elif 'densenet161_ddn_jpeg'.__eq__(model_name):
        logging.info("load model densenet161_ddn_jpeg")

        m = models.densenet161(pretrained=pretrained)
        weight = './weights/jpeg_ddn_densenet161/jpeg_ddn_densenet161.pth'

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        loaded_state_dict = torch.load(weight)
        model.load_state_dict(loaded_state_dict)

    elif 'hrnet_w64_ddn_jpeg'.__eq__(model_name):
        logging.info("load model: hrnet_w64_ddn_jpeg")
        model = timm.create_model('hrnet_w64', pretrained=pretrained)
        image_mean = torch.tensor([0.5000, 0.5000, 0.5000]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.5000, 0.5000, 0.5000]).view(1, 3, 1, 1)
        model = NormalizedModel(model=model, mean=image_mean, std=image_std)
        weight= './weights/jpeg_ddn_hrnet_w64/jpeg_ddn_hrnet_w64.pth'
        loaded_state_dict = torch.load(weight)
        model.load_state_dict(loaded_state_dict)

    # elif 'Ensemble_dsn161_jpeg_rn162_jpeg_wrn101_jpeg'.__eq__(model_name):
    #     logging.info("load model: Ensemble_dsn161_jpeg_rn162_jpeg_wrn101_jpeg")
    #     model1 = load_model("densenet161_ddn_jpeg")
    #     model2 = load_model("resnet152_ddn_jpeg")
    #     model3 = load_model("wide_resnet101_2_dnn_jpeg")
    #     model = Ensemble(model1, model2, model3)
    # elif 'Ensemble_dsn161_jpeg_wrn101_jpeg_hrn_jpeg'.__eq__(model_name):
    #     logging.info("load model: Ensemble_dsn161_jpeg_wrn101_jpeg_hrn_jpeg")
    #     model1 = load_model("densenet161_ddn_jpeg")
    #     model2 = load_model("wide_resnet101_2_dnn_jpeg")
    #     model3 = load_model("hrnet_w64_ddn_jpeg")
    #     model = Ensemble3_hrn(model1, model2, model3)
    # elif 'Ensemble_dsn161_jpeg_wrn101_jpeg'.__eq__(model_name):
    #     logging.info("load model: Ensemble_dsn161_jpeg_wrn101_jpeg")
    #     model1 = load_model("densenet161_ddn_jpeg")
    #     model2 = load_model("wide_resnet101_2_dnn_jpeg")
    #     model = Ensemble2(model1, model2)
    # elif 'Ensemble_dsn161_jpeg_rn162_jpeg_wrn101_jpeg_hrn_jpeg'.__eq__(model_name):
    #     logging.info("load model: Ensemble_dsn161_jpeg_rn162_jpeg_wrn101_jpeg_hrn_jpeg")
    #     model1 = load_model("densenet161_ddn_jpeg")
    #     model2 = load_model("resnet152_ddn_jpeg")
    #     model3 = load_model("wide_resnet101_2_dnn_jpeg")
    #     model4 = load_model("hrnet_w64_ddn_jpeg")
    #     model = Ensemble4(model1, model2, model3,model4)
    #
    # elif 'Ensemble_dsn161_jpeg_rn162_jpeg'.__eq__(model_name):
    #     logging.info("load model: Ensemble_dsn161_jpeg_rn162_jpeg")
    #     model1 = load_model("densenet161_ddn_jpeg")
    #     model2 = load_model("resnet152_ddn_jpeg")
    #     model = Ensemble2(model1, model2)
    # elif 'Adv_Denoise_Resnext101'.__eq__(model_name):
    #     logging.info("load model: Adv_Denoise_Resnext101")
    #     m = resnet101_denoise()
    #     weight = './weights/Adv_Denoise_Resnext101.pytorch'
    #     loaded_state_dict = torch.load(weight)
    #     m.load_state_dict(loaded_state_dict, strict=True)
    #     image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    #     image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    #     model = NormalizedModel(model=m, mean=image_mean, std=image_std)
    # elif 'Adv_Denoise_Resnet152'.__eq__(model_name):
    #     logging.info("load model: Adv_Denoise_Resnet152")
    #     m = resnet152_denoise()
    #     weight = './weights/Adv_Denoise_Resnet152.pytorch'
    #     loaded_state_dict = torch.load(weight)
    #     m.load_state_dict(loaded_state_dict, strict=True)
    #     image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    #     image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    #     model = NormalizedModel(model=m, mean=image_mean, std=image_std)
    # elif 'Adv_Resnet152'.__eq__(model_name):
    #     logging.info("load model: Adv_Resnet152")
    #     m = resnet152()
    #     weight = './weights/Adv_Resnet152.pytorch'
    #     loaded_state_dict = torch.load(weight)
    #     m.load_state_dict(loaded_state_dict, strict=True)
    #     image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    #     image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    #     model = NormalizedModel(model=m, mean=image_mean, std=image_std)
    elif 'ens_adv_inception_resnet_v2'.__eq__(model_name):
        print("load model: ens_adv_inception_resnet_v2")
        model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=pretrained)
        image_mean = torch.tensor([0.5000, 0.5000, 0.5000]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.5000, 0.5000, 0.5000]).view(1, 3, 1, 1)
        model = NormalizedModel(model=model, mean=image_mean, std=image_std)
    else:
        logging.info("can not load model")

    return model


def save_checkpoint(state: OrderedDict, filename: str = 'checkpoint.pth', cpu: bool = False) -> None:
    if cpu:
        new_state = OrderedDict()
        for k in state.keys():
            newk = k.replace('module.', '')  # remove module. if model was trained using DataParallel
            new_state[newk] = state[k].cpu()
        state = new_state
    if torch.__version__ >= '1.6.0':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


class NormalizedModel(nn.Module):
    """
    Wrapper for a model to account for the mean and std of a dataset.
    mean and std do not require grad as they should not be learned, but determined beforehand.
    mean and std should be broadcastable (see pytorch doc on broadcasting) with the data.
    Args:

        model (nn.Module): model to use to predict
        mean (torch.Tensor): sequence of means for each channel
        std (torch.Tensor): sequence of standard deviations for each channel
    """

    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super(NormalizedModel, self).__init__()

        self.model = model
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return self.model(normalized_input)

    def forward_with_label(self, input, labels) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return self.model.forward_with_label(normalized_input, labels)

    def forward_at(self, input: torch.Tensor, input2) :
        normalized_input = (input - self.mean) / self.std
        normalized_input2 = (input2 - self.mean) / self.std
        return self.model.forward_at(normalized_input, normalized_input2)

    def feature_map(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return self.model.feature_map(normalized_input)
    def feature_map2(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return self.model.feature_map2(normalized_input)



def requires_grad_(model:nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()

if __name__ == '__main__':
    model = load_model("ecaresnet269d")
    # load_model("nfnet_f7")
    total = sum([param.nelement() for param in model.parameters()])
    logging.info(total)