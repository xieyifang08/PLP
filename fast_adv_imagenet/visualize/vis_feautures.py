from tensorflow.keras import Model
from pylab import *
import numpy as np
import cv2

def conv_output(model, layer_name, img):
    """Get the output of conv layer.

    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.

    Returns:
           intermediate_output: feature map.
    """
    # this is the placeholder for the input images
    input_img = model.input

    try:
        # this is the placeholder for the conv output
        out_conv = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    # get the intermediate layer model
    intermediate_layer_model = Model(inputs=input_img, outputs=out_conv)

    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(img)

    return intermediate_output[0]

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    feature_map_combination = []
    plt.figure()
    num_pic = feature_map.shape[2]  # 获取通道数（featuremap数量）
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')
        #title('feature_map_{}'.format(i))

    plt.savefig('feature_map.jpg')
    plt.show()

    # # 各个特征图按1:1
    # feature_map_sum = sum(ele for ele in feature_map_combination)
    #
    # feature_map_sum = (feature_map_sum - np.min(feature_map_sum)) / (
    #         np.max(feature_map_sum) - np.min(feature_map_sum))  # 融合后进一步归一化
    # y_predict = np.array(feature_map_sum).astype('float')
    # y_predict = np.round(y_predict, 0).astype('uint8')
    # y_predict *= 255
    # y_predict = np.squeeze(y_predict).astype('uint8')
    # cv2.imwrite("/home/gzy/cq/Image_Forgery_Detect_refined/1.jpg", y_predict)
    #
    # plt.imshow(y_predict)
    # plt.savefig("y_predict1.jpg")