#HOW TO INSTALL ANNACONDA: https://www.youtube.com/watch?v=YJC6ldI3hWk
#WHAT IS IMAGENET DATABASE: https://www.youtube.com/watch?v=gogV2wKKF_8
import torch
import cv2
import os
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

# base_path = '/Users/jethrotsoi/git/CS4186/AssignmentOne/AssignmentOne/CV_assignment1_guideline_CNN/assignment1_torch_code'
base_path = '/Users/jethrotsoi/git/CS4186/AssignmentOne'
gallery_dir = './CS4186_dataset/gallery_4186'
feat_savedir = './CNN/data/gallery_feature/'
query_path = './CS4186_dataset/query_img_4186'
txt_path = './CS4186_dataset/query_img_box_4186'
save_path = './CS4186_dataset/cropped_query'
featsave_path = './CNN/data/query_feat'
query_path = os.path.join(base_path, query_path)
txt_path = os.path.join(base_path, txt_path)
save_path = os.path.join(base_path, save_path)
featsave_path = os.path.join(base_path, featsave_path)
resnet_featsave_path = os.path.join(base_path, './CNN/data/query_feat/resnet')
efficientnet_featsave_path = os.path.join(base_path, './CNN/data/query_feat/efficientnet')
gallery_dir = os.path.join(base_path, gallery_dir)
feat_savedir = os.path.join(base_path, feat_savedir)


#crop the instance region. For the images containing two instances, you need to crop both of them.
def query_crop(query_path, txt_path, save_path):
    if os.path.isfile(save_path):
        return
    query_img = cv2.imread(query_path)
    query_img = query_img[:,:,::-1] #bgr2rgb
    txt = np.loadtxt(txt_path)     #load the coordinates of the bounding box
    crop = query_img[int(txt[1]):int(txt[1] + txt[3]), int(txt[0]):int(txt[0] + txt[2]), :] #crop the instance region
    cv2.imwrite(save_path, crop[:,:,::-1])  #save the cropped region
    return crop

# Note that I feed the whole image into the pretrained vgg11 model to extract the feature, which will lead to a poor retrieval performance.
# To extract more fine-grained features, you could preprocess the gallery images by cropping them using windows with different sizes and shapes.
# Hint: opencv provides some off-the-shelf tools for image segmentation.
# def vgg_11_extraction(img, featsave_path):
#     resnet_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])])
#     img_transform = resnet_transform(img) #normalize the input image and transform it to tensor.
#     img_transform = torch.unsqueeze(img_transform, 0) #Set batchsize as 1. You can enlarge the batchsize to accelerate.

#     # initialize the weights pretrained on the ImageNet dataset, you can also use other backbones (e.g. ResNet, XceptionNet, AlexNet, ...)
#     # and extract features from more than one layer.
#     vgg11 = models.vgg11(pretrained=True)
#     vgg11_feat_extractor = vgg11.features #define the feature extractor
#     vgg11_feat_extractor.eval()  #set the mode as evaluation
#     feats = vgg11(img_transform) # extract feature
#     feats_np = feats.cpu().detach().numpy() # convert tensor to numpy
#     np.save(featsave_path, feats_np) # save the feature

import torch
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet

# Initialize ResNet
resnet = models.resnet50(pretrained=True)

# Initialize EfficientNet
efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

def resnet_extraction(img, featsave_path):
    if os.path.isfile(featsave_path):
        return
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    resnet.eval()
    features = resnet(img)
    np.save(featsave_path, features.detach().numpy())

def efficientnet_extraction(img, featsave_path):
    if os.path.isfile(featsave_path):
        return
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    efficientnet.eval()
    features = efficientnet(img)
    np.save(featsave_path, features.detach().numpy())

import os
from tqdm import tqdm

def feat_extractor_gallery(gallery_dir: str):#, feat_savedir):
    for img_file in tqdm(os.listdir(gallery_dir)):
        img_path = os.path.join(gallery_dir, img_file)
        img = cv2.imread(img_path)
        img = img[:,:,::-1] #bgr2rgb
        img_resize = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) # resize the image
        # featsave_path = os.path.join(feat_savedir, img_file.split('.')[0]+'.npy')
        resnet_festsave_path = os.path.join(feat_savedir, 'resnet', img_file.split('.')[0]+'.npy')
        efficientnet_festsave_path = os.path.join(feat_savedir, 'efficientnet', img_file.split('.')[0]+'.npy')
        resnet_extraction(img_resize, resnet_festsave_path)
        efficientnet_extraction(img_resize, efficientnet_festsave_path)

# Extract the query feature
def feat_extractor_query():
    for img_file in tqdm(os.listdir(query_path)):
        query_one_file_path = os.path.join(query_path, img_file)
        txt_one_file_path = os.path.join(txt_path, img_file.split('.')[0]+'.txt')
        save_one_file_path = os.path.join(save_path, img_file)
        crop = query_crop(query_one_file_path, txt_one_file_path, save_one_file_path)
        crop_resize = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        resnet_one_featsave_path = os.path.join(resnet_featsave_path, img_file.split('.')[0]+'.npy')
        efficientnet_one_featsave_path = os.path.join(efficientnet_featsave_path, img_file.split('.')[0]+'.npy')
        resnet_extraction(crop_resize, resnet_one_featsave_path)
        efficientnet_extraction(crop_resize, efficientnet_one_featsave_path)

def main():
    feat_extractor_query()
    # gallery_dir = './data/gallery/'
    # feat_savedir = './data/gallery_feature/'
    # feat_savedir = os.path.join(base_path, feat_savedir)
    
    feat_extractor_gallery(gallery_dir)#, feat_savedir)

if __name__=='__main__':
    main()