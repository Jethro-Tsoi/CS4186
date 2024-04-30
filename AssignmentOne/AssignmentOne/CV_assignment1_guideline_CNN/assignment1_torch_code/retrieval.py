# Retrieve the most similar images by measuring the similarity between features.
import numpy as np
import os
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

top_no_matching_images = 5
# base_path = '/Users/jethrotsoi/git/CS4186/AssignmentOne/AssignmentOne/CV_assignment1_guideline_CNN/assignment1_torch_code'
base_path = '/Users/jethrotsoi/git/CS4186/AssignmentOne'
query_path = './CS4186_dataset/query_img_4186'
feat_savedir = './CNN/data/gallery_feature'
gallery_dir = './CS4186_dataset/gallery_4186'
query_path = os.path.join(base_path, query_path)
feat_savedir = os.path.join(base_path, feat_savedir)
gallery_dir = os.path.join(base_path, gallery_dir)

# Measure the similarity scores between query feature and gallery features.
# You could also use other metrics to measure the similarity scores between features.
def similarity(query_feat, gallery_feat):
    sim = cosine_similarity(query_feat, gallery_feat)
    sim = np.squeeze(sim)
    return sim

def retrival_idx(query_path: str):#, gallery_dir):
    query_feat = np.load(query_path)
    dict = {}
    for gallery_file in os.listdir(gallery_dir):
        # gallery_feat = np.load(os.path.join(gallery_dir, gallery_file))
        # gallery_idx = gallery_file.split('.')[0] + '.jpg'
        # sim = similarity(query_feat, gallery_feat)
        # dict[gallery_idx] *= sim
        feat_name = gallery_file.split('.')[0] + '.npy'
        resnet_feat = np.load(os.path.join(feat_savedir, 'resnet', feat_name))
        efficientnet_feat = np.load(os.path.join(feat_savedir, 'efficientnet', feat_name))
        resnet_sim = similarity(query_feat, resnet_feat)
        efficientnet_sim = similarity(query_feat, efficientnet_feat)
        sim = resnet_sim * efficientnet_sim
        # sim = (resnet_sim + efficientnet_sim) / 2
        dict[gallery_file] = sim
    sorted_dict = sorted(dict.items(), key=lambda item: item[1]) # Sort the similarity score
    best_five = sorted_dict[-top_no_matching_images:] # Get the best five retrived images
    return best_five

def visulization(retrived, query):
    plt.subplot(2, 3, 1)
    plt.title('query')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:,:,::-1]
    plt.imshow(img_rgb_rgb)
    for i in range(top_no_matching_images):
        img_path = './data/gallery/' + retrived[i][0]
        img_path = os.path.join(base_path, img_path)

        img = cv2.imread(img_path)
        img_rgb = img[:,:,::-1]
        plt.subplot(2, 3, i+2)
        plt.title(retrived[i][1])
        plt.imshow(img_rgb)
    plt.show()

if __name__ == '__main__':
    for query_file in os.listdir(query_path):
        query_file_path = os.path.join(query_path, query_file)
        best_five = retrival_idx(query_file_path) # retrieve top 5 matching images in the gallery.
        best_five.reverse()
        visulization(best_five, query_file_path) # Visualize the retrieval results
    # best_five = retrival_idx(query_path) # retrieve top 5 matching images in the gallery.
    # best_five.reverse()
    # # query_path = './data/query/query.jpg'
    # # query_path = os.path.join(base_path, query_path)
    # # print(query_path)

    # visulization(best_five, query_path) # Visualize the retrieval results

