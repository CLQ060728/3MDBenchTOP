import numpy as np
import cv2 as cv
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_sift_keypoints(img_path):
    img = cv.imread(img_path)
    # gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(img, None)
    
    return img, kp, sift


def get_img_sift_descriptor(img_path):
    img = cv.imread(img_path)
    sift = cv.SIFT_create()
    key_points, descriptor = sift.detectAndCompute(img, None)
    
    return key_points, descriptor


def compute_surf_keypoints(img_path):
    img = cv.imread(img_path)
    # gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    surf = cv.xfeatures2d.SURF_create(400)  # 300 - 500
    kp = surf.detect(img, None)
    
    return img, kp, surf


def get_img_surf_descriptor(img_path):
    img = cv.imread(img_path)
    surf = cv.xfeatures2d.SURF_create(400)
    surf.setExtended(True)
    key_points, descriptor = surf.detectAndCompute(img, None)
    logger.info(surf.descriptorSize())
    
    return key_points, descriptor


def save_keypoints_annotated_img(img_path, output_path, descriptor_type):
    if descriptor_type == "sift":
        img, key_points, _ = compute_sift_keypoints(img_path)
    else:
        img, key_points, _ = compute_surf_keypoints(img_path)
        
    img = cv.drawKeypoints(img, key_points, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite(output_path, img)


def save_keypoints_matching_imgs(img1_path, img2_path, output_path, descriptor_type):
    # img1 = cv.imread(img1_path)          # queryImage
    # img2 = cv.imread(img2_path)          # trainImage cv.IMREAD_GRAYSCALE
    
    # # Initiate SIFT detector
    # sift = cv.SIFT_create()
    
    # # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)
    if descriptor_type== "sift":
        img1, kp1, sift1 = compute_sift_keypoints(img1_path)
        img2, kp2, sift2 = compute_sift_keypoints(img2_path)
        kp1, des1 = sift1.compute(img1, kp1)
        kp2, des2 = sift2.compute(img2, kp2)
    else:
        img1, kp1, surf1 = compute_surf_keypoints(img1_path)
        img2, kp2, surf2 = compute_surf_keypoints(img2_path)
        kp1, des1 = surf1.compute(img1, kp1)
        kp2, des2 = surf2.compute(img2, kp2)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    # FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # index_params= dict(algorithm = FLANN_INDEX_LSH,
    #                    table_number = 12, # 12 6
    #                    key_size = 20,     # 20 12
    #                    multi_probe_level = 2) #2 1 # for ORB
    search_params = dict(checks=50)   # or pass empty dictionary checks=50
    
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i]=[1,0]
    
    draw_params = dict(matchColor = (204,255,204),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # DrawMatchesFlags_DEFAULT
    # show_length = 200 if len(matches) >= 200 else len(matches)
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    # plt.imshow(img3,)
    # plt.show()
    cv.imwrite(output_path, img3)
