''' import packages '''
import cv2 as cv
import numpy as np

class Stitcher:
    '''Stitching class'''
    def __init__(self):
        return

    def stitch(self, images, threshold=10.0):
        '''load images and stitch'''
        (image_b, image_a) = images
        (keypoints_a, descriptors_a) = self.orb_detect(image_a)
        (keypoints_b, descriptors_b) = self.orb_detect(image_b)
        M = self.keypoints_match(keypoints_a, keypoints_b, descriptors_a, descriptors_b, threshold)
        if M is None:
            return None
        (matches, H, status) = M
        (image_left, image_right, alter_H) = self.pos_detect(image_a, image_b, matches, keypoints_a, keypoints_b, H)
        result = cv.warpPerspective(image_right, alter_H, (image_left.shape[1] + image_right.shape[1], image_left.shape[0]))
        result[0:image_left.shape[0], 0:image_left.shape[1]] = image_left
        return result

    def orb_detect(self, image):
        '''Using ORB feature detection'''
        orb = cv.ORB_create()
        (keypoints, descriptors) = orb.detectAndCompute(image, None)
        return (keypoints, descriptors)

    def keypoints_match(self, keypoints_a, keypoints_b, descriptors_a, descriptors_b, threshold):
        '''Match keypoints from images using brute force'''
        matcher = cv.BFMatcher_create(cv.NORM_HAMMING, True)
        #Cross check is alternative to the ratio test.
        matches = matcher.match(descriptors_a, descriptors_b)
        if len(matches) > 4:
            #Generate homography matrix
            points_a = np.float32([keypoints_a[m.queryIdx].pt for m in matches])
            points_b = np.float32([keypoints_b[m.trainIdx].pt for m in matches])
            (H, status) = cv.findHomography(points_a, points_b, cv.RANSAC, threshold)
            return (matches, H, status)
        return None

    def pos_detect(self, image_a, image_b, matches, keypoints_a, keypoints_b, H):
        '''Rearrange image order'''
        prop_imga = 0
        prop_imgb = 0
        i = 0
        while i < len(matches):
            if keypoints_a[matches[i].queryIdx].pt[0] > (image_a.shape[1] / 2):
                prop_imga += 1
            if keypoints_b[matches[i].trainIdx].pt[0] > (image_b.shape[1] / 2):
                prop_imgb += 1
            i += 1

        need_relocate = False
        if (prop_imga / len(matches)) > (prop_imgb / len(matches)):
            #If input order is not equal to logic order
            image_left = image_a
            need_relocate = True
            points_a = np.float32([keypoints_a[m.queryIdx].pt for m in matches])
            points_b = np.float32([keypoints_b[m.trainIdx].pt for m in matches])
            (alter_H, status) = cv.findHomography(points_b, points_a, cv.RANSAC, 10.0)
        else:
            #If input order is equal to logic order
            image_left = image_b
            alter_H = H

        if need_relocate:
            image_right = image_b
        else:
            image_right = image_a

        return (image_left, image_right, alter_H)
