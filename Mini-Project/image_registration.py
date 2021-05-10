
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math 
import argparse
import sys

parser = argparse.ArgumentParser(description = 'Code for remote satellite image registration')
parser.add_argument('--ref_img', default='R_img1.png', help='Set reference image path')
parser.add_argument('--sen_img', default='S_img1.png', help='Set sensed image path')
pass_value = parser.parse_args()

def dis(p1, p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def warpPerspectivePadded(src, dst, transf):
    src_h, src_w = src.shape[:2]
    lin_homg_pts = np.array([[0, src_w, src_w, 0], [0, 0, src_h, src_h], [1, 1, 1, 1]])

    trans_lin_homg_pts = transf.dot(lin_homg_pts)
    trans_lin_homg_pts /= trans_lin_homg_pts[2,:]

    minX = np.min(trans_lin_homg_pts[0,:])
    minY = np.min(trans_lin_homg_pts[1,:])
    maxX = np.max(trans_lin_homg_pts[0,:])
    maxY = np.max(trans_lin_homg_pts[1,:])

    # calculate the needed padding and create a blank image to place dst within
    dst_sz = list(dst.shape)
    pad_sz = dst_sz.copy() # to get the same number of channels
    pad_sz[0] = np.round(np.maximum(dst_sz[0], maxY) - np.minimum(0, minY)).astype(int)
    pad_sz[1] = np.round(np.maximum(dst_sz[1], maxX) - np.minimum(0, minX)).astype(int)
    dst_pad = np.zeros(pad_sz, dtype=np.uint8)

    # add translation to the transformation matrix to shift to positive values
    anchorX, anchorY = 0, 0
    transl_transf = np.eye(3,3)
    if minX < 0: 
        anchorX = np.round(-minX).astype(int)
        transl_transf[0,2] += anchorX
    if minY < 0:
        anchorY = np.round(-minY).astype(int)
        transl_transf[1,2] += anchorY
    new_transf = transl_transf.dot(transf)
    new_transf /= new_transf[2,2]

    dst_pad[anchorY:anchorY+dst_sz[0], anchorX:anchorX+dst_sz[1]] = dst

    warped = cv.warpPerspective(src, new_transf, (pad_sz[1],pad_sz[0]))

    return dst_pad, warped

def find_BF_matches(R_img, S_img):
	sift = cv.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(R_img,None)
	kp2, des2 = sift.detectAndCompute(S_img,None)

	bf = cv.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)

	return matches, kp1, kp2

def perform_ratio_test(matches, d_threshold):
	good_matches = []
	for m,n in matches:
		if m.distance < d_threshold*n.distance:
			good_matches.append([m])

	return good_matches

def perform_outlier_removal(R_points, S_points, bin_size=50):

	D = np.zeros((len(R_points),len(R_points)))
	for i in range(len(R_points)):
		for j in range(i+1,len(R_points)):
			temp = dis(S_points[i], S_points[j])
	     	if(temp != 0):
				D[i][j] = dis(R_points[i],R_points[j])/dis(S_points[i], S_points[j])
	max_value = np.amax(D)


	bin_size = 50
	bin_width = max_value/bin_size
	hist = {}
	for i in range(len(R_points)):
		for j in range(i+1,len(R_points)):
			temp = dis(S_points[i], S_points[j])
	     	if(temp!=0):
	       		temp = dis(R_points[i],R_points[j])/dis(S_points[i], S_points[j])
	       		idx = int(temp/(bin_width+0.000000001))
	       		if idx in hist.keys():
	         		hist[idx].append((i,j))
	       		else:
	         		hist[idx] = [(i,j)]

	max_key = -1
	max_val = 0
	for (key, val) in hist.items():
		if len(val) > max_val:
	    	max_val = len(val)
	    	max_key = key

	index = set()
	for k in hist[max_key]:
		index.add(k[0])
		index.add(k[1])

	if (len(index) < 3):
		filtered_R_points = R_points
	  	filtered_S_points = S_points
	else:
	  	filtered_R_points = []
	  	filtered_S_points = []
	  	for k in index:
	    	filtered_R_points.append(R_points[k])
	    	filtered_S_points.append(S_points[k])

	return filtered_R_points, filtered_S_points


def find_affine_matrix(filtered_R_points, filtered_S_points):
	src_pts = np.float32(filtered_S_points).reshape(-1, 1, 2)
 	dst_pts = np.float32(filtered_R_points).reshape(-1, 1, 2)
 	transformation_rigid_matrix, rigid_mask = cv.estimateAffinePartial2D(src_pts, dst_pts)
 	T = np.zeros((3,3))
 	for i in range(2):
   		for j in range(3):
     		T[i][j] = transformation_rigid_matrix[i][j]
 	T[2][2] = 1.0

 	return T


if __name__ == "__main__":

	ref_img = pass_value.ref_img
	sensed_img = pass_value.sen_img

	R_img = cv.imread(ref_img,cv.IMREAD_GRAYSCALE)          
	S_img = cv.imread(sensed_img, cv.IMREAD_GRAYSCALE) 

	reference_image = 'Reference Image'
	sensed_image = 'Sensed Image'
	feature_matching = 'Featured matched Image'
	registered_image = 'Final Registered Image'

	cv.imshow(reference_image, R_img)
	cv.imshow(sensed_image, S_img)

	matches, kpr, kps = find_BF_matches(R_img, S_img)

	print("Total matches before ratio test: {}".format(len(matches)))

	good_matches = perform_ratio_test(matches, 0.6)

	print("Total matches after ratio test: {}".format(len(good_matches)))

	img3 = cv.drawMatchesKnn(R_img,kpr,S_img,kps,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	cv.imshow(feature_matching, img3)

	R_points = [kpr[k[0].queryIdx].pt for k in good_matches]
	S_points = [kps[k[0].trainIdx].pt for k in good_matches]

	filtered_R_points, filtered_S_points = perform_outlier_removal(R_points, S_points, 50)

	print("Total matched point after outlier removal: {0}".format(len(filtered_R_points)))

	if(len(filtered_R_points) < 3):
		print("\nError: Very few matching points are found")
		print("Try increasing ratio threshold and try again")
		sys.exit()

	A_matrix = find_affine_matrix(filtered_R_points, filtered_S_points)

	print("Estimated Affine Matrix:")
	print(A_matrix)

	dst_pad, warped = warpPerspectivePadded(S_img, R_img, A_matrix)

	alpha = 0.5
	beta = 1 - alpha
	blended = cv.addWeighted(warped, alpha, dst_pad, beta, 1.0)

	cv.imshow(registered_image, blended)

	cv.waitKey(0)
	cv.destroyAllWindows()