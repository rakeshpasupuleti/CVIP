'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import cv2
import numpy as np
import os
import sys
import math

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    detection_results: List[List[float]] = [] # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.

    
    face_boxes = face_recognition.face_locations(img, number_of_times_to_upsample=2, model='hog')
    """

    face_locations(img) method returns Returns an array of bounding boxes of human faces in a image
    Returns: A list of tuples of found face locations in css (top, right, bottom, left) order.
    
    
    """
    for loc in face_boxes:
        detection_results.append([loc[3]*1.0, loc[0]*1.0, (loc[1]-loc[3])*1.0, (loc[2]-loc[0])*1.0])

    return detection_results


def cluster_faces(imgs: Dict[str, np.ndarray], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    cluster_results: List[List[str]] = [[]] * K # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.
    face_vectors=[]
    for key in imgs:
        face_box = face_recognition.face_locations(imgs[key], number_of_times_to_upsample=1, model='hog')
        encoded_vector= face_recognition.face_encodings(imgs[key], known_face_locations=face_box, num_jitters=1, model='small')
        face_vectors.append(encoded_vector)
        

    cluster_results=K_means_clustering(imgs, K, face_vectors,200)

    
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# Your functions. (if needed)
def K_means_clustering(imgs, K,face_vectors, max_iterations):


    image_names=[]
    for key in imgs:
        image_names.append(key)
    

    data_points=np.array(face_vectors)
   
#initializing the centroids


    centroids=[]
    temp_centroids=[]
    count=0
    for point in data_points:
        
        condition=True
        for cen in centroids:
            ec_dist=np.linalg.norm((cen-point))
            condition= condition * (ec_dist > 0.5)
        if(condition):
            centroids.append(point)
            temp_centroids.append(count)
        if(len(centroids) ==K):
            break

        count +=1

   



    
    iteration=0
    flag=True

    while iteration < max_iterations and flag:
        cluster=[[] for _ in range(K)]
        

        #Assigining the points to the clusters 
        cluster_idx=[]
        for point in data_points:
            distances=[]
            for center in centroids:
                temp_dist= np.sqrt(np.sum((center-point)**2))
                
                distances.append(temp_dist)

            cluster_idx.append(np.argmin(distances))
            cluster[np.argmin(distances)].append(point)

        new_centroids=[]
        for i in range (0,len(cluster)):
            new_centroids.append(np.mean(cluster[i], axis=0)) 

                 
        centroids=new_centroids
        iteration = iteration + 1

    cluster_results=[[] for _ in range(K)]
    for idx in range(0, len(cluster_idx)):
        cluster_results[cluster_idx[idx]].append(image_names[idx])


    
    return cluster_results