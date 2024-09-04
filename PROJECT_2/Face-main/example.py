import cv2
import numpy as np
import os
import sys
import math

import face_recognition

from typing import Dict, List
from utils import show_image


a=np.array([3,4,5,6,7,8,9,10,11])
b=np.array([1,2,3,4,5,6,7,8,9])

dist=np.sqrt(np.sum((a-b)**2))

ex=[7,8,9,10,1,2,3,4,5,6]
print(np.argmin(ex))
cl=[[],[],[],[],[]]
cl[0].append(10)
print(cl)

l1=[2.3, 2.5, 2.7, 2.9, 2.10]
l2=[2.5, 2.7, 2.9, 2.10, 2.3]

print("The check is", set(l1)==set(l2))
a=[4 ,0, 0, 0, 0 ,1 ,1 ,1 ,1 ,1, 1, 4, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 0]
b=[4, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 4, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 0] 

print("The length of b is ", len(b))
print("The verification of the two lists are", a==b)
print(dist)

import random

# Specify the range and length of the list
range_start = 0
range_end = 35
list_length = 5

# Generate a list of unique random numbers
random_numbers = random.sample(range(range_start, range_end + 1), list_length)

# Print the result
# print("Random numbers:", random_numbers)

cluster_res=[['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg'], ['15.jpg', '16.jpg', '19.jpg', '23.jpg', '25.jpg', '28.jpg', '29.jpg'], ['10.jpg', '11.jpg', '12.jpg', '13.jpg', '9.jpg'], ['14.jpg', '17.jpg', '18.jpg', '20.jpg', '21.jpg', '22.jpg', '24.jpg', '26.jpg', '27.jpg', '30.jpg'], ['31.jpg', '32.jpg', '33.jpg', '34.jpg', '35.jpg', '36.jpg']]
sorted_res=[]
for i in range(0, len(cluster_res)):
    sorted_res.append(sorted(cluster_res[i]))

#cluster_res.sort()

print("The sorted form of the cluster results are", sorted_res)
# 1 to 8 cluster 1 [0 ,11, 22,30,31,32,33,34]
# 9 to 13 cluster 2 [1 ,2,3,4,35]
# 14 to 21 cluster 3 [5,6,7,8,9,10,12,13]
# 22 to 30 cluster 4 [14,15,16,17,18,19,20,21,23]
# 31 to 36 cluster 5 [24, 25,26,27,28,29]

print("True * True",True * (5>0))

 # cv2.imshow("Faces", img)  56
 # cv2.imshow("Faces", imgs[key]) 115
 # cv2.imshow("Faces", imgs[key]) 128



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
    idx=0
    for key in imgs:
        print(idx,key)
        idx +=1
        # print(key, imgs[key])
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        face_box = face_recognition.face_locations(imgs[key], number_of_times_to_upsample=1, model='hog')
        encoded_vector= face_recognition.face_encodings(imgs[key], known_face_locations=face_box, num_jitters=1, model='small')
        face_vectors.append(encoded_vector)
        # print("The size of the face vector for the image 1 is ", len(encoded_vector))
        # # Drawing  rectangles around the faces using OpenCV
        # for location in face_box:
        #     top, right, bottom, left = location
        #     cv2.rectangle(imgs[key], (left, top), (right, bottom), (0, 255, 0), 2)

        # # Display the image with faces marked
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # for i in range(0 , len(face_vectors)):
    #     print("The  face vector at loc i is ", i, "  is: ", face_vectors[i])
    # 1 to 8 cluster 1 [0 ,11, 22,30,31,32,33,34]
    # 9 to 13 cluster 2 [1 ,2,3,4,35]
    # 14 to 21 cluster 3 [5,6,7,8,9,10,12,13]
    # 22 to 30 cluster 4 [14,15,16,17,18,19,20,21,23]
    # 31 to 36 cluster 5 [24, 25,26,27,28,29]
    # test_set=[[0 ,11, 22,30,31,32,33,34],[1 ,2,3,4,35],[5,6,7,8,9,10,12,13],[14,15,16,17,18,19,20,21,23],[24, 25,26,27,28,29]]
    # test_set1=[24, 25,26,27,28,29]
    # test_set2=[0 ,11, 22,30,31,32,33,34,1 ,2,3,4,35,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,23]

    
    # fv=np.array(face_vectors)
    
    # for i in test_set1:
    #     min_val=100000
    #     max_val=-10
    #     print("\n\n\n\n\n\n\nFor the index", i)
    #     for j in test_set2:
    #         print( " The dist is :",np.linalg.norm((fv[i]-fv[j])),"Using normal ", np.sqrt(np.sum((fv[i]-fv[j])**2)) )
    #         if(np.linalg.norm((fv[i]-fv[j])) < min_val):
    #             min_val=np.linalg.norm((fv[i]-fv[j]))
    #         if(np.linalg.norm((fv[i]-fv[j]))> max_val):
    #             max_val=np.linalg.norm((fv[i]-fv[j]))

    #     print(min_val, max_val)
    #print("The values of the first cluster is")
    # for i in test_set[0]:
    #     print(face_vectors[i])
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
    
    #print("The names of the images are ", image_names)

    data_points=np.array(face_vectors)
    import random

    # Specify the range and length of the list
#     range_start = 0
#     range_end = 35
#     list_length = 5

#     # Generate a list of unique random numbers
#     rn = random.sample(range(range_start, range_end + 1), list_length)
#   #  centroids=np.array([face_vectors[2],face_vectors[10], face_vectors[16], face_vectors[24], face_vectors[32] ])
    # centroids=np.array([face_vectors[0],face_vectors[1], face_vectors[5], face_vectors[14], face_vectors[24] ])
#     #print("The initial centroids are", centroids)

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

    print("The Initial centroids are ",temp_centroids)
    print(centroids)



    
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
           # print("The distances are ", distances)
           # print(np.argmin(distances))
            cluster_idx.append(np.argmin(distances))
           # print("The temp is ", cluster_idx)
            cluster[np.argmin(distances)].append(point)

        new_centroids=[]
        for i in range (0,len(cluster)):
            new_centroids.append(np.mean(cluster[i], axis=0))     
           
        centroids=new_centroids

        # for i in range(0, len(cluster)):
        #     for val in cluster[i]:
        #         print("The index in face_vector is ", val)
        #         #cluster_results[i].append(image_names[face_vectors.index(val)])

        # print("For the iteration i the cluster results are",cluster_results )

        #print("For the iteration ", iteration, "The clister indexs are", cluster_idx)

        iteration = iteration + 1

    cluster_results=[[] for _ in range(K)]
    for idx in range(0, len(cluster_idx)):
        cluster_results[cluster_idx[idx]].append(image_names[idx])

    print("The  clusters are ", cluster_results)
    
    return cluster_results