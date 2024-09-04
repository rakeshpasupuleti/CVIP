import numpy as np
from typing import List, Tuple
import cv2

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
Please do NOT read or write any file, or show any images in your final submission! 
'''

#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    rot_xyz2XYZ = np.eye(3).astype(float)
    
    # Your implementation

    #rotate around z axis with α (the current coordinate is x’y’z’);
    rot_xyz2XYZ=rotate_around_Z(alpha,rot_xyz2XYZ)

    #rotate around x′ axis with β (the current coordinate is x”y”z”);
    rot_xyz2XYZ=rotate_around_X(beta, rot_xyz2XYZ)

    #rotate around z′′ axis with γ (the current coordinate is XYZ)
    rot_xyz2XYZ=rotate_around_Z(gamma, rot_xyz2XYZ)

    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    rot_XYZ2xyz = np.eye(3).astype(float)
    # Your implementation

    #rotate around Z axis with -γ (the current coordinate is x”y”z”)
    rot_XYZ2xyz=rotate_around_Z(-(gamma), rot_XYZ2xyz)

    #rotate around x′′ axis with -β (the current coordinate is x’y’z’);
    rot_XYZ2xyz=rotate_around_X(-(beta), rot_XYZ2xyz)

    #rotate around x’ axis with -α (the current coordinate is xyz);
    rot_XYZ2xyz=rotate_around_Z(-(alpha),rot_XYZ2xyz)
    

    # print("After performing calculations the out put od the task 2 is")
    # print(rot_XYZ2xyz)


    return rot_XYZ2xyz

"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1


#function to perform rotation around Z axis
def rotate_around_Z(angle: float, mat: np.ndarray) ->np.ndarray:

    #convert the angle to radians
    radians=np.radians(angle)

    #Declear the rotation matrix of the Z
    rot_matrix_of_Z=[ [np.cos(radians), -(np.sin(radians)), 0],
                      [np.sin(radians), np.cos(radians), 0],
                      [0, 0, 1] 
                     ]
    
    #perform matrix multiplication
    result=np.matmul(rot_matrix_of_Z, mat)

    return result
    
#function to perform rotation around X axis
def rotate_around_X(angle: float, mat: np.ndarray) ->np.ndarray:

    #convert the angle to radians
    radians=np.radians(angle)

    #Declear the rotation matrix of the X
    rot_matrix_of_X=[[1, 0, 0],
                     [0, np.cos(radians), -(np.sin(radians))],
                     [0, np.sin(radians), np.cos(radians)] 
                     ]
    
    #perform matrix multiplication
    result=np.matmul(rot_matrix_of_X, mat)

    return result







#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2], dtype=float)

    
    #convert the input image into the gray scale to reduce the computational complexity

    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #finding the chessboard corners using the findChessboardCorners method
   
    return_value, corners = cv2.findChessboardCorners(gray_image, (9,4),None)

    #Check if the corners are detected or not
    
    if return_value == True:
    
        #delete the unwanted points 
        corners = np.delete(corners, [4,13,22,31], axis=0)
        

        # Draw and display the corners
        cv2.drawChessboardCorners(image, (8,4), corners,return_value)
        image = cv2.resize(image, (1000,600))

   
    #convert the convers into 32*2 shape
    img_coord=np.resize(corners,(32,2))

    return img_coord


def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    world_coord = np.zeros([32, 3], dtype=float)

    

    # Your implementation
    index=0

    for z in range(10,41,10):
    
        for x in range(40,9,-10):
            world_coord[index]=[x,0,z]
            index+=1
            
        for y in range(10,41,10):
            world_coord[index]=[0,y,z]
            index+=1

    return world_coord


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    # Your implementation

    mat_m=find_matrix_M(img_coord,world_coord)
    
    #finding the value of Cx
    M1=[mat_m[0], mat_m[1], mat_m[2]]
    M3=[mat_m[8], mat_m[9], mat_m[10]]

    cx=np.matmul(M1,np.transpose(M3))

    #finding the value of Cy

    M2=[mat_m[4], mat_m[5], mat_m[6]]
    

    cy=np.matmul(M2,np.transpose(M3))

    fx=np.sqrt(np.matmul(M1,np.transpose(M1))-cx**2)
    fy=np.sqrt(np.matmul(M2,np.transpose(M2))-cy**2)

    
    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)

    mat=find_matrix_M(img_coord,world_coord)
    mat_m=np.array([[mat[0], mat[1], mat[2], mat[3]],
                    [mat[4], mat[5], mat[6], mat[7]], 
                    [mat[8], mat[9], mat[10], mat[11]]
                  ])
    fx, fy, cx, cy=find_intrinsic(img_coord,world_coord)

    
    mat_k=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    mat_extrinsic=np.matmul(np.linalg.inv(mat_k), mat_m)

    R = mat_extrinsic[:, :3]
    T = mat_extrinsic[:, 3]

    return R, T


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2
def find_matrix_M(img_coord: np.ndarray, world_coord: np.ndarray)-> np.ndarray:

    mat=np.zeros([64, 12], dtype=float)
    idx=0
    for i in range(0,64,2):

        mat[i]=  [world_coord[idx][0], world_coord[idx][1], world_coord[idx][2], 1, 0, 0, 0, 0, -(img_coord[idx][0]*world_coord[idx][0]), -(img_coord[idx][0]*world_coord[idx][1]), -(img_coord[idx][0]*world_coord[idx][2]), -(img_coord[idx][0])]
        mat[i+1]=[0, 0, 0, 0, world_coord[idx][0], world_coord[idx][1], world_coord[idx][2], 1, -(img_coord[idx][1]*world_coord[idx][0]), -(img_coord[idx][1]*world_coord[idx][1]), -(img_coord[idx][1]*world_coord[idx][2]), -(img_coord[idx][1])]

        idx+=1

    u, s, v = np.linalg.svd(mat, full_matrices=False) 

    mat_x=v[-1]
    lamda=np.sqrt(mat_x[8]**2+ mat_x[9]**2 + mat_x[10]**2)
    
    mat_m=mat_x/lamda
    

    
    return mat_m







#---------------------------------------------------------------------------------------------------------------------