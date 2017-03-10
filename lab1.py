# -*- coding: utf-8 -*-
#Lab1
import numpy as np
import scipy as sc
import cv2 
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math



#Load 3D Points
#Load 2d Points1
#Load 2d Points2

def readPoints(filename,flag):
    points = []
    f = open(filename,'r')
    for line in f:    
        line = line.split() 
        if line:            
            if flag == 'int':
                line = [int(i) for i in line]
            if flag == 'float':
                line = [float(i) for i in line]
            points.append(line)
    return points
    
pts2AN = readPoints("C:/cv/pts2d-norm-pic_a.txt","float")
pts3DN = readPoints("C:/cv/pts3d-norm.txt","float")        
        
pts2A = readPoints("C:/cv/pts2d-pic_a.txt","int")
pts2B = readPoints("C:/cv/pts2d-pic_b.txt","int")
pts3D = readPoints("C:/cv/pts3d.txt","float")


##############################
def createSystemForP(pts2d,pts3d):
    
    """
    Взять соответствующие точки в 2д и 3в и написать систему линейных уравнений
    """
     pts2d =  np.array ( [1.0486, -0.3645], [-1.6851, -0.4004], [-0.9437, -0.4200],
     [1.0682, 0.0699], [1.0682, 0.0699], [1.0682, 0.0699],
     [0.6077, -0.0771], [1.2543, -0.6454], [-0.2709, 0.8635],
     [-0.4571, -0.3645], [-0.7902, 0.0307], [0.7318, 0.6382],
     [-1.0580, 0.3312], [0.3464, 0.3377], [0.3137, 0.1189],
     [-0.4310, 0.0242], [-0.4310, 0.0242], [-0.4799, 0.2920],
     [-0.4799, 0.2920], [0.6109, 0.0830], [-0.4081, 0.2920],
     [-0.1109, -0.2992], [0.5129, -0.0575], [0.1406, -0.4527])
    
  pts3d = np.array ( [1.5706, -0.149, 0.2598], [-1.5282, 0.9695, 0.3802], [-0.6821, 1.2856, 0.4078],
                  [0.4124, -1.0201, -0.0915], [1.2095, 0.2812, -0.128], [0.8819, -0.8481, 0.5255], 
                  [-0.9442, -1.1583, -0.3759], [0.0415, 1.3445, 0.324], [-0.7975, 0.3017, -0.0826],
                  [-0.4329, -1.4151, -0.2774], [-1.1475, -0.0772, -0.2667], [-0.5149, -1.1784, -0.1401], 
                  [0.1993, -0.2854, -0.2114], [-0.432, 0.2143, -0.1053], [-0.7481, -0.384, -0.2408],
                  [0.8078, -0.1196, -0.2631], [-0.7605, -0.5792, -0.1936], [0.3237, 0.797, 0.217],
                  [1.3089, 0.5786, -0.1887], [1.2323, 1.4421, 0.4506] )
    
  u = pts2d([:,1])
  v = pts2d([:,2])
  
  X = pts3d([:,1])
  Y = pts3d([:,2])
  Z = pts3d([:,3])
  
  A = np.zeros(40,12)
  for i in [1:2:40]:
      j = ((i + 1) / 2)
      A([i:i + 1], [:]) = mcat
([X(j), Y(j), Z(j) [ 1, 0, 0, 0, 0] - u(j)*X(j)-u(j)*Y(j)-u(j)*Z(j)-u(j), 
  [0, 0, 0, 0] X(j), Y(j), Z(j), [1] - v(j)*X(j)-v(j)*Y(j)-v(j)*Z(j)-v(j)])
    end
        U, D, Q = np.linalg.svd(A, full_matrices=False)
    
    return A

def solveForP(A):
    
    """ Решить систему, учитывая, что Ах= 0
       Использовать SVD numpy.linalg.svd
       Для 3х4 матрицы Р
    """
    

    return P 

def KRTfromP(P):
 # факторизация первой части
#  K,R = linalg.rq(self.P[:,:3])
#
#  # К по диагонали делаем положительными
#  T = diag(sign(diag(K)))
#  if linalg.det(T) < 0:
#    T[1,1] *= -1
#
#  K = dot(K,T)
#  R = dot(T,R) # T инверсия самого себя
#  

    return (K,R,T)

def error(P,p2,p3D):
    error = 0.
    for i in range(20):
        p3H = p3D[i][0:3]
        p3H.append(1.0)
        p2h = np.dot(P,p3H)
        p2h = p2h/p2h[2]
         diff = (p2[i][0] - p2h[0])*(p2[i][0] - p2h[0]) + ((p2[i][1] - p2h[1])*(p2[i][1] - p2h[1]))
        error = error + math.sqrt(diff)
    print (error)
        
def calibrate(pts2A,pts3D):
    
    A = createSystemForP(pts2A,pts3D)
    P = solveForP(A)    
    K,R,T = KRTfromP(P)
    return (P, K, R, T)
    
#////////////////////////////////////////////////    
    
def task1():
    P,K,R,T = calibrate(pts2AN,pts3DN)
    error(P,pts2AN,pts3DN)
    P,K,R,T = calibrate(pts2A,pts3D)
    error(P,pts2A,pts3D)
    print (P)    
    print (K)
    print (R)
    print (T)


task1()
