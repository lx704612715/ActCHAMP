#!/usr/bin/env python3

import roslib
roslib.load_manifest('changepoint_detection')
import rospy 
import numpy as np
from pylab import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from changepoint_detection.srv import *
from changepoint_detection.msg import *
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Circle
from itertools import product
import pickle


def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to 
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M   
def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1,0,0), index)

    normal /= norm(normal) #Make sure the vector is normalised

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector    
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])   
def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta   
def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z       
def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]
      
def makeDetectRequest(req):
    try:
        dc = rospy.ServiceProxy('changepoint_detection/detect_changepoint_detections', Detectchangepoint_detections)  
        resp = dc(req)
        return resp
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

        
        
        
if __name__ == '__main__':
    rospy.init_node('changepoint_detection_test')
    
    #Load data - we want the relative difference between 2 object poses
    #m1 and m2 are the trajectories of obj1 and obj2 alone
    #traj1 is obj1-obj2, traj2 is obj2-obj1.  Notice a significant difference. 
    f = open('data/artic_data.txt', 'r')    
    [m1, m2, traj1, traj2] = pickle.load(f)
    traj = traj2

    #print traj1, "\nlength = ", len(traj1)

    req = Detectchangepoint_detectionsRequest()
    req.data = [DataPoint(x) for x in traj]
    req.model_type = 'changepoint_detection/ArticulationFitter'
    
    req.cp_params.len_mean = 100.0
    req.cp_params.len_sigma = 5.0
    req.cp_params.min_seg_len = 10
    req.cp_params.max_particles = 10
    req.cp_params.resamp_particles = 10
    
    resp = makeDetectRequest(req)
    
    for seg in resp.segments:
        print "Model:", seg.model_name, "   Length:", seg.last_point - seg.first_point + 1
        print "Start:", seg.first_point
        print "End:", seg.last_point
        for i in xrange(len(seg.model_params)):
            print "  ", seg.param_names[i], ":", seg.model_params[i]
        print
    
    ########### Drawing code ###########
    
    X = np.array([traj[i][0] for i in xrange(len(traj))])
    Y = np.array([traj[i][1] for i in xrange(len(traj))])
    Z = np.array([traj[i][2] for i in xrange(len(traj))])
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
        
    colors = []
    choices = ['red','green','blue','purple','yellow','black']
    i = 0
    for seg in resp.segments:
        colors += [choices[i]] * (seg.last_point - seg.first_point + 1) 
        
        params = seg.model_params
        if(seg.model_name == "rotational"):
            p = Circle((0,0), params[0], facecolor = 'None', edgecolor = choices[i], alpha = .6)
            ax.add_patch(p)
            
            q = (params[7], params[4], params[5], params[6])
            v = (0,0,1) # The default normal of a circle that we want to rotate by q
            qv = qv_mult(q,v)
            
            pathpatch_2d_to_3d(p, z=0, normal = qv)
            pathpatch_translate(p, (params[1], params[2], params[3]))
            
        if(seg.model_name == "prismatic"):
            llen = 1
            sx = params[0]
            sy = params[1]
            sz = params[2]
            px = params[3]  
            py = params[4]
            pz = params[5]  
            lx = [px-(sx*llen),px+(sx*llen)]
            ly = [py-(sy*llen),py+(sy*llen)]
            lz = [pz-(sz*llen),pz+(sz*llen)]
            ax.plot(lx,ly,lz, color=choices[i], alpha=0.6)
            
        if(seg.model_name == "rigid"):
            rx = params[0]
            ry = params[1]
            rz = params[2]
            rad = 0.0075 * 3 #3-sigma
            
            phi = np.linspace(0, 2 * np.pi, 100)
            theta = np.linspace(0, np.pi, 100)
            xm = rad * np.outer(np.cos(phi), np.sin(theta)) + rx
            ym = rad * np.outer(np.sin(phi), np.sin(theta)) + ry
            zm = rad * np.outer(np.ones(np.size(phi)), np.cos(theta)) + rz
            ax.plot_surface(xm, ym, zm, color=choices[i], alpha=0.4, linewidth=0)
                            
        i = (i+1) % len(choices)
    
    #Create equal aspect ratio that is just large enough for all points
    ax.scatter(X,Y,Z, c=colors)
    max_range = max(X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min())
    X_buffer = (max_range - (X.max()-X.min())) / 2.0
    Y_buffer = (max_range - (Y.max()-Y.min())) / 2.0
    Z_buffer = (max_range - (Z.max()-Z.min())) / 2.0
    ax.auto_scale_xyz([X.min()-X_buffer, X.max()+X_buffer], [Y.min()-Y_buffer, Y.max()+Y_buffer], [Z.min()-Z_buffer, Z.max()+Z_buffer])
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    plt.show()       

