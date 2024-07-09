import numpy as np
import copy

def normalize(v):
    return v/np.linalg.norm(v)

def world_to_camera(camera_pos, camera_lookat_pos):
    up = np.array([0,1,0])
    R = np.diag(np.ones(4))

    forward_ = normalize(camera_lookat_pos-camera_pos)  #z  v
    right_ = normalize(np.cross(up,forward_))          #x  r
    up_ = normalize(np.cross(forward_,right_))     #y  u  

    # R[2,:3] = normalize(camera_lookat_pos-camera_pos)  #z  v
    # R[0,:3] = normalize(np.cross(up,R[2,:3]))          #x  r
    # R[1,:3] = normalize(np.cross(R[2,:3],R[0,:3]))     #y  u  


    T = np.diag(np.ones(4))
    T[:3,3] = -camera_pos

    R[0,:3] = right_
    R[1,:3] = up_
    R[2,:3] = forward_

    return R.dot(T)

def camera_to_world(camera_pos, camera_lookat_pos):
    return np.linalg.inv(world_to_camera(camera_pos,camera_lookat_pos))


if __name__ == '__main__':
    vert = np.random.rand(10,3)
    camera_pos = np.array([0,0,10])
    camera_lookat_pos = np.array([0,0,0])

    vert_in_camera = copy.deepcopy(vert)
    vert_in_camera[...,2] = 10 - vert_in_camera[...,2]
    vert_in_camera[...,0] = - vert_in_camera[...,0]

    vert_in_camera2 = np.concatenate([copy.deepcopy(vert),np.ones([10,1])],axis=1)

    w2c = world_to_camera(camera_pos,camera_lookat_pos)
    vert_in_camera2_result = w2c @ vert_in_camera2.transpose(1,0)

    vert_in_camera2_result = vert_in_camera2_result[0:3,:].transpose(1,0)

    print(np.max(np.abs(vert_in_camera-vert_in_camera2_result)))
