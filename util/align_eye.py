from util import meshio
import numpy as np
import os
from util.util import mean_list,write_obj,list2str
import shutil
originRightEyeSocket = {
    'up':{
        # (legacy) 'position':[-0.317677, 0.369061, 0.503918],
        'position':[-0.312527,  0.370437,  0.499466],
        'vertexIdx': 12380
    },
    'down':{
        # (legacy) 'position':[-0.320983, 0.292345, 0.473489],
        'position':[-0.316781,  0.295401,  0.467381],
        'vertexIdx': 12376
    },
    'left':{
        # (legacy) 'position':[-0.431144, 0.334249, 0.417197],
        'position':[-0.426569,  0.337997,  0.413985],
        'vertexIdx': 14378
    },
    'right':{
        # (legacy) 'position':[-0.199648, 0.320599, 0.438594],
        'position':[-0.197961,  0.320205,  0.435106],
        'vertexIdx': 13705
    }
}

originLeftEyeSocket = {
    'up':{
        # (legacy) 'position':[0.330716, 0.367276, 0.503975],
        'position':[0.323598, 0.366296, 0.49885 ],
        'vertexIdx': 7154
    },
    'down':{
        # (legacy) 'position':[0.320267, 0.290892, 0.472083],
        'position':[0.313866, 0.291965, 0.464933],
        'vertexIdx': 7217
    },
    'left':{
        # (legacy) 'position':[0.192079, 0.320808, 0.442595],
        'position':[0.188731, 0.319415, 0.439505],
        'vertexIdx': 8587
    },
    'right':{
        # (legacy) 'position':[0.433081, 0.329936, 0.405873],
        'position':[0.425498, 0.331722, 0.401567],
        'vertexIdx': 9021
    }
}

originUpTeethSocket = {
    'frontal_mid':{
        'position':[0.002131,-0.254924,0.440351],
        'vertexIdx':8026
    }
}

originBottomTeethSocket = {
    'frontal_mid':{
        'position':[0.003274,-0.453495,0.320951],
        'vertexIdx':718
    },

    'back_left':{
        'position':[0.298968,-0.285018,-0.117213],
        'vertexIdx':1607
    },
    'back_right':{
        'position':[-0.294084,-0.286575,-0.115433],
        'vertexIdx':4152
    }
}

originRightEyeSocketCenter =    (np.array(originRightEyeSocket['up']['position'])+\
                                np.array(originRightEyeSocket['down']['position'])+\
                                np.array(originRightEyeSocket['left']['position'])+\
                                np.array(originRightEyeSocket['right']['position']))/4

originLeftEyeSocketCenter =    (np.array(originLeftEyeSocket['up']['position'])+\
                                np.array(originLeftEyeSocket['down']['position'])+\
                                np.array(originLeftEyeSocket['left']['position'])+\
                                np.array(originLeftEyeSocket['right']['position']))/4

originUpTeethSocketCenter = np.array(originUpTeethSocket['frontal_mid']['position'])

originBottomTeethSocketCenter = np.array(originBottomTeethSocket['frontal_mid']['position'])



def compute_eyeball_offset(inputMesh):
    vertices = inputMesh['vertices']
    inputMeshRSC = (vertices[originRightEyeSocket['up']['vertexIdx']]+\
                                vertices[originRightEyeSocket['down']['vertexIdx']]+\
                                vertices[originRightEyeSocket['left']['vertexIdx']]+\
                                vertices[originRightEyeSocket['right']['vertexIdx']])/4
    inputMeshLSC = (vertices[originLeftEyeSocket['up']['vertexIdx']]+\
                                vertices[originLeftEyeSocket['down']['vertexIdx']]+\
                                vertices[originLeftEyeSocket['left']['vertexIdx']]+\
                                vertices[originLeftEyeSocket['right']['vertexIdx']])/4
    
    offsetR = inputMeshRSC - originRightEyeSocketCenter
    offsetL = inputMeshLSC - originLeftEyeSocketCenter
    return (offsetR + offsetL) / 2, offsetL, offsetR


def align_eyeball_from_dir(result_dir,separate=False,filterFileSuffix=('eye.obj','teeth.obj','newgame.obj','onlyface.obj','DP_neu.obj','DP_exp.obj')):
    # legacy:v1: eyeMesh = meshio.read_obj('./assets/eye_teeth/eyes.obj')
    eyeMesh = meshio.read_obj('./assets/eye_teeth/v2/combined-eye.obj')
    if separate:
        # leftEyeMesh = meshio.read_obj('./assets/eye_teeth/left-eye.obj')
        # rightEyeMesh = meshio.read_obj('./assets/eye_teeth/right-eye.obj')
        leftEyeMesh = meshio.read_obj('./assets/eye_teeth/v2/left-eye.obj')
        rightEyeMesh = meshio.read_obj('./assets/eye_teeth/v2/right-eye.obj')
    for file in os.listdir(result_dir):
        if file.endswith('.obj') and not file.endswith(filterFileSuffix):
            name = file.split('.')[0]
            mesh = meshio.read_obj(os.path.join(result_dir,file)) # keys: vertices faces texcoords texcoord_idxs
            offset, l_offset, r_offset= compute_eyeball_offset(mesh)
            if separate:
                write_obj(
                    os.path.join(result_dir,name+'_left-eye.obj'),
                    leftEyeMesh['vertices'] + l_offset,
                    leftEyeMesh['texcoords'],
                    leftEyeMesh['faces'],
                    leftEyeMesh['texcoord_idxs'],
                    os.path.join(result_dir,'eyeball_diffuse.png')
                )
                write_obj(
                    os.path.join(result_dir,name+'_right-eye.obj'),
                    rightEyeMesh['vertices'] + r_offset,
                    rightEyeMesh['texcoords'],
                    rightEyeMesh['faces'],
                    rightEyeMesh['texcoord_idxs'],
                    os.path.join(result_dir,'eyeball_diffuse.png')
                )
            else:            
                write_obj(
                    os.path.join(result_dir,name+'_eye.obj'),
                    eyeMesh['vertices'] + offset,
                    eyeMesh['texcoords'],
                    eyeMesh['faces'],
                    eyeMesh['texcoord_idxs'],
                    os.path.join(result_dir,'eyeball_diffuse.png')
                )
    shutil.copy('./assets/eye_teeth/EyeBall_diffuse_v2.png',os.path.join(result_dir,'eyeball_diffuse.png'))



def compute_teeth_offset(inputMesh):
    vertices = inputMesh['vertices']
    inputMeshUSC = (vertices[originUpTeethSocket['frontal_mid']['vertexIdx']])
    inputMeshBSC = (vertices[originBottomTeethSocket['frontal_mid']['vertexIdx']])

    
    offsetUp = inputMeshUSC - originUpTeethSocketCenter
    offsetBottom = inputMeshBSC - originBottomTeethSocketCenter
    return offsetUp, offsetBottom


def align_teeth_from_dir(result_dir,filterFileSuffix=('eye.obj','teeth.obj','newgame.obj','onlyface.obj','DP_neu.obj','DP_exp.obj')):
    upTeethMesh = meshio.read_obj('./assets/eye_teeth/up_teeth.obj')
    bottomTeethMesh = meshio.read_obj('./assets/eye_teeth/bottom_teeth.obj')
    for file in os.listdir(result_dir):
        if file.endswith('.obj') and not file.endswith(filterFileSuffix):
            name = file.split('.')[0]
            mesh = meshio.read_obj(os.path.join(result_dir,file)) # keys: vertices faces texcoords texcoord_idxs
            offsetUp, offsetBottom = compute_teeth_offset(mesh)

            write_obj(
                os.path.join(result_dir,name+'_up-teeth.obj'),
                upTeethMesh['vertices'] + offsetUp,
                upTeethMesh['texcoords'],
                upTeethMesh['faces'],
                upTeethMesh['texcoord_idxs'],
                os.path.join(result_dir,'teeth_diffuse.png')
            )
            write_obj(
                os.path.join(result_dir,name+'_bottom-teeth.obj'),
                bottomTeethMesh['vertices'] + offsetBottom,
                bottomTeethMesh['texcoords'],
                bottomTeethMesh['faces'],
                bottomTeethMesh['texcoord_idxs'],
                os.path.join(result_dir,'teeth_diffuse.png')
            )
    shutil.copy('./assets/eye_teeth/teeth_color.png',os.path.join(result_dir,'teeth_diffuse.png'))