"""This script defines the parametric 3d face model using hifi3dmm for Deep3DFaceRecon_pytorch
"""

from asyncore import write
from re import T
import numpy as np
import  torch
import torch.nn.functional as F
from scipy.io import loadmat
import os
import glob
import scipy.io
from PIL import Image
from util import tangent_normal_map
import nvdiffrast.torch as dr
from data.base_dataset import get_transform
AlignBFMCoeffDim = True              # Aligned with the dimensions of deep3drec's original prediction coefficients, the first 80 bases of id of Hifi3dmm, the first 64 bases of exp, and the first 80 bases of tex were taken

ALIGN_SCALE = 10                     # Pre-zoom to a certain scale, roughly aligned with bfm on the scale

ALIGN_EXP_SCALE = 16                 # HIFI3D的表情基尺度太大，预先缩小与形状基对齐

def perspective_projection(focal, center):
    # return p.T (N, 3) @ (3, 3) 
    return np.array([
        focal, 0, center,
        0, focal, center,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()

class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]

class SH_new:
    def __init__(self):
        pi = np.pi
        sqrt = np.sqrt
        self.basis = [
            #l=0
            0.5*sqrt(1/pi),
            #l=1
            sqrt(3/(4*pi)),sqrt(3/(4*pi)),sqrt(3/(4*pi)),
            #l=2
            0.5*sqrt(15/pi),0.5*sqrt(15/pi),0.25*sqrt(5/pi),0.5*sqrt(15/pi),0.25*sqrt(15/pi),
            #l=3
            0.25*sqrt(35/(2*pi)),0.5*sqrt(105/pi),0.25*sqrt(21/(2*pi)),0.25*sqrt(7/pi),0.25*sqrt(21/(2*pi)),0.25*sqrt(105/pi),0.25*sqrt(35/(2*pi)),
            #l=4
            3/4*sqrt(35/pi),3/4*sqrt(35/(2*pi)),3/4*sqrt(5/pi),3/4*sqrt(5/(2*pi)),3/16*sqrt(1/pi),3/4*sqrt(5/(2*pi)),3/8*sqrt(5/pi),3/4*sqrt(35/(2*pi)),3/16*sqrt(35/pi)
        ]


def parse_vt2t(model):
    if not os.path.isfile('./HIFI3D/vt2v.npz'):
        mean_shape = model['mu_shape']
        v_num = mean_shape.shape[1] // 3
        # vertex indices for each face. starts from 0. [40832,3]
        face_v = model['tri'].astype(np.int64)
        
        # texture coord [20792,2]
        vt = model['vt_list']
        vt_num = vt.shape[0]
        # face texture index [40832,3] Each face corresponds to three vt
        face_vt = model['tri_vt']

        face_num = face_v.shape[0]
        assert face_v.shape[0] == face_vt.shape[0]

        vt2v = np.zeros(vt_num) #.astype(np.int64)
        v2vt = [None] * v_num
        for i in range(v_num):
            v2vt[i] = set()
        
        for j in range(face_num):
            # v and vt correspond one to one
            v1, v2, v3 = face_v[j]
            vt1, vt2, vt3 = face_vt[j]
            vt2v[vt1] = v1
            vt2v[vt2] = v2
            vt2v[vt3] = v3

            v2vt[v1].add(vt1)
            v2vt[v2].add(vt2)
            v2vt[v3].add(vt3)
        
        vt_count = np.zeros(v_num)
        for k in range(v_num):
            vt_count[k] = len(v2vt[k])

        save_path = './HIFI3D/vt2v.npz'
        np.savez(save_path,
                vt2v = vt2v,
                v2vt = v2vt,
                vt_count = vt_count 
            )
    else:
        file = np.load('./HIFI3D/vt2v.npz', allow_pickle=True)
        vt2v = file['vt2v']
        v2vt = file['v2vt']
        vt_count = file['vt_count']

    return vt2v, v2vt, vt_count

def load_3dmm_basis(basis_path, uv_path=None, is_whole_uv=True, limit_dim=-1,used_region_tex_type=['uv','uv2k','normal2k']):
    """load 3dmm basis and other useful files.
    from https://github.com/tencent-ailab/hifi3dface
    :param basis_path:
        - *.mat, 3DMM basis path.
        - It contains shape/exp bases, mesh triangle definition and face vertex mask in bool.
    :param uv_path:
        - If is_whole_uv is set to true, then uv_path is a file path.
        - Otherwise, it is a directory to load regional UVs.
    :param is_whole_uv:
        - bool. indicate whether we use albedo whole uv bases or regional pyramid bases.
    :param limit_dim:
        - int. the number of dimension is used for the geometry bases. Default: -1, indicating using all dimensions.

    """

    basis3dmm = scipy.io.loadmat(basis_path)
    basis3dmm["keypoints"] = np.squeeze(basis3dmm["keypoints"])

    # load global uv basis
    if uv_path is not None and is_whole_uv:
        config = scipy.io.loadmat(uv_path)
        config["basis"] = config["basis"] * config["sigma"]
        config["indices"] = config["indices"].astype(np.int32)
        del config["sigma"]
        if config["basis"].shape[0] > config["basis"].shape[1]:
            config["basis"] = np.transpose(config["basis"])
        assert config["basis"].shape[0] < config["basis"].shape[1]
        basis3dmm["uv"] = config


    # load region uv basis
    if uv_path is not None and not is_whole_uv:
        if 'uv' in used_region_tex_type:
            uv_region_paths = sorted(glob.glob(os.path.join(uv_path, "*_uv512.mat")))
            uv_region_bases = {}
            for region_path in uv_region_paths:
                print("loading %s" % region_path)
                region_name = region_path.split("/")[-1].split("_uv")[0]
                region_config = scipy.io.loadmat(region_path)
                region_config["basis"] = np.transpose(
                    region_config["basis"] * region_config["sigma"]
                )
                region_config["indices"] = region_config["indices"].astype(np.int32)
                del region_config["sigma"]
                assert region_config["basis"].shape[0] < region_config["basis"].shape[1]
                uv_region_bases[region_name] = region_config
            basis3dmm["uv"] = uv_region_bases

        if 'uv2k' in used_region_tex_type:
            uv_region_paths = sorted(glob.glob(os.path.join(uv_path, "*_uv.mat")))
            uv_region_bases = {}
            for region_path in uv_region_paths:
                print("loading %s" % region_path)
                region_name = region_path.split("/")[-1].split("_uv")[0]
                region_config = scipy.io.loadmat(region_path)
                region_config["basis"] = np.transpose(
                    region_config["basis"] * region_config["sigma"]
                )
                region_config["indices"] = region_config["indices"].astype(np.int32)
                del region_config["sigma"]
                assert region_config["basis"].shape[0] < region_config["basis"].shape[1]
                uv_region_bases[region_name] = region_config
            basis3dmm["uv2k"] = uv_region_bases

        if 'normal2k' in used_region_tex_type:
            normal_region_paths = sorted(glob.glob(os.path.join(uv_path, "*_normal.mat")))
            normal_region_bases = {}
            for region_path in normal_region_paths:
                print("loading %s" % region_path)
                region_name = region_path.split("/")[-1].split("_normal")[0]
                region_config = scipy.io.loadmat(region_path)
                region_config["basis"] = np.transpose(
                    region_config["basis"] * region_config["sigma"]
                )
                region_config["indices"] = region_config["indices"].astype(np.int32)
                del region_config["sigma"]
                assert region_config["basis"].shape[0] < region_config["basis"].shape[1]
                normal_region_bases[region_name] = region_config
            basis3dmm["normal2k"] = normal_region_bases

    if limit_dim > 0 and limit_dim < basis3dmm["basis_shape"].shape[0]:
        basis3dmm["basis_shape"] = basis3dmm["basis_shape"][:limit_dim, :]

    return basis3dmm


class HIFIParametricFaceModel:
    def __init__(self, 
                hifi_folder='./HIFI3D', 
                recenter=True,
                camera_distance=10.,
                init_lit=np.array([
                    0.8, 0, 0, 0, 0, 0, 0, 0, 0
                    ]),
                focal=1015.,
                center=112.,          # indicates the img size is 224
                is_train=True,
                default_shape_name='HIFI3D++.mat',
                default_texture_name='AI-NExT-Albedo-Global.mat',
                uv_size = 512,
                opt_id_dim = 80,
                opt_exp_dim = 64,
                opt_tex_dim = 80,
                clamp_light_type = 'none',
                lighting_radiance_factor = 1.,
                tex_basis_factor = 1.,
                use_region_uv = False,
                used_region_tex_type = ['uv','uv2k','normal2k'],
                default_region_name = 'AI-NExT-AlbedoNormal-RPB',
                device = 'cuda',
                used_external_exp_basis='51bs_exp_basis.npy',
                use_external_exp = False
                ):
        
        if not os.path.isfile(os.path.join(hifi_folder, default_shape_name)):
            print('not a correct folder')
            exit()    

        self.use_region_uv=use_region_uv
        self.use_external_exp=use_external_exp
        self.device = device

        model = load_3dmm_basis(
        os.path.join(hifi_folder,default_shape_name),
        os.path.join(hifi_folder,default_texture_name),
        is_whole_uv=True,
        limit_dim=-1,
        )

        ## add by TJL
        vt2v, v2vt, vt_count = parse_vt2t(model)
        # print(vt2v.dtype)
        # print(v2vt.dtype) # object
        # print(vt_count.dtype)
        self.vt2v = vt2v
        # self.v2vt = v2vt
        self.vt_count = vt_count
        self.v2vt = np.zeros(len(v2vt),)
        for v_idx in range(len(v2vt)):
            self.v2vt[v_idx] = list(v2vt[v_idx])[0]



        if use_region_uv:
            self.used_region_tex_type = used_region_tex_type
            self.region_names = ['cheek', 'contour', 'eye', 'eyebrow', 'jaw', 'mouth', 'nose', 'nosetip']
            self.region_basis_dim = {'cheek':179, 'contour':39, 'eye':184,  'eyebrow': 184,'jaw': 131,'mouth': 184, 'nose': 184,'nosetip': 184}

            self.region_model = load_3dmm_basis(
            os.path.join(hifi_folder,default_shape_name),
            os.path.join(hifi_folder,default_region_name),
            is_whole_uv=False,
            limit_dim=-1,
            used_region_tex_type=used_region_tex_type,
            )
            self.region_tex_dim = 1269
        # mean face shape. [1,61443]   = xyz coordinates of 20481x3 points
        self.mean_shape = model['mu_shape'].astype(np.float32) / ALIGN_SCALE
        # identity basis. [526,61443]
        self.id_base = model['basis_shape'].astype(np.float32) / ALIGN_SCALE
        # if AlignBFMCoeffDim:
        self.id_base = self.id_base[:opt_id_dim,:]
        self.id_dim = self.id_base.shape[0]
        # expression basis. [203,61443]
        self.exp_base = model['basis_exp'].astype(np.float32) / ALIGN_SCALE
        self.exp_base = self.exp_base / ALIGN_EXP_SCALE
        # if AlignBFMCoeffDim:
        self.exp_base = self.exp_base[:opt_exp_dim,:]
        self.exp_dim = self.exp_base.shape[0]

        self.uv_size = uv_size
        # texture mean [1,248430]
        self.mean_tex = model['uv']['mu'].astype(np.float32)
        # texture basis. [294,248430]
        self.tex_base = model['uv']['basis'].astype(np.float32) * tex_basis_factor
        # if AlignBFMCoeffDim:
        self.tex_base = self.tex_base[:opt_tex_dim,:]
        self.tex_dim = self.tex_base.shape[0]
        # texture pixel indices [82810,2]
        self.tex_indices = model['uv']['indices'].astype(np.int64) # generate texture_map

        # texture coord [20792,2]      # The number of vt_list is not necessarily equal to the number of vertex, vt is the point defined on the uv map and vertex is the point defined in three-dimensional space
        self.vt = model['vt_list']
        # face texture index [40832,3]
        self.face_vt = model['tri_vt']
        self.complete_face_vt = model['tri_vt']

        
        # vertex indices for each face. starts from 0. [F,3] it possibly became only face-region triangles when training
        self.face_buf = model['tri'].astype(np.int64)
        # vertex indices for each face, starts from 0. [F,3]
        self.tri = model['tri'].astype(np.int64)
        # triangle indices for each vertex. starts from 0. [V,8],  Align to 8 bits and fill in the blanks len(self.face_buf)
        self.point_buf = self.get_point_buf(os.path.join(hifi_folder, 'hifi3d_point_buf.npy'))

        # vertex indices for landmarks. starts from 0. [86,1],hifi3d has 86 keypoints in model
        # self.keypoints = np.squeeze(model['keypoints']).astype(np.int64)

        # vertex indices for landmarks, [68,1]
        # self.keypoints = np.load(os.path.join(hifi_folder, 'hifi_lm_68.npy')).astype(np.int64)

        if is_train:
        #     # TODO
            # mask of vertex indices for small face region
            self.vertex_face_region_mask = model['mask_face'].astype(np.int64)
            face_region_tri_idx = self.get_face_tri_idx(os.path.join(hifi_folder, 'hifi3d_face_region_idx.npy')) # 把face_buf更新成面部区域的triangle
            self.face_region_tri_idx = face_region_tri_idx

        
        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

        self.persc_proj = perspective_projection(focal, center)
        self.camera_distance = camera_distance
        self.SH = SH()
        self.init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)
        self.clamp_light_type = clamp_light_type
        self.lighting_radiance_factor = lighting_radiance_factor
        transform = get_transform()
        # self.standard_normal_map = transform(Image.open(os.path.join(hifi_folder, 'tangent_normal_map.png')).convert('RGB')) *2-1 # normal map存储为图像时经过+1/2的映射

        # self.uw_exclude_mask = np.array(Image.open(os.path.join(hifi_folder,'noseend.png'))).astype(np.float32) / 255
        # self.specular_atten_map = np.array(Image.open(os.path.join(hifi_folder,'atten.png'))).astype(np.float32) / 255
        # self.specular_exclude_map = np.array(Image.open(os.path.join(hifi_folder,'exclude.png'))).astype(np.float32) / 255


        # self.uv_mask_512 = np.array(Image.open(os.path.join(hifi_folder,'uv_mask_512.png'))).astype(np.float32)
        # self.uv_mask_1024 = np.array(Image.open(os.path.join(hifi_folder,'uv_mask_1024.png'))).astype(np.float32)
        # self.uv_mask_2048 = np.array(Image.open(os.path.join(hifi_folder,'uv_mask_2048.png'))).astype(np.float32)
        self.to(self.device)

    def construct_region_uv_tensor(self, region_uv_raw, device):
        return {
            'mu': torch.tensor(region_uv_raw['mu'].astype(np.float32)).to(device),
            'basis': torch.tensor(region_uv_raw['basis'].astype(np.float32)).to(device),
            'weight': torch.tensor(region_uv_raw['weight'].astype(np.float32)).to(device),
            'indices': torch.tensor(region_uv_raw['indices'].astype(np.int64)).to(device),
        }

    def move_region_model(self,device):
        self.region_tex512 = {}
        self.region_tex2k = {}
        self.region_normal2k = {}
        for region_name in self.region_names:
            if 'uv' in self.used_region_tex_type:
                self.region_tex512[region_name] = self.construct_region_uv_tensor(self.region_model['uv'][region_name],device)
            if 'uv2k' in self.used_region_tex_type:
                self.region_tex2k[region_name] = self.construct_region_uv_tensor(self.region_model['uv2k'][region_name],device)
            if 'normal2k' in self.used_region_tex_type:
                self.region_normal2k[region_name] = self.construct_region_uv_tensor(self.region_model['normal2k'][region_name],device)

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))
        # self.standard_normal_map = self.standard_normal_map.to(device)
        if self.use_region_uv:
            self.move_region_model(device)
    def get_point_buf(self,pointbuf_path,max_degree=8):
        if os.path.exists(pointbuf_path):
            point_buf = np.load(pointbuf_path).astype(np.int64)
            return point_buf
        else:
            print('computing the point-to-face buf...')
            point_num = int(self.mean_shape.shape[1] / 3)
            face_num = self.tri.shape[0]
            point_buf = np.ones([point_num,max_degree]).astype(np.int64) * face_num
            point_buf_used = np.zeros([point_num]).astype(np.int64)
            for face_idx, tmp_face in enumerate(self.tri):
                v0_idx = tmp_face[0]
                v1_idx = tmp_face[1]
                v2_idx = tmp_face[2]
                v0_pbuf_useidx = point_buf_used[v0_idx]
                point_buf_used[v0_idx] += 1
                v1_pbuf_useidx = point_buf_used[v1_idx]
                point_buf_used[v1_idx] += 1
                v2_pbuf_useidx = point_buf_used[v2_idx]
                point_buf_used[v2_idx] += 1
                
                point_buf[v0_idx,v0_pbuf_useidx] = face_idx
                point_buf[v1_idx,v1_pbuf_useidx] = face_idx
                point_buf[v2_idx,v2_pbuf_useidx] = face_idx
            
            np.save(pointbuf_path,point_buf)
            return point_buf



    def compute_shape(self, id_coeff, exp_coeff):
        """
        Return:
            face_shape vertex position       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        # id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
        # exp_part = torch.einsum('ij,aj->ai', self.exp_base, exp_coeff)
        id_part = torch.matmul(id_coeff,self.id_base)
        exp_part = torch.matmul(exp_coeff,self.exp_base)
        face_shape = id_part + exp_part + self.mean_shape.view([1, -1])
        return face_shape.view([batch_size, -1, 3])
    
    def compute_cameraCoord_shape(self,pred_coeffs_dict):
        face_shape = self.compute_shape(pred_coeffs_dict['id'],pred_coeffs_dict['exp'])
        rotation = self.compute_rotation(pred_coeffs_dict['angle'])
        vertices_transformed = self.transform(face_shape, rotation, pred_coeffs_dict['trans'])
        rec_vertices = self.to_camera(vertices_transformed.clone())
        return rec_vertices

    def scatter_nd_torch(self,indices,updates,shape):
        batch_size = updates.shape[0]
        indices_len = indices.shape[0]
        target = torch.zeros(shape).to(self.device).type(updates.dtype)
        indices_y = indices[:,0]
        indices_x = indices[:,1]

        # batch repeat
        target = target.unsqueeze(0).repeat(batch_size,1,1,1)
        indices_batch = torch.Tensor(np.arange(0,batch_size)).long().view(batch_size,1).repeat(1,indices_len).to(self.device)     # 0,1,2...batch_size, index for batch dim
        indices_y = indices_y.unsqueeze(0).repeat(batch_size,1)
        indices_x = indices_x.unsqueeze(0).repeat(batch_size,1)
        updates = updates.to(self.device)

        return torch.index_put(target,(indices_batch, indices_y,indices_x),updates,accumulate=True)

    def compute_region_texture(self, tex_coeff, tex_type, need_mask=False):
        assert tex_type in ['tex512','tex2k','normal2k']
        uv_size = {
            'tex512':512,
            'tex2k':2048,
            'normal2k':2048
        }
        uv_model = getattr(self,'region_'+tex_type)
        tex_coeff_dict = self.split_tex_coeff(tex_coeff)
        
        res = []
        msk = []
        for r in self.region_names:
            tmp_res, tmp_msk = self.compute_single_region_tex(tex_coeff_dict[r],uv_model[r], uv_size[tex_type], need_mask=True)
            res.append(tmp_res.unsqueeze(-1))
            msk.append(tmp_msk.unsqueeze(-1))
        
        merged_tex = torch.sum(torch.cat(res,dim=-1),dim=-1)
        merged_msk = torch.sum(torch.cat(msk,dim=-1),dim=-1)
        merged_msk = torch.clamp(merged_msk,min=0,max=1)
        if need_mask:
            return merged_tex, merged_msk
        else:
            return merged_tex        

    def compute_single_region_tex(self,tex_coeff,uv_model, uv_size, need_mask=False):
        """
        Return:
            region_texture   -- torch.tensor, size (B, uv_size, uv_size, 3), in RGB order, range (0, 1.)

        Parameters:
            tex_coeff        -- torch.tensor, size (B, tex_coeff_dim)
            uv_model         dict, {['mu'],['basis'],['weight'],['indices']}
        """
        batch_size = tex_coeff.shape[0]
        face_texture = torch.matmul(tex_coeff,uv_model['basis']) + uv_model['mu']
        face_texture = face_texture.view([batch_size,-1,3])
        face_texture = face_texture * uv_model['weight']
        uv_map = self.scatter_nd_torch(uv_model['indices'].long(),face_texture,(uv_size,uv_size,3))
        if need_mask:
            uv_mask = self.scatter_nd_torch(uv_model['indices'].long(),torch.ones_like(face_texture),(uv_size,uv_size,3))
            return uv_map,uv_mask
        else:
            return uv_map


    def split_tex_coeff(self,tex_coeff):
        return {
            'cheek':    tex_coeff[:,0:179],
            'contour':  tex_coeff[:,179:218],
            'eye':      tex_coeff[:,218:402],
            'eyebrow':  tex_coeff[:,402:586],
            'jaw':      tex_coeff[:,586:717],
            'mouth':    tex_coeff[:,717:901],
            'nose':     tex_coeff[:,901:1085],
            'nosetip':  tex_coeff[:,1085:1269]
        }

    def compute_texture(self, tex_coeff, normalize=True,need_mask=False):
        """
        Return:
            face_texture     -- torch.tensor, size (B, self.uv_size, self.uv_size, 3), in RGB order, range (0, 1.)

        Parameters:
            tex_coeff        -- torch.tensor, size (B, 80)
        """
        batch_size = tex_coeff.shape[0]
        face_texture = torch.matmul(tex_coeff,self.tex_base) + self.mean_tex   #[bs,248430]
        face_texture = face_texture.view([batch_size,-1,3])
        uv_map = self.scatter_nd_torch(self.tex_indices.long(),face_texture,(self.uv_size,self.uv_size,3))
        if need_mask:
            uv_mask = self.scatter_nd_torch(self.tex_indices.long(),torch.ones_like(face_texture),(self.uv_size,self.uv_size,3))
            return uv_map,uv_mask
        else:
            return uv_map


    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)  # 顶点对应的norm法向

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)  # 顶点坐标
        """

        v1 = face_shape[:, self.tri[:, 0]]
        v2 = face_shape[:, self.tri[:, 1]]
        v3 = face_shape[:, self.tri[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)
        
        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm


    def compute_for_render_with_dp_map(self, coeffs, dp_map, use_external_exp=False):
        coeff_dict = self.split_coeff(coeffs)
        if not use_external_exp:
            face_shape = self.compute_shape(coeff_dict['id'], coeff_dict['exp'])
        else:
            face_shape = self.compute_shape_external_exp(coeff_dict['id'],coeff_dict['external_exp'])
        face_shape_dp = self.update_vertex_by_dp_map(face_shape,dp_map)

        rotation = self.compute_rotation(coeff_dict['angle'])
        face_shape_transformed = self.transform(face_shape_dp, rotation, coeff_dict['trans'])
        face_vertex = self.to_camera(face_shape_transformed) # to_camera is an inplace operation that changes face_shape_transformed
        vertex_norm = self.compute_norm(face_shape) @ rotation
        self.vertex_norm = vertex_norm
        return face_vertex, vertex_norm


    def resample(self, uv_map, texcoords, vt2v, vt_count):
        batch_size, channel_num, _, _ = uv_map.shape
        v_num = vt_count.shape[0]
        texcoords[:, 1] = 1. - texcoords[:, 1]
        texcoords = texcoords * 2 - 1
        texcoords = texcoords.type(uv_map.dtype).to(uv_map.device)
        vt2v = vt2v.long().to(uv_map.device)
        vt_count = vt_count.type(uv_map.dtype).to(uv_map.device)

        uv_grid = texcoords[None, None, :, :].expand(batch_size, -1, -1, -1)
        vt = F.grid_sample(uv_map, uv_grid, mode='bilinear') # (bs, 3, 1, num_vt)
        vt = vt.squeeze(2).permute(0, 2, 1) # (bs num_vt, 3)
        v = vt.new_zeros([batch_size, v_num, channel_num])
        v.index_add_(1, vt2v, vt)
        v = v / vt_count[None, :, None] # (bs, num_v, 3)
        return v

    def update_vertex_by_dp_map(self, vertex, dp_map):
        vertex_vt = self.vt.view(1,-1,1,2).contiguous()
        vertex_norm = self.compute_norm(vertex)
        vertex_displacement = dr.texture(dp_map.flip(1,),vertex_vt[:,self.v2vt.int(),:,:]).view(1,-1,1).contiguous()
        vertex_offset = vertex_norm * vertex_displacement
        return vertex+vertex_offset


    def compute_color(self, pix_tex, pix_norm, coeffs, white_light):
        # Spherical harmonic illumination is performed, calculating the color of each vertex according to the texture (diffuse color) and norm (normal direction) of the vertex, and the gamma coefficient of the spherical harmonic
        # In the texture uv map representation, required
        # 1. First dr.rasterize to get bary coord and face idx for each pixel
        # Based on the result of 1, calculate the texture coord for each pixel using dr.interpolate
        # 3. Calculate the norm for each vertex
        # 4. Based on the result of 1, calculate the norm of each pixel using dr.interpolate
        # 5. According to the result of 2, use the dr.texture to get the specific diffuse color of each pixel
        # 6. According to the results of 4 and 5, compute_color is used to calculate the color of each pixel under spherical harmonic illumination
        # This function performs the calculation in step 6
        """
        Return:
            pix_color                       -- torch.tensor, size (B, imgsize, imgsize, 3), range (0, 1.)

        Parameters:
            pix_tex (face_texture)          -- torch.tensor, size (B, imgsize, imgsize, 3), each pixel's diffuse RGB, range (0, 1.)
            pix_norm  (face_norm)           -- torch.tensor, size (B, imgsize, imgsize, 3), rotated face normal
            coeffs                          -- torch.tensor, size (B, coeff_len), the complete predicted coeff, which contains SH coeffs
            white_light                     -- Boolen, indicates whether use a pure white light
        """
        gamma = self.split_coeff(coeffs)['gamma']
        batch_size = gamma.shape[0]
        img_size = pix_tex.shape[1]

        face_texture = pix_tex.view(batch_size,-1,3)
        face_norm = pix_norm.view(batch_size,-1,3)

        v_num = face_texture.shape[1]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])        

        # scale the lighting radiance by the pre-defined factor, it controls how fast the lighting's radiance changes in training
        # gamma[...,0] *= self.lighting_radiance_factor 
        SHlight_scale = torch.ones([9]).to(self.device)
        SHlight_scale[0] = self.lighting_radiance_factor
        gamma = gamma * SHlight_scale
        
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),
            -a[1] * c[1] * face_norm[..., 1:2],
             a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
             a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)
        ], dim=-1)

        if white_light:
            r = Y @ gamma[..., :1]
            g = Y @ gamma[..., :1]  # assume all img with white light
            b = Y @ gamma[..., :1]
        else:
            r = Y @ gamma[..., :1]
            g = Y @ gamma[..., 1:2]
            b = Y @ gamma[..., 2:]

        if self.clamp_light_type == 'none':
            face_color = torch.cat([r, g, b], dim=-1) * face_texture
        elif self.clamp_light_type == 'sigmoid':
            sig = torch.nn.Sigmoid()
            face_color = sig(torch.cat([r, g, b], dim=-1)) * face_texture
        elif self.clamp_light_type == 'hard':
            face_color = torch.clamp(torch.cat([r, g, b], dim=-1),min=0.,max=1.) * face_texture
        elif self.clamp_light_type == 'diff':
            from util.util import dclamp
            face_color = dclamp(torch.cat([r, g, b], dim=-1),min=0.,max=1.) * face_texture
        else:
            raise NotImplementedError

        face_color = face_color.view(batch_size,img_size,img_size,3)
        return face_color


    def compute_color_old(self, face_texture, face_norm, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        v_num = face_texture.shape[1]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),
            -a[1] * c[1] * face_norm[..., 1:2],
             a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
             a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)
        ], dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1) * face_texture
        return face_color

    
    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)


    def to_camera(self, face_shape):
        face_shape_camera = face_shape.clone()
        face_shape_camera[..., -1] = self.camera_distance - face_shape_camera[..., -1]
        return face_shape_camera

    def to_world(self, face_vertex):
        face_vertex_world = face_vertex.clone()
        face_vertex_world[..., -1] = self.camera_distance - face_vertex_world[..., -1]
        return face_vertex_world

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        return face_proj


    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)


    def split_coeff(self, coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        global_tex_dim = 80
        region_tex_dim = self.region_tex_dim if self.use_region_uv else 0
        external_exp_dim = self.external_exp_dim if self.use_external_exp else 0
        id_coeffs = coeffs[:, :self.id_dim]
        exp_coeffs = coeffs[:, self.id_dim: self.id_dim+self.exp_dim]
        tex_coeffs = coeffs[:, self.id_dim+self.exp_dim: self.id_dim+self.exp_dim+global_tex_dim]
        angles = coeffs[:, self.id_dim+self.exp_dim+global_tex_dim: self.id_dim+self.exp_dim+global_tex_dim+3]
        gammas = coeffs[:, self.id_dim+self.exp_dim+global_tex_dim+3: self.id_dim+self.exp_dim+global_tex_dim+30]
        translations = coeffs[:, self.id_dim+self.exp_dim+global_tex_dim+30:self.id_dim+self.exp_dim+global_tex_dim+33]
        region_tex_coeffs = coeffs[:,self.id_dim+self.exp_dim+global_tex_dim+33:self.id_dim+self.exp_dim+global_tex_dim+33+region_tex_dim]
        external_exp_coeffs = coeffs[:,self.id_dim+self.exp_dim+global_tex_dim+33+region_tex_dim:self.id_dim+self.exp_dim+global_tex_dim+33+region_tex_dim+external_exp_dim]
        return {
            'id': id_coeffs, # 3dmm Identity coefficient
            'exp': exp_coeffs,# 3dmm Expression coefficient
            'tex': tex_coeffs, # 3dmm Texture coefficient
            'angle': angles, # Rotation Angle
            'gamma': gammas, # Spherical harmonic illumination coefficient
            'trans': translations, # Translation vector
            'region_tex':region_tex_coeffs, # Regional texture coefficient
            'external_exp': external_exp_coeffs # External exp base coefficient
        }

    def change_specific_coeff_by_order(self, coeffs, change_type, order):
        """
        permute the specific part of the coeffs by input order(index)
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, coeff_dim)
            change_type     -- string, the key of coeff
            order           -- np.array, (B,) indicates the new order
        """
        global_tex_dim = 80
        region_tex_dim = self.region_tex_dim if self.use_region_uv else 0
        external_exp_dim = self.external_exp_dim if self.use_external_exp else 0
        id_coeffs = coeffs[:, :self.id_dim]
        if change_type == 'id':
            id_coeffs = id_coeffs[order]

        exp_coeffs = coeffs[:, self.id_dim: self.id_dim+self.exp_dim]
        if change_type == 'exp':
            exp_coeffs = exp_coeffs[order]
        
        tex_coeffs = coeffs[:, self.id_dim+self.exp_dim: self.id_dim+self.exp_dim+global_tex_dim]
        if change_type == 'tex':
            tex_coeffs = tex_coeffs[order]

        angles = coeffs[:, self.id_dim+self.exp_dim+global_tex_dim: self.id_dim+self.exp_dim+global_tex_dim+3]
        gammas = coeffs[:, self.id_dim+self.exp_dim+global_tex_dim+3: self.id_dim+self.exp_dim+global_tex_dim+30]
        translations = coeffs[:, self.id_dim+self.exp_dim+global_tex_dim+30:self.id_dim+self.exp_dim+global_tex_dim+33]
        region_tex_coeffs = coeffs[:,self.id_dim+self.exp_dim+global_tex_dim+33:self.id_dim+self.exp_dim+global_tex_dim+33+region_tex_dim]
        external_exp_coeffs = coeffs[:,self.id_dim+self.exp_dim+global_tex_dim+33+region_tex_dim:self.id_dim+self.exp_dim+global_tex_dim+33+region_tex_dim+external_exp_dim]
        return torch.cat([id_coeffs,exp_coeffs,tex_coeffs,angles,gammas,translations,region_tex_coeffs,external_exp_coeffs],dim=1)

    def set_exp_coeff(self,src_coeff,tgt_coeff):
        src_coeff[:, self.id_dim: self.id_dim+self.exp_dim] = tgt_coeff[:, self.id_dim: self.id_dim+self.exp_dim]
        return src_coeff

    def set_grey_coeff(self,src_coeff):
        value = [0.6,-0.2,0.25,-0.15,0,-0.15,0,0,0]
        for i in range(9):
            src_coeff[:,self.id_dim+self.exp_dim+self.tex_dim+3+i] = value[i]
            src_coeff[:,self.id_dim+self.exp_dim+self.tex_dim+12+i] = value[i]
            src_coeff[:,self.id_dim+self.exp_dim+self.tex_dim+21+i] = value[i]
        return src_coeff


    def compute_for_render(self, coeffs, use_external_exp=False):
        '''
        According to the predicted 3dmm coefficient, calculate the vertex position (face_vertex), the predicted diffuse map (texture map) of hifi3d, the normal direction of vertex_norm of vertices, and the projected two-dimensional coordinates of key points (landmark).

        Return:
            N1 is the number of vertices
            N2 is the number of texture map vertices.

            face_vertex     -- torch.tensor, size (B, N1, 3), in camera coordinate (has been rotated)
            texture_map     -- torch.tensor, size (B, 512, 512, 3), predicted texture(diffuse) map
            vertex_norm     -- torch.tensor, size (B, N1, 3), in camera coordinate (has been rotated)
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        '''
        coef_dict = self.split_coeff(coeffs)
        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
        rotation = self.compute_rotation(coef_dict['angle'])


        face_shape_transformed = self.transform(face_shape, rotation, coef_dict['trans'])
        face_vertex = self.to_camera(face_shape_transformed)
        
        face_proj = self.to_image(face_vertex)

        if self.use_region_uv:
            face_texture = self.compute_region_texture(coef_dict['region_tex'],'tex512',need_mask=False) / 255.
        else:
            face_texture = self.compute_texture(coef_dict['tex'],need_mask=False) / 255.
        face_norm = self.compute_norm(face_shape)                                           # The variable is called face_norm, but the actual shape is the same as the vertices, and the normal of each vertex is calculated
        self.origin_face_norm = face_norm
        face_norm_roted = face_norm @ rotation                                              # Calculate the normal after rotation for subsequent shading
   
        texture_map = face_texture
        vertex_norm = face_norm_roted

        return face_vertex, texture_map, vertex_norm

    def compute_for_viewVector(self, coeffs):
        coef_dict = self.split_coeff(coeffs)
        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
        rotation = self.compute_rotation(coef_dict['angle'])
        face_shape_transformed = self.transform(face_shape, rotation, coef_dict['trans'])
        face_viewVector = F.normalize(torch.tensor([0,0,self.camera_distance]).to(self.device) - face_shape_transformed, dim=-1)
        return face_viewVector

    def compute_for_reflectViewVector(self,coeffs):
        coef_dict = self.split_coeff(coeffs)
        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
        rotation = self.compute_rotation(coef_dict['angle'])
        face_shape_transformed = self.transform(face_shape, rotation, coef_dict['trans'])
        # compute norm N
        rotated_norm = self.compute_norm(face_shape_transformed)
        # compute view V
        viewVector = F.normalize(torch.tensor([0,0,self.camera_distance]).to(self.device) - face_shape_transformed, dim=-1)
        # compute RV
        reflectViewVecotr = 2*torch.sum(rotated_norm * viewVector, dim=-1).unsqueeze(-1) * rotated_norm - viewVector
        return reflectViewVecotr, viewVector, rotated_norm



    def compute_for_TBN(self, coeffs):
        '''
        The tangent space basis of TBN is calculated based on the predicted 3dmm coefficients

        Return:
            N1 is the number of vertices

            T(tangent basis)        -- torch.tensor, size (B, N1, 3), in camera coordinate (has been rotated)
            B(bitangent basis)      -- torch.tensor, size (B, N1, 3), in camera coordinate (has been rotated)
            N(normal basis)         -- torch.tensor, size (B, N1, 3), in camera coordinate (has been rotated)
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        '''
        coef_dict = self.split_coeff(coeffs)
        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
        rotation = self.compute_rotation(coef_dict['angle'])

        self.origin_face_norm = self.compute_norm(face_shape)
        self.origin_face_tangent,self.origin_face_bitangent=tangent_normal_map.compute_TBN(face_shape, self.origin_face_norm, self.vt, self.tri, self.complete_face_vt, self.point_buf)

        rotated_norm = self.origin_face_norm @ rotation
        rotated_tangent = self.origin_face_tangent @ rotation
        rotated_bitangent = self.origin_face_bitangent @ rotation
        return rotated_tangent, rotated_bitangent, rotated_norm


    def get_face_tri_idx(self,face_tri_path):
        if os.path.exists(face_tri_path):
            new_face_buf_idx = np.load(face_tri_path).astype(np.int64)
            return new_face_buf_idx
        else:
            new_face_buf_idx = []
            vertex_face_region_index = np.where(self.vertex_face_region_mask[0] != 0)[0]
            for idx, triangle in enumerate(self.face_buf):
                xin = triangle[0] in vertex_face_region_index
                yin = triangle[1] in vertex_face_region_index
                zin = triangle[2] in vertex_face_region_index
                if xin or yin or zin:
                    new_face_buf_idx.append(idx)
            
            np.save(face_tri_path,np.array(new_face_buf_idx))
            return np.array(new_face_buf_idx)

    # def get_specular_weight_mask(self,atten_weight):
    #     weight_mask = torch.ones(512,512,1).to(self.device)
    #     weight_mask = weight_mask - self.specular_exclude_map.unsqueeze(-1) + self.specular_atten_map.unsqueeze(-1) * atten_weight
    #     return weight_mask

    def save_mean_shape(self,only_face=False):
        vert = self.mean_shape.view([-1, 3])
        tex_coord = self.vt
        triangle = self.tri
        tex_tri = self.complete_face_vt

        if only_face:
            face_region_tri_idx = self.get_face_tri_idx(os.path.join('./HIFI3D', 'hifi3d_face_region_idx.npy'))
            triangle = triangle[face_region_tri_idx]
            tex_tri = tex_tri[face_region_tri_idx]
            write_obj('./HIFI3D/mean_shape_only_face.obj',
            v_arr=vert.detach().cpu().numpy(),
            vt_arr=tex_coord.detach().cpu().numpy(),
            tri_v_arr=triangle.detach().cpu().numpy(),
            tri_t_arr=tex_tri.detach().cpu().numpy())
        else:
            write_obj('./HIFI3D/mean_shape_complete.obj',
            v_arr=vert.detach().cpu().numpy(),
            vt_arr=tex_coord.detach().cpu().numpy(),
            tri_v_arr=triangle.detach().cpu().numpy(),
            tri_t_arr=tex_tri.detach().cpu().numpy())

'''for debug'''
def write_obj(obj_path, v_arr, vt_arr, tri_v_arr, tri_t_arr):
    """write mesh data to .obj file.

    Param:
      obj_path: path to .obj file
      v_arr   : N x 3 (x, y, z values for geometry)
      vt_arr  : N x 2 (u, v values for texture)
      f_arr   : M x 3 (mesh faces and their corresponding triplets)

    Returns:
      None
    """
    tri_v_arr = np.copy(tri_v_arr)
    tri_t_arr = np.copy(tri_t_arr)

    
    with open(obj_path, "w") as fp:
        fp.write("mtllib test.mtl\n")
        for x, y, z in v_arr:
            fp.write("v %f %f %f\n" % (x, y, z))
        # for u, v in vt_arr:
        #    fp.write('vt %f %f\n' % (v, 1-u))
        if type(vt_arr) != type(None) and type(tri_t_arr) != type(None):
            if np.amax(vt_arr) > 1 or np.amin(vt_arr) < 0:
                print("Error: the uv values should be ranged between 0 and 1")
            for u, v in vt_arr:
                fp.write("vt %f %f\n" % (u, v))

            tri_v_arr += 1
            tri_t_arr += 1
            for (v1, v2, v3), (t1, t2, t3) in zip(tri_v_arr, tri_t_arr):
                fp.write(
                    "f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (v1, t1, t1, v2, t2, t2, v3, t3, t3)
                )
        else:
            for (v1,v2,v3) in tri_v_arr:
                fp.write(
                    "f %d %d %d\n" % (v1,  v2, v3)
                )


if __name__ == '__main__':
    hifi3d = HIFIParametricFaceModel()
    hifi3d.to('cuda')

    batch_size =32

    rand_para_id = torch.randn(size=[batch_size, hifi3d.id_base.shape[0]]).to(hifi3d.device)
    rand_para_exp = torch.randn(size=[batch_size, hifi3d.exp_base.shape[0]]).to(hifi3d.device) * 0.1
    rand_para_uv = torch.randn(size=[batch_size, hifi3d.tex_base.shape[0]]).to(hifi3d.device)
    
    rand_para_angles = torch.randn(size=[batch_size, 3]).to(hifi3d.device).to(hifi3d.device)
    rand_para_gammas = torch.randn(size=[batch_size, 27]).to(hifi3d.device).to(hifi3d.device)
    rand_para_translations = torch.randn(size=[batch_size, 3]).to(hifi3d.device).to(hifi3d.device)

    coeffs = torch.cat([rand_para_id,rand_para_exp,rand_para_uv,rand_para_angles,rand_para_gammas,rand_para_translations],dim = 1)

    hifi3d.compute_for_render(coeffs)

    ver_shape = hifi3d.compute_shape(rand_para_id,rand_para_exp)
    uv_texture, uv_mask = hifi3d.compute_texture(rand_para_uv,need_mask=True)

    for i in range(batch_size):
        write_obj(
            "test/%d_rand_shape.obj"%i,
            ver_shape[i].cpu().numpy(),
            hifi3d.vt.cpu().numpy(),
            hifi3d.tri.cpu().numpy(),
            hifi3d.face_vt.cpu().numpy(),
            # "face.mtl",
        )
    
    for i in range(batch_size):
        Image.fromarray(uv_texture[i].cpu().numpy().astype(np.uint8)).save("test/%d_rand_uv.png"%i)
        Image.fromarray((uv_mask[i].cpu().numpy()*255.).astype(np.uint8)).save("test/%d_rand_uv_mask.png"%i)