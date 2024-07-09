"""This script is the differentiable renderer for Deep3DFaceRecon_pytorch
    Attention, antialiasing step is missing in current version.
"""

import torch
import torch.nn.functional as F
import kornia
from kornia.geometry.camera import pixel2cam
import numpy as np
from typing import List
import nvdiffrast.torch as dr
from scipy.io import loadmat
from torch import nn

def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0, n/-x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

class MeshRenderer(nn.Module):
    def __init__(self,
                rasterize_fov,
                znear=0.1,
                zfar=10, 
                rasterize_size=224):
        super(MeshRenderer, self).__init__()

        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(ndc_projection(x=x, n=znear, f=zfar)).matmul(
                torch.diag(torch.tensor([1., -1, -1, 1])))
        self.rasterize_size = rasterize_size
        self.glctx = None

    def forward(self, vertex, tri, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, C), features
        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1] 


        vertex_ndc = vertex @ ndc_proj.t()
        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(device=device)
            print("create glctx on device cuda:%d"%device.index)
        
        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device) 
            fstartidx = torch.cumsum(fnum, dim=0) - fnum 
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i*vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.glctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)

        depth, _ = dr.interpolate(vertex.reshape([-1,4])[...,2].unsqueeze(1).contiguous(), rast_out, tri) 
        depth = depth.permute(0, 3, 1, 2)
        mask =  (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth
        

        image = None
        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri)
            image = image.permute(0, 3, 1, 2)
            image = mask * image
        
        return mask, depth, image


    def transform_to_ndc(self, vertex):
        device = vertex.device
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1] 
        vertex_ndc = vertex @ ndc_proj.t()
        return vertex_ndc.contiguous()

    def bi_direction_forward(self, vertex_ndc_input, tri_input, uv_map_input, uv_coord_input, uv_idx_input, specific_img_size=0, specific_uv_size=0, interpolate_mode='linear', direction='obverse'):
        '''
        can be used in bi-direction way:
        obverse: 
            # given ndc coordinate, mesh, and uv texture, rendering a image       
            vertex_ndc: ndc coordinates of project points   [B*Nv*4] (-1,1)
            tri:        trianlge of vertex idx              [B*Nf]
            uv_map:     the uv texture                      [B*uvsize*uvsize*3] (0,Nf)
            uv_coord:   the coordinates of texture points   [N-uv*2]
            uv_idx:     trianlge of uv idx                  [Nf-uv*3] (0,N-uv)
        reverse:
            # given a ndc coordinate, mesh, and a rendered image, re-project a uv texture 
            vertex_ndc: transformed texture points ( [0,1] -> [-1,1] )
            tri:        trianlge of uv idx
            uv_map:     the rendered image
            uv_coord:   ndc coordinates of project points
            uv_idx:     triangle of vertex idx
        '''
        device = vertex_ndc_input.device
        # rsize = int(self.rasterize_size) if specific_r_size==0 else specific_r_size
        if specific_img_size == 0:
            specific_img_size = self.rasterize_size
        if specific_uv_size == 0:
            specific_uv_size = self.rasterize_size

        vertex_ndc_input = vertex_ndc_input / vertex_ndc_input[:,:,3:]
        if direction == 'obverse':
            vertex_ndc = vertex_ndc_input.contiguous()
            tri = tri_input.contiguous()
            uv_map = uv_map_input.contiguous()
            uv_coord = uv_coord_input.contiguous()
            uv_idx = uv_idx_input.contiguous()
        if direction == 'reverse':
            vertex_ndc = torch.cat([uv_coord_input.unsqueeze(0)*2-1,0.5*torch.ones(1,uv_coord_input.shape[0],1).to(device),torch.ones(1,uv_coord_input.shape[0],1).to(device)], dim=2).contiguous()
            tri = uv_idx_input.contiguous()
            uv_map = uv_map_input.contiguous()
            uv_coord = (vertex_ndc_input[:,:,:2]*0.5+0.5).contiguous()
            uv_idx = tri_input.contiguous().int()

            
            vertex_ndc = torch.cat([uv_coord_input[:,0:1].unsqueeze(0)*2-1,
                                    (1-uv_coord_input[:,1:2]).unsqueeze(0)*2-1,
                                    0.5*torch.ones(1,uv_coord_input.shape[0],1).to(device),
                                    torch.ones(1,uv_coord_input.shape[0],1).to(device)], dim=2).contiguous()
            uv_coord = (vertex_ndc_input[:,:,:2]*0.5+0.5).contiguous()
            uv_coord[:,:,1] = 1-uv_coord[:,:,1]
            uv_coord[uv_coord[:,:,0]>1] = 0
            uv_coord[uv_coord[:,:,0]<0] = 0
            uv_coord[uv_coord[:,:,1]>1] = 0
            uv_coord[uv_coord[:,:,1]<0] = 0

        if self.glctx is None:
            try: 
                self.glctx = dr.RasterizeCudaContext(device=device)
                print("create glctx on device cuda:%d"%device.index)
            except:
                self.glctx = dr.RasterizeGLContext(device=device)
                print("create glctx on device cuda:%d"%device.index)
        
        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device) 
            fstartidx = torch.cumsum(fnum, dim=0) - fnum 
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i*vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.glctx, vertex_ndc.contiguous(), tri, resolution=[specific_img_size, specific_img_size], ranges=ranges)

        # filter out overlapped behind triangles
        if direction == 'reverse':
            rast_out_obverse,_ = dr.rasterize(self.glctx,  vertex_ndc_input.contiguous(), tri_input.int().contiguous(), resolution=[specific_uv_size, specific_uv_size], ranges=ranges)
            visible_triangle_idxs = torch.unique(rast_out_obverse[:,:,:,3].view(-1))
            rast_out[torch.isin(rast_out[:,:,:,3], visible_triangle_idxs).logical_not()] = 0

        texc, _ = dr.interpolate(uv_coord.contiguous(), rast_out, uv_idx.contiguous()) # [B, img_size, img_size, 2] 像素的uv坐标
        pix_tex = dr.texture(uv_map.flip(1,), texc.detach(), filter_mode=interpolate_mode)  # [B, img_size, img_size, 3] 像素的diffuse rgb颜色，从预测的uv map中按照texc坐标取得
        pix_tex = pix_tex.permute(0, 3, 1, 2)

        mask =  (rast_out[..., 3] > 0).float().unsqueeze(1)
        pix_tex = mask * pix_tex

        return mask, pix_tex



    def forward_with_texture_map(self, vertex, tri, uv_map, uv_coord, uv_idx, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            pix_tex            -- torch.tensor, size (B, 3, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            uv_map          -- torch.tensor, size (B, uv_size, uv_size, 3)
            uv_coord        -- torch.tensor, size (B, N2, 2) or (N2, 2), uv coordinates
            uv_idx          -- torch.tensor, size (B, M, 3) or (M, 3),  triangles' uv idx
            feat(optional)  -- torch.tensor, size (B, N, C), vertex-wise C-dim features

        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1] 


        vertex_ndc = vertex @ ndc_proj.t()
        vertex_ndc = vertex_ndc.float()
        if self.glctx is None:
            try: 
                self.glctx = dr.RasterizeCudaContext(device=device)
                print("create glctx on device cuda:%d"%device.index)
            except:
                self.glctx = dr.RasterizeGLContext(device=device)
                print("create glctx on device cuda:%d"%device.index)
        
        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device) 
            fstartidx = torch.cumsum(fnum, dim=0) - fnum 
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i*vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.glctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)

        rast_out, _ = dr.rasterize(self.glctx, (vertex_ndc/vertex_ndc[:,:,3:]).contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)

        # depth, _ = dr.interpolate(vertex.reshape([-1,4])[...,2].unsqueeze(1).contiguous(), rast_out, tri) 
        # 两种写法等价
        # TODO: 检查interpolate的第一个参数应该是vertex还是vertex_ndc,原版本为vertex
        depth, _ = dr.interpolate(vertex[:,:,2].unsqueeze(2).contiguous(), rast_out, tri) 
        depth = depth.permute(0, 3, 1, 2)

        texc, _ = dr.interpolate(uv_coord.contiguous(), rast_out, uv_idx.contiguous()) # [B, img_size, img_size, 2] 像素的uv坐标
        pix_tex = dr.texture(uv_map.flip(1,), texc.detach(), filter_mode='linear')  # [B, img_size, img_size, 3] 像素的diffuse rgb颜色，从预测的uv map中按照texc坐标取得
        pix_tex = pix_tex.permute(0, 3, 1, 2)

        mask =  (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth
        pix_tex = mask * pix_tex

        feat_image = None
        if feat is not None:
            feat_image, _ = dr.interpolate(feat, rast_out, tri)
            feat_image = feat_image.permute(0, 3, 1, 2)
            feat_image = mask * feat_image

        return mask, depth, pix_tex, feat_image

    def forward_with_normal_mlp(self, world_vertex, vertex, tri, uv_map,uv_coord, uv_idx, v_N,v_T,v_B,mlp):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            pix_tex            -- torch.tensor, size (B, 3, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            world_vertex    -- torch.tensor, size (B, N, 3), world coordinate
            vertex          -- torch.tensor, size (B, N, 3), camera coordinate
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            tex_pos_mlp     -- lib.fantasia3d.render.mlptexture.MLPTexture3D, use .sample(pos) to get a texture(,9) (kd,ks,normal)
            uv_coord        -- torch.tensor, size (B, N2, 2) or (N2, 2), uv coordinates
            uv_idx          -- torch.tensor, size (B, M, 3) or (M, 3),  triangles' uv idx
            feat(optional)  -- torch.tensor, size (B, N, C), vertex-wise C-dim features

        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1] 


        vertex_ndc = vertex @ ndc_proj.t()

        if self.glctx is None:
            try: 
                self.glctx = dr.RasterizeCudaContext(device=device)
                print("create glctx on device cuda:%d"%device.index)
            except:
                self.glctx = dr.RasterizeGLContext(device=device)
                print("create glctx on device cuda:%d"%device.index)
        
        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device) 
            fstartidx = torch.cumsum(fnum, dim=0) - fnum 
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i*vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.glctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)

        rast_out, _ = dr.rasterize(self.glctx, (vertex_ndc/vertex_ndc[:,:,3:]).contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)

        depth, _ = dr.interpolate(vertex[:,:,2].unsqueeze(2).contiguous(), rast_out, tri) 
        depth = depth.permute(0, 3, 1, 2)

        # 预测pix_tex
        texc, _ = dr.interpolate(uv_coord.contiguous(), rast_out, uv_idx.contiguous()) # [B, img_size, img_size, 2] 像素的uv坐标
        pix_tex = dr.texture(uv_map.flip(1,), texc.detach(), filter_mode='linear')  # [B, img_size, img_size, 3] 像素的diffuse rgb颜色，从预测的uv map中按照texc坐标取得
        pix_tex = pix_tex.permute(0, 3, 1, 2)

        mask =  (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth
        pix_tex = mask * pix_tex

        # 使用mlp预测法线偏移 
        world_pos, _ = dr.interpolate(world_vertex.contiguous(), rast_out, tri)
        pix_tangent_normal = mlp.sample(world_pos)
        pix_tangent_normal[:,:,:,2] = pix_tangent_normal[:,:,:,2] * 0.5 + 0.5 # B , size, size, 3

        # 法线归一化
        pix_tangent_normal = F.normalize(pix_tangent_normal, dim=-1, p=2)

        # 像素级的tbn
        pix_T, _ = dr.interpolate(v_T, rast_out, tri)
        pix_B, _ = dr.interpolate(v_B, rast_out, tri)
        pix_N, _ = dr.interpolate(v_N, rast_out, tri)
        pix_T = mask.permute(0,2,3,1) * pix_T # B , size, size, 3
        pix_B = mask.permute(0,2,3,1) * pix_B # B, size, size, 3
        pix_N = mask.permute(0,2,3,1) * pix_N # B , size, size, 3
        pix_TBN = torch.cat([pix_T.unsqueeze(-1),pix_B.unsqueeze(-1),pix_N.unsqueeze(-1)],dim=-1) # B ,size,size, 3 , 3

        pix_norm = pix_TBN @ pix_tangent_normal.unsqueeze(-1) # B , size, size, 3, 1
        pix_norm = pix_norm[...,0].permute(0,3,1,2) * mask

        return mask, depth, pix_tex, pix_norm     

    def forward_with_texture_mlp(self, world_vertex, vertex, tri, tex_pos_mlp, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            pix_tex            -- torch.tensor, size (B, 3, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            world_vertex    -- torch.tensor, size (B, N, 3), world coordinate
            vertex          -- torch.tensor, size (B, N, 3), camera coordinate
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            tex_pos_mlp     -- lib.fantasia3d.render.mlptexture.MLPTexture3D, use .sample(pos) to get a texture(,9) (kd,ks,normal)
            uv_coord        -- torch.tensor, size (B, N2, 2) or (N2, 2), uv coordinates
            uv_idx          -- torch.tensor, size (B, M, 3) or (M, 3),  triangles' uv idx
            feat(optional)  -- torch.tensor, size (B, N, C), vertex-wise C-dim features

        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1] 


        vertex_ndc = vertex @ ndc_proj.t()

        if self.glctx is None:
            try: 
                self.glctx = dr.RasterizeCudaContext(device=device)
                print("create glctx on device cuda:%d"%device.index)
            except:
                self.glctx = dr.RasterizeGLContext(device=device)
                print("create glctx on device cuda:%d"%device.index)
        
        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device) 
            fstartidx = torch.cumsum(fnum, dim=0) - fnum 
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i*vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.glctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)

        rast_out, _ = dr.rasterize(self.glctx, (vertex_ndc/vertex_ndc[:,:,3:]).contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)

        depth, _ = dr.interpolate(vertex[:,:,2].unsqueeze(2).contiguous(), rast_out, tri) 
        depth = depth.permute(0, 3, 1, 2)

        # 使用mlp 预测pix_tex
        world_pos, _ = dr.interpolate(world_vertex.contiguous(), rast_out, tri)
        pix_tex = tex_pos_mlp.sample(world_pos).permute(0,3,1,2)

        mask =  (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth
        pix_tex = mask * pix_tex

        feat_image = None
        if feat is not None:
            feat_image, _ = dr.interpolate(feat, rast_out, tri)
            feat_image = feat_image.permute(0, 3, 1, 2)
            feat_image = mask * feat_image

        return mask, depth, pix_tex, feat_image     



    def forward_with_detail_map(self, vertex, tri, uv_map, uv_coord, uv_idx, detail_map, unwrapper,normal_smooth_kernel=0,feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            pix_tex            -- torch.tensor, size (B, 3, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            uv_map          -- torch.tensor, size (B, uv_size, uv_size, 3)
            uv_coord        -- torch.tensor, size (B, N2, 2) or (N2, 2), uv coordinates
            uv_idx          -- torch.tensor, size (B, M, 3) or (M, 3),  triangles' uv idx
            detail_map      -- dict
                ['diffuse_offset'] diffuse offset map,      size (B, uv_size, uv_size, 3)
                ['normal_offset'] normal offset map,       size (B, uv_size, uv_size, 3)
                [2] ...
            feat(optional)  -- torch.tensor, size (B, C), features

        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1] 


        vertex_ndc = vertex @ ndc_proj.t()
        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(device=device)
            print("create glctx on device cuda:%d"%device.index)
        
        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device) 
            fstartidx = torch.cumsum(fnum, dim=0) - fnum 
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i*vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.glctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)

        # depth, _ = dr.interpolate(vertex.reshape([-1,4])[...,2].unsqueeze(1).contiguous(), rast_out, tri) 
        # 两种写法等价
        # TODO: 检查interpolate的第一个参数应该是vertex还是vertex_ndc,原版本为vertex
        depth, _ = dr.interpolate(vertex[:,:,2].unsqueeze(2).contiguous(), rast_out, tri) 
        depth = depth.permute(0, 3, 1, 2)

        texc, _ = dr.interpolate(uv_coord.contiguous(), rast_out, uv_idx.contiguous()) # [B, img_size, img_size, 2] 像素的uv坐标

        detail_diffuse_map = uv_map + detail_map['diffuse_offset']                      # [B, uv_size, uv_size , 3] 添加了细节offset后的diffuse map
        # detail_diffuse_map = uv_map
        pix_tex = dr.texture(detail_diffuse_map.contiguous().flip(1,), texc.detach(), filter_mode='linear')  # [B, img_size, img_size, 3] 像素的diffuse rgb颜色，从预测的uv map中按照texc坐标取得
        pix_tex = pix_tex.permute(0, 3, 1, 2)

        uv_tangent_normal_map = detail_map['tangent_normal_map']
        if 'tangent_normal_offset' in detail_map.keys():
            if normal_smooth_kernel != 0:
                from models import losses
                tangent_normal_offset = losses.gaussian_blur(detail_map['tangent_normal_offset'].permute(0,3,1,2), normal_smooth_kernel*2+1)
                tangent_normal_offset = tangent_normal_offset.permute(0,2,3,1)
            else:
                tangent_normal_offset = detail_map['tangent_normal_offset']
            uv_tangent_normal_map = uv_tangent_normal_map + tangent_normal_offset


        # imageio.imwrite('./test-smooth-normalmap.png',uv_tangent_normal_map[0].detach().cpu().numpy())        

        #-----test blur smooth------
        
        # world_uv_map = 
        uv_world_normal_map = unwrapper.recover_pix_normal_2(uv_tangent_normal_map,detail_map['T_basis_map'],detail_map['B_basis_map'],detail_map['N_basis_map'])
        pix_norm = dr.texture(uv_world_normal_map.flip(1,),texc.detach(), filter_mode='linear')
        pix_norm = F.normalize(pix_norm,dim=-1)
        pix_norm = pix_norm.permute(0, 3, 1, 2)

        mask =  (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth
        pix_tex = mask * pix_tex
        pix_norm = mask * pix_norm

        feat_image = None
        if feat is not None:
            feat_image, _ = dr.interpolate(feat, rast_out, tri)
            feat_image = feat_image.permute(0, 3, 1, 2)
            feat_image = mask * feat_image

        return mask, depth, pix_tex, pix_norm, feat_image