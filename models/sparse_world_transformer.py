import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import bias_init_with_prob, Scale
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmcv.ops import knn
from mmdet.models.utils.builder import TRANSFORMER
from .bbox.utils import decode_bbox, decode_points, encode_points
from .utils import inverse_sigmoid, DUMP, MLN
from .sparse_world_sampling import sampling_4d
from .checkpoint import checkpoint as cp
from .csrc.wrapper import MSMV_CUDA


@TRANSFORMER.register_module()
class SparseWorldTransformer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 future_frames=[],
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=[1, 2, 4, 8, 16, 32],
                 scales=[1.0],
                 pc_range=[],
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)

        self.future_frames = future_frames
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_refines = num_refines

        self.decoder = SparseWorldTransformerDecoder(
            embed_dims, num_frames, future_frames, num_views, num_points, num_layers, num_levels,
            num_classes, num_refines, num_groups, scales, pc_range=pc_range)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, img_metas, fut2cur, fut_list):
        cls_scores, refine_pts = self.decoder(
            query_points, query_feat, mlvl_feats, img_metas, fut2cur, fut_list)

        cls_scores = [torch.nan_to_num(score) for score in cls_scores]
        refine_pts = [torch.nan_to_num(pts) for pts in refine_pts]

        return cls_scores, refine_pts


class SparseWorldTransformerDecoder(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 future_frames=[],
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_refines=16,
                 num_groups=4,
                 scales=[1.0],
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.num_frames = num_frames
        self.future_frames = future_frames
        self.num_views = num_views
        self.num_groups = num_groups
        self.embed_dims = embed_dims

        if len(scales) == 1:
            scales = scales * num_layers
        if not isinstance(num_refines, list):
            num_refines = [num_refines]
        if len(num_refines) == 1:
            num_refines = num_refines * num_layers
        last_refines = [1] + num_refines

        # params are shared across all decoder layers
        self.decoder_layers = ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                SparseWorldTransformerDecoderLayer(
                    embed_dims, num_frames, future_frames, num_views, num_points, num_levels, num_classes, 
                    num_groups, num_refines[i], last_refines[i], layer_idx=i, 
                    scale=scales[i], pc_range=pc_range)
            )
        
        ## embedding for fut2cur, pos_enc, pos_attn
        self.pe_mln = MLN(16)

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layers.init_weights()
    
    def time_embedding(self, future_frames):
        """
        sinusoidal_embedding
        future_frames = [0, 2, 4, 6]
        """
        frames = torch.tensor(future_frames, dtype=torch.float32)
        
        half_dim = self.embed_dims // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -embeddings)
        embeddings = frames[:, None] * embeddings[None, :]
        
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        if self.embed_dims % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros(len(future_frames), 1)], dim=-1)
            
        return embeddings

    def forward(self, query_points, query_feat, mlvl_feats, img_metas, fut2cur, fut_list):
        """
            Symbol meaning:
            B: batch size
            Q: num of queries
            T: num of frames
            G: num of groups (we follow the group sampling mechanism of AdaMixer)
            P: num of sampling points per frame per group
            N: num of views (six for nuScenes)
            L: num of layers of feature pyramid (typically it is 4: C2, C3, C4, C5)
        """
        cls_scores, refine_pts = [], []
        FT = len(fut2cur)
        # organize projections matrix and copy to CUDA
        lidar2img = np.asarray([m['lidar2img'] for m in img_metas]).astype(np.float32)
        lidar2img = query_feat.new_tensor(lidar2img) # [B, TN, 4, 4], [1, 48, 4, 4]
        ego2lidar = np.asarray([m['ego2lidar'] for m in img_metas]).astype(np.float32)
        ego2lidar = query_feat.new_tensor(ego2lidar) # [B, 4, 4]
        ego2lidar = ego2lidar.unsqueeze(1).expand_as(lidar2img)  # [B, TN, 4, 4], [1, 48, 4, 4]
        occ2img = torch.matmul(lidar2img, ego2lidar)  # [8, 48, 4, 4] 

        occ2img = occ2img.repeat(FT, 1, 1, 1)  

        # group image features in advance for sampling, see `sampling_4d` for more details
        # MSMV_CUDA = False  # if use pytorch grid sampling
        for lvl, feat in enumerate(mlvl_feats):
            # feat = feat.repeat(FT, 1, 1, 1, 1)
            B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
            N, T, G, C = self.num_views, self.num_frames, self.num_groups, GC//self.num_groups
            assert T*N == TN
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:  # Our CUDA operator requires channel_last
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
                feat = feat.reshape(B*T*G, N, H, W, C)
            else:  # Torch's grid_sample requires channel_first
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)  # [B, T, G, C, N, H, W]
                feat = feat.reshape(B*T*G, C, N, H, W)

            mlvl_feats[lvl] = feat.contiguous()

        B, Q, C = query_feat.shape
        ## FT = len(self.future_frames)
        ## TE
        time_embed = self.time_embedding(fut_list)  # [FT, 256]
        device = query_feat.device
        time_embed = time_embed.to(device=device)
        time_embed = time_embed.repeat_interleave(B, dim=0)  # [B*FT, C]
        time_embed = time_embed.unsqueeze(1)  # [B*FT, 1, C]
        time_embed = time_embed.expand(-1, Q, -1)  # [B*FT, Q, C]

        query_points = query_points.repeat(FT, 1, 1, 1)  # [B, Q, _, 3] -> [B*FT, Q, _, 3]
        query_feat = query_feat.repeat(FT, 1, 1)  # [B, Q, C] -> [B*FT, Q, C]

        query_feat = query_feat + time_embed  # [B*FT, Q, C]

        pos_mat = torch.cat(fut2cur, dim=0)
        pos_mat = pos_mat.view(B*FT, -1)  # [B*FT, 4, 4] -> [B*FT, 16]

        ## PE MLN
        pos_mat = pos_mat.unsqueeze(1)  # [B*FT, 1, 16]
        pos_mat = pos_mat.repeat(1, Q, 1)  # [B*FT, Q, 16]
        query_feat = self.pe_mln(query_feat, pos_mat)

        for i, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i

            query_points = query_points.detach()
            query_feat, cls_score, query_points = decoder_layer(
                query_points, query_feat, mlvl_feats, occ2img, img_metas, fut2cur)

            cls_scores.append(cls_score)
            refine_pts.append(query_points)

        return cls_scores, refine_pts


class SparseWorldTransformerDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 future_frames=[],
                 num_views=6,
                 num_points=4,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=16,
                 last_refines=16,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 layer_idx=0,
                 scale=1.0,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.future_frames = future_frames
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.num_points = num_points
        self.num_refines = num_refines
        self.last_refines = last_refines
        self.layer_idx = layer_idx
        self.scale = scale

        self.position_encoder = nn.Sequential(
            nn.Linear(3 * self.last_refines, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = SparseWorldSelfAttention(
            embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = SparseWorldSampling(embed_dims, num_frames=num_frames, num_views=num_views,
                                     num_groups=num_groups, num_points=num_points, 
                                     num_levels=num_levels, pc_range=pc_range)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames,
                                     n_groups=num_groups, out_points=32)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        self.tempo_attn = MultiheadAttention(embed_dims, num_heads=8, dropout=0.1, batch_first=True)
        self.ln = nn.LayerNorm(embed_dims)

        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(
            self.embed_dims, self.num_classes * self.num_refines))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, 3 * self.num_refines))
        self.reg_branch = nn.Sequential(*reg_branch)

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def refine_points(self, points_proposal, points_delta):
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, self.num_refines, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    def forward(self, query_points, query_feat, mlvl_feats, occ2img, img_metas, fut2cur):
        # query_points: [B*FT, Q, _, 3], [x, y, z]
        FT = len(fut2cur)
        query_pos = self.position_encoder(query_points.flatten(2, 3))
        # [B*FT, Q, _, 3] --> [B*FT, Q, _, 3] --> [B*FT, Q, _, C]
        query_feat = query_feat + query_pos  # [B*FT, Q, C]

        sampled_feat = self.sampling(
            query_points, query_feat, mlvl_feats, occ2img, img_metas, fut2cur)  # [B, Q, G, FP, C]  

        query_feat = self.norm1(self.mixing(sampled_feat, query_feat))
        BT, Q, C = query_feat.shape
        query_feat = query_feat.reshape(int(BT/FT), FT * Q, C)
        query_feat = self.ln(self.ffn(self.tempo_attn(query_feat)))
        query_feat = query_feat.reshape(BT, Q, C)
        query_feat = self.norm2(self.self_attn(query_points, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))
    
        B, Q = query_points.shape[:2]  # B=B*FT
        cls_score = self.cls_branch(query_feat)  # [B, Q, P * num_classes]
        reg_offset = self.scale * self.reg_branch(query_feat)  # [B, Q, P * 3]
        cls_score = cls_score.reshape(B, Q, self.num_refines, self.num_classes)
        refine_pt = self.refine_points(query_points, reg_offset)

        return query_feat, cls_score, refine_pt


class SparseWorldSelfAttention(BaseModule):
    def __init__(self, 
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = pc_range

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_points, query_feat):
        """
        query_points: [B, Q, _, 3]
        query_feat: [B, Q, C]
        """
        dist = self.calc_points_dists(query_points)
        tau = self.gen_tau(query_feat)  # [B, Q, 8]

        if DUMP.enabled:
            torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_points, query_feat):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat,
                      use_reentrant=False)
        else:
            return self.inner_forward(query_points, query_feat)

    @torch.no_grad()
    def calc_points_dists(self, points):
        points = decode_points(points, self.pc_range)
        points = points.mean(dim=2)
        dist = torch.norm(points.unsqueeze(-2) - points.unsqueeze(-3), dim=-1)
        return -dist


class SparseWorldSampling(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_frames=4,
                 num_views=6,
                 num_groups=4,
                 num_points=8,
                 num_levels=4,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames  # 8
        self.num_points = num_points  # 4
        self.num_views = num_views    # 6
        self.num_groups = num_groups  # 4
        self.num_levels = num_levels  # 4
        self.pc_range = pc_range      # [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_points, query_feat, mlvl_feats, occ2img, img_metas, fut2cur):
        '''
        query_points: [B, Q, _, 3]
        query_feat: [B, Q, C]
        '''
        B, Q = query_points.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # query points
        query_points = decode_points(query_points, self.pc_range)
        if query_points.shape[2] == 1:
            query_center = query_points
            query_scale = torch.zeros_like(query_center)
        else:
            query_center = query_points.mean(dim=2, keepdim=True)
            query_scale = query_points.std(dim=2, keepdim=True)

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, -1, 3)

        sampling_points = query_center + sampling_offset * query_scale
        sampling_points = sampling_points.view(B, Q, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            occ2img,
            fut2cur,
            image_h, image_w,
            self.num_views
        )  # [B, Q, G, FP, C]

        return sampled_feats

    def forward(self, query_points, query_feat, mlvl_feats, occ2img, fut2cur, img_metas):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat, mlvl_feats,
                      occ2img, fut2cur, img_metas, use_reentrant=False)
        else:
            return self.inner_forward(query_points, query_feat, mlvl_feats,
                                      occ2img, fut2cur, img_metas)


class AdaptiveMixing(nn.Module):
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super().__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        out = x.reshape(B*Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B*Q, G, self.out_points, self.in_points)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)
