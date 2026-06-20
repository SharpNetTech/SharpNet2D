#!/usr/bin/env false
# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
import torch
from torch.autograd.function import Function, FunctionCtx
import torch.nn as nn
import logging
from collections import defaultdict
import networkx as nx
import igraph as ig
from typing import Any, Optional, Tuple, Sequence

class MollifierNonZero(Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x2_1 = x.square() - 1
        # assert not (x2_1>=0).any()
        res = (1.0 / x2_1).exp()
        return res * torch.e, x2_1, res
    
    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs, output):
        (x, ) = inputs
        _, x2_1, res = output
        # ctx.mark_non_differentiable(x2_1, res)
        ctx.save_for_backward(x, x2_1, res)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_val_loss: torch.Tensor, _0, _1) -> torch.Tensor: # type: ignore
        x, x2_1, res, = ctx.saved_tensors
        grad_x_val = res * (-2.0 * x / x2_1.square()) * torch.e
        grad_x_loss = grad_x_val * grad_val_loss
        return grad_x_loss

def cat(*arrays,axis=0,**kwargs):
    return np.concatenate(arrays,axis=axis,**kwargs)

def stack(*arrays,axis=0,**kwargs):
    return np.stack(arrays,axis=axis,**kwargs)

def dot(a:torch.Tensor, b:torch.Tensor)->torch.Tensor:
    return torch.sum(a*b,dim=-1,keepdim=True)

def cross2d(a:torch.Tensor, b:torch.Tensor)->torch.Tensor:            
    a0=a[:,:,[0]]
    a1=a[:,:,[1]]
    b0=b[:,:,[0]]
    b1=b[:,:,[1]]
    return a0*b1-a1*b0

def cross3d(a:torch.Tensor, b:torch.Tensor)->torch.Tensor:
    return a[:,:,[1,2,0]]*b[:,:,[2,0,1]]-a[:,:,[2,0,1]]*b[:,:,[1,2,0]]

def C0_2d(pt:torch.Tensor, face:torch.Tensor, eps:torch.Tensor)->torch.Tensor:
    '''
    P x {F} x 2 , {P} x F x 2 x 2 => P x F
    '''
    v = face-pt.unsqueeze(dim=-2) # P x F x 2 x 2
    va = v[:,:,0,:] # P x F x 2
    vb = v[:,:,1,:]
    vd = face[:,:,1,:]-face[:,:,0,:]
    ld = vd.norm(dim=-1,keepdim=True) # P x F x 1
    l = torch.abs(cross2d(va,vb))/ld
    L_ = torch.square(l)+eps**2
    l_ = torch.sqrt(L_)
    def f(t):
        return t*(torch.log(torch.square(t)+L_)-2)/2+l_*torch.atan(t/l_)
    res = f(dot(vb,vd)/ld)-f(dot(va,vd)/ld)
    return res.squeeze(dim=-1)/torch.pi # P x F

def make_cpu_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x).cpu()
    else:
        x = torch.tensor(x).cpu()
    return x

def edge_split(edges:np.ndarray)->Sequence[Tuple[np.ndarray]]:
    e=edges.max()+1
    g=ig.Graph(n=e,edges=edges,vertex_attrs={'index':np.arange(e)})
    d=np.array(g.degree())
    mask=d!=2
    index=np.arange(e)
    juncs=index[mask]
    mids=index[np.logical_not(mask)]
    
    res=[]
    eids=np.sum(np.sort(edges,axis=-1)*np.array([[e,1]]),axis=-1)
    equery={eid:i for i,eid in enumerate(eids)}
    jnb=g.neighborhood(vertices=juncs,order=1,mindist=1)
    def find(x,num=-1):
            res=[]
            for i,nb in enumerate(jnb):
                if x in nb:
                    res.append(juncs[i])
            assert num<0 or len(res)==num
            return np.array(res)
    
    if len(mids)>0:
        g.delete_vertices(juncs)
        clusters=g.connected_components('weak')
        subgraphs=clusters.subgraphs()
        for subgraph in subgraphs:
            subindex=np.array(subgraph.vs['index'])
            d=np.array(subgraph.degree())
            mask=d!=2
            if mask.any():
                vertices=np.array(subgraph.vs.indices)
                ep=vertices[mask]
                if len(vertices)>=2:
                    assert len(ep)==2
                    paths=subgraph.get_all_simple_paths(ep[0],ep[1])
                    assert len(paths)==1
                    path=subindex[np.array(paths[0])]
                    assert len(path)==len(subindex)
                    a,b=subindex[ep]
                    path=np.concatenate((find(a,1),path,find(b,1)))
                else:
                    assert len(ep)==1
                    m=subindex[ep[0]]
                    l,r=find(m,2)
                    path=np.array([l,m,r])
            else:
                paths=subgraph.get_all_simple_paths(0,1)
                assert len(paths)==2
                path_t=subindex[np.array(paths[0])]
                path_r=subindex[np.array(paths[1])][::-1]
                path=cat(path_t,path_r[1:])
                assert len(path)==len(subindex)+1
            pedges=np.sort(np.stack((path[1:],path[:-1]),axis=-1),axis=-1)
            peids=pedges[:,0]*e+pedges[:,1]
            indice=np.array([equery[peid] for peid in peids])
            res.append((path,indice))
    
    jset=set()
    for i,junc in enumerate(juncs):
        for junc_ in jnb[i]:
            if junc_ in juncs:
                pair=sorted([junc,junc_])
                jset.add(equery[pair[0]*e+pair[1]])
    for eid in jset:
        res.append((edges[eid],np.array([eid])))
    
    return res

def C_n_2_pair(x:np.ndarray):
    n=len(x)
    assert n>0 and len(x.shape)==1
    if n==1:
        idx=np.zeros((0,2),dtype=np.int64)
    else:
        catlist=[stack(i+np.zeros(n-1-i,dtype=np.int64),
                       i+1+np.arange(n-1-i,dtype=np.int64),
                       axis=-1) for i in range(n-1)]
        idx=cat(*catlist,axis=0,dtype=np.int64)
    return x[idx]

def junc_mask(edges:np.ndarray):
    junc2path=defaultdict(set)
    paths=edge_split(edges)
    for pathid,p in enumerate(paths):
        junc2path[p[0][0]].add(pathid)
        junc2path[p[0][-1]].add(pathid)
    pedges=[C_n_2_pair(np.array(list(junc2path[junc]),dtype=np.int64)) for junc in junc2path]
    pedges=cat(*pedges)
    G=nx.Graph()
    G.add_nodes_from(range(len(paths)))
    G.add_edges_from(pedges)
    pathcolor=nx.coloring.greedy_color(G,strategy="largest_first")
    edgecolor=np.empty(len(edges),dtype=np.int64)
    for pathid,p in enumerate(paths):
        edgecolor[p[1]]=pathcolor[pathid]
    return edgecolor

class BEMquery(nn.Module):
    '''
    vertices: V x 2(3) , edges: E x 1(2)
    '''
    def __init__(
        self,
        vertices,
        edges,
        nonmanifolds = None,
        eps=1e-16,
        radius=-1.0,
        once_bias=False,
        split_feature=False,
        **kwargs
    ):
        super().__init__()
        if len(kwargs.keys()) > 0:
            key_list = list(map(lambda x : '`' + str(x) + '`', list(kwargs.keys())))
            logging.warning("BEMquery received extra arguments: {}".format(', '.join(key_list)))

        self.learnable = False
        self.eps = eps
        self.radius = radius

        # make input tensor
        vertices = make_cpu_tensor(vertices).type(torch.float32)
        edges = make_cpu_tensor(edges).type(torch.int32)

        # split the vertices into fixed and movable ones
        fixed_vertices, movable_vertices, new_edges = self.analyze_init(vertices, edges)

        self.indim = vertices.shape[-1] # i: 2(3)

        self.fixed_vertices = nn.Parameter(fixed_vertices, requires_grad=False)
        self.movable_vertices = nn.Parameter(movable_vertices, requires_grad=False)
        logging.info("BEMquery: fixed vertices shape: {}".format(self.fixed_vertices.shape))
        logging.info("BEMquery: movable vertices shape: {}".format(self.movable_vertices.shape))

        self.edges = nn.Parameter(new_edges, requires_grad=False)
        logging.info("BEMquery: edges shape: {}".format(self.edges.shape))
        self.E = self.edges.size(dim=0)

        if once_bias:
            self.bias = nn.Parameter(torch.tensor(torch.nan)).requires_grad_(False)
        self.use_bias = once_bias

        if split_feature:
            mask_mat=self.edge_to_mask()
            self.split_dim=mask_mat.shape[1]
            self.mask_mat=nn.Parameter(mask_mat.type(torch.float32), requires_grad=False)
        else:
            self.split_dim=1
            self.mask_mat=nn.Parameter(torch.ones((len(self.edges),1),
                                        dtype=torch.float32), requires_grad=False)

    def edge_to_mask(self):
        edges=self.edges.detach().cpu().numpy()
        F=len(edges)
        color=junc_mask(edges)
        mask_mat=torch.zeros((F,color.max()+1),dtype=torch.float32)
        mask_mat[torch.arange(F),color]+=1
        return mask_mat

    @staticmethod
    def analyze_init(vertices: torch.Tensor, edges: torch.Tensor):
        """ Analyse the initial mesh """
        # Gather the degrees of each vertex
        degree = {i: 0 for i in range(vertices.shape[0])}
        for edge in edges:
            degree[edge[0].item()] += 1
            degree[edge[1].item()] += 1

        fixed_vertice_index = [i for i, cnt in degree.items() if cnt == 1]
        movable_vertice_index = [i for i, cnt in degree.items() if cnt != 1]

        # reshuffle the indices
        old_index_to_new_index = [i for i in range(vertices.shape[0])]
        for new_idx, old_idx in enumerate(fixed_vertice_index + movable_vertice_index):
            old_index_to_new_index[old_idx] = new_idx

        new_edges = []
        for edge in edges:
            new_edges.append([old_index_to_new_index[edge[0].item()], old_index_to_new_index[edge[1].item()]])

        new_edges = torch.tensor(new_edges, device='cpu', dtype=torch.int32)
        fixed_vertices = vertices[fixed_vertice_index, :]
        movable_vertices = vertices[movable_vertice_index, :]
        return fixed_vertices, movable_vertices, new_edges
    
    def vertices_all(self):
        return torch.cat([self.fixed_vertices, self.movable_vertices], dim=0)

    def edge(self):
        return torch.index_select(self.vertices_all(), 0, self.edges.flatten()) \
                .reshape(self.E, self.indim, self.indim) # F x i x i

    @staticmethod
    def mollifier_nz(x: torch.Tensor) -> torch.Tensor:
        return MollifierNonZero.apply(x)[0]    # type: ignore

    @staticmethod
    def solve_nan(x: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
        return torch.where(x == 0.0, eps, x)

    def output_dim(self) -> int:
        return self.split_dim

    def set_learnable(self, learnable: bool):
        if self.learnable != learnable:
            self.learnable = learnable
            self.movable_vertices.requires_grad_(learnable)
            logging.info(f"BEMquery: set movable vertices learnable={learnable}")

    @staticmethod
    def original_function(u:torch.Tensor, v:torch.Tensor, eps:torch.Tensor):
        L = v.square()+eps.square()
        l = L.sqrt()
        res = u*(torch.log(torch.square(u)+L)-2)/2+l*torch.atan(u/l)
        return res

    @torch.no_grad()
    def get_valid_indice(self, x: torch.Tensor, radius: float) -> torch.Tensor:
        v_all = self.vertices_all() # V x 2
        edge_ = v_all[self.edges].unsqueeze(0) # 1*e*2*2
        mid_ = edge_.mean(-2) # 1*e*2
        p2e_ = mid_ - x.unsqueeze(1) # p*e*2
        indice = ((p2e_.norm(dim=-1) - radius) < 0) \
            .to_sparse().indices() # p*e -> 2*c
        return indice
    
    def forward_sparse(self, x: torch.Tensor, eps: torch.Tensor, radius: float, batch_size: int = -1) -> torch.Tensor:
        P = x.shape[0]
        E = self.edges.shape[0]
        p_e_indice = self.get_valid_indice(x, radius) # 2*c
        if p_e_indice.shape[1]:
            pt = x[p_e_indice[0]] # c*2
            v_all = self.vertices_all() # V x 2
            edge = v_all[self.edges[p_e_indice[1]]] # c*2*2
            p2e = edge - pt.unsqueeze(1)
            p2m = edge.mean(-2) - pt # c*2
            e2e = edge[:,1,:] - edge[:,0,:] # c*2
            len_u = e2e.norm(dim=-1, keepdim=True) # c*1
            m_weight = self.mollifier_nz(p2m.norm(dim=-1) / radius)
            u = e2e / self.solve_nan(len_u) # c*2
            v = torch.stack((-u[:,1], u[:,0]), dim=1)
            u_0_1 = (p2e * u.unsqueeze(1)).sum(dim=-1)
            v_0 = (p2e[:,0,:] * v).sum(dim=-1, keepdim=True) # c*1
            res_0_1 = self.original_function(u_0_1, v_0, eps=eps) # c*2
            out = torch.sparse_coo_tensor(
                indices=p_e_indice,
                values=(res_0_1[:,1]-res_0_1[:,0])*m_weight,
                size=(P,E)
            )
            # out = torch.sparse.sum(out,[1]).to_dense().unsqueeze(1)
        else:
            out = torch.sparse_coo_tensor(
                indices=[[0,P-1],[0,E-1]],
                values=0.0*x[0],
                size=(P,E),
                dtype=x.dtype,device=x.device
            )
            # # out = torch.zeros_like(x[..., :1],device=x.device)
            # out = 0.0 * x[..., :1]  # P x 1     # Must somehow involve math on x to allow for autograd to work
        return out
    
    def forward(self, pt:torch.Tensor, eps: Optional[torch.Tensor]=None, radius:Optional[float]=None):
        '''
        P x 2(3) => P x 1
        '''
        single_point = False
        if len(pt.shape) == 1:
            pt = pt.unsqueeze(dim=0) # 1 x i
            single_point = True
        rpt = pt.unsqueeze(dim=1) # P x {1} x i

        if eps is None:
            eps = torch.tensor(self.eps, device=pt.device, dtype=pt.dtype)
        elif not isinstance(eps, torch.Tensor):
            eps = torch.tensor(eps, device=pt.device, dtype=pt.dtype)
        eps = torch.clamp(eps, min=1e-16)

        if radius is None:
            radius = self.radius

        if radius<0:
            redge = self.edge().unsqueeze(dim=0) # {1} x F x i x i
            res = C0_2d(rpt,redge,eps=eps) # P x F
            res = torch.mm(res,self.mask_mat) # P x 1
        else:
            res_sparse = self.forward_sparse(pt, eps, radius)
            res = torch.sparse.mm(res_sparse,self.mask_mat)

        if self.use_bias:
            if torch.isnan(self.bias):
                self.bias.data = -res.mean()
            res = res + self.bias

        if single_point:
            res = res.squeeze() # 1 x 1 -> scalar
        return res

    def gradient(self, pt: torch.Tensor, channel: int=0, eps: Optional[torch.Tensor]=None, radius:Optional[float]=None):
        pt.requires_grad_(True)
        y = self.forward(pt, eps=eps, radius=radius)[..., channel:channel+1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=pt,
            grad_outputs=d_output,
            create_graph=False,
            retain_graph=False,
            only_inputs=True)[0]
        return gradients

    def laplacian_loss(self, a: float=0.3):
        verts = self.vertices_all()
        if hasattr(self, "connected2"):
            lap_center, lap_side = self.connected2
        else:
            # Get a list of neighbors for each vertex
            verts_np = verts.detach().cpu().numpy()
            edges_np = self.edges.detach().cpu().numpy()
            neighbors = [[] for _ in range(verts_np.shape[0])]
            for edge in edges_np:
                neighbors[edge[0]].append(edge[1])
                neighbors[edge[1]].append(edge[0])
            lap_center = []
            lap_side = []
            for i, neighbor in enumerate(neighbors):
                if len(neighbor) == 2:
                    lap_center.append(i)
                    lap_side.append(neighbor)
            lap_center = torch.tensor(lap_center, dtype=torch.int32, device=verts.device)  # M
            lap_side = torch.tensor(lap_side, dtype=torch.int32, device=verts.device)  # M x 2
            self.connected2 = (lap_center, lap_side)
        with torch.no_grad():
            # Create an M x 2 x 2 matrix, which contains the coordinates indexed by lap_side
            side = torch.index_select(verts, 0, lap_side.flatten()).reshape(-1, 2, 2)
            target = side.mean(dim=1)   # M x 2
            vec = side[:,1,:] - side[:,0,:]
            vec = vec / (1e-7 + vec.norm(dim=1, keepdim=True))  # M x 2
        center = torch.index_select(verts, 0, lap_center)  # M x 2
        pos_loss = (center - target).square().sum(dim=-1).mean()
        ortho_loss = ((center - target) * vec).sum(dim=-1).square().mean()
        return pos_loss * a + ortho_loss * (1 - a)
