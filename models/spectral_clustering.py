import torch
from kmeans_pytorch import kmeans,kmeans_predict
from similarity import *

def spectral_embedding(
    affinity='rbf',
    feats=None,
    n_components=3,
    norm_laplacian=True,
    drop_first=True,
    gamma=1
):
# only support rbf kernel
    B = feats.size(0)
    d = rbf_similarity(similarity_matrix(feats,feats,'euclidean'),gamma)
    D_ = d.sum(-1)
    D = torch.diag_embed(D_)
    L = D - d

    if norm_laplacian:
        D = torch.diag_embed(torch.pow(D_,-0.5))
        L=D@L@D

    U, V = torch.linalg.eig(L)
    U = torch.real(U)
    V = torch.real(V)

    _,idic=torch.sort(U)
    mask = idic.clone()
    mask[:] = 0
    if drop_first:
        mask = mask.scatter_(1,idic[:,1:n_components+1],1) == 1
    else:
        mask = mask.scatter_(1,idic[:,:n_components],1) == 1

    V = V[mask].view(B,n_components,-1).transpose(1,2)
    return V

def spectral_clustering(
    affinity='rbf',
    feats=None,
    n_clusters=3,
    n_components=None,
    n_init=10,
    assign_labels="kmeans",
    gamma=1,
    cluster_center=None,
    is_training=True
):
    n_components = n_clusters if n_components is None else n_components

    # We now obtain the real valued solution matrix to the
    # relaxed Ncut problem, solving the eigenvalue problem
    # L_sym x = lambda x  and recovering u = D^-1/2 x.
    # The first eigenvector is constant only for fully connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    # See spectral_embedding documentation.
    maps = spectral_embedding(
        affinity,
        feats,
        n_components=n_components,
        norm_laplacian=True,
        drop_first=False,
        gamma=gamma,
    )
    # Only support kmeans
    if assign_labels == "kmeans":
        if is_training:
            _, labels, _ = kmeans(
                maps, n_clusters, cluster_center=cluster_center, n_init=n_init, tqdm_flag=False, 
            )
        else:
            labels = kmeans_predict(X=maps,device=torch.cuda.current_device(),cluster_centers = self.get_parameter('cluster_centers').data,tqdm_flag=False,distance=self.cluster_distance)

    return labels