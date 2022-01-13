import torch
from kmeans_pytorch import kmeans
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
    distance='euclidean',
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
            for b in maps.size(0):
                labels, cluster_center = kmeans(
                    maps[b], n_clusters, cluster_center=cluster_center, n_init=n_init, tqdm_flag=False, 
                )
            return labels,cluster_center
        else:
            labels = kmeans_predict_bs(X=maps,device=torch.cuda.current_device(),cluster_centers = cluster_center,distance=distance)

            return labels

def kmeans_predict_bs(
    X,
    cluster_centers,
    distance='euclidean',
    device=torch.device('cpu'),
):
    """
    predict using cluster centers for batch data
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) (batch_size,cluster ids)
    """
    assert len(cluster_centers.size()) == 2
    if len(X.size()) == 2:
        X.unsqueeze_(0)
    if distance == 'euclidean':
        pairwise_distance_function = SimilarityMatrix('euclidean')
    elif distance == 'cosine':
        pairwise_distance_function = SimilarityMatrix('cosine',True)
    else:
        raise NotImplementedError
    
    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)
    
    dis = pairwise_distance_function(X, cluster_centers.unsqueeze(0).repeat(X.size(0),1,1))
    choice_cluster = torch.argmin(dis, dim=2)

    return choice_cluster