import torch


def cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Token-wise Cosine Similarity between Attention Heads"""
    _a, _b = torch.tensor(A.Attentions.to_numpy()).cuda(), torch.tensor(
        B.Attentions.to_numpy()
    ).cuda()
    return torch.nn.CosineSimilarity(dim=0)(_a, _b).cpu()

def cosine_distance(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Token-wise Cosine Distance between Attention Heads"""
    return 1 - cosine_similarity(A, B)

def euclidean_distance(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Token-wise Euclidean Distance between Attention Heads"""
    _a, _b = torch.tensor(A.Attentions.to_numpy()).cuda(), torch.tensor(
        B.Attentions.to_numpy()
    ).cuda()
    return torch.cdist(_a, _b, p=2).cpu()

def euclidean_distance_of_targets(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Euclidean Distance of target tokens between Attention Heads"""
    
    _a, _b = torch.tensor(A.Attentions.to_numpy()).cuda(), torch.tensor(
        B.Attentions.to_numpy()
    ).cuda()
    return torch.cdist(_a, _b, p=2).cpu()


def linear_distance(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Token-wise Linear Distance between Attention Heads"""
    _a, _b = torch.tensor(A.Attentions.to_numpy()).cuda(), torch.tensor(
        B.Attentions.to_numpy()
    ).cuda()
    return torch.cdist(_a, _b, p=1)
