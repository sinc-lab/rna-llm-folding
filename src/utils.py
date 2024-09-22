import torch
import numpy as np


def outer_concat(t1: torch.Tensor, t2: torch.Tensor):
    # t1, t2: shape = B x L x E
    assert t1.shape == t2.shape, f"Shapes of input tensors must match! ({t1.shape} != {t2.shape})"

    seq_len = t1.shape[1]
    a = t1.unsqueeze(-2).expand(-1, -1, seq_len, -1)
    b = t2.unsqueeze(-3).expand(-1, seq_len, -1, -1)

    return torch.concat((a, b), dim=-1)

def mat2bp(x):
    """Get base-pairs from conection matrix [N, N]. It uses upper
    triangular matrix only, without the diagonal. Positions are 1-based. """
    ind = torch.triu_indices(x.shape[0], x.shape[1], offset=1)
    pairs_ind = torch.where(x[ind[0], ind[1]] > 0)[0]

    pairs_ind = ind[:, pairs_ind].T
    # remove multiplets pairs
    multiplets = []
    for i, j in pairs_ind:
        ind = torch.where(pairs_ind[:, 1]==i)[0]
        if len(ind)>0:
            pairs = [bp.tolist() for bp in pairs_ind[ind]] + [[i.item(), j.item()]]
            best_pair = torch.tensor([x[bp[0], bp[1]] for bp in pairs]).argmax()
                
            multiplets += [pairs[k] for k in range(len(pairs)) if k!=best_pair]   
            
    pairs_ind = [[bp[0]+1, bp[1]+1] for bp in pairs_ind.tolist() if bp not in multiplets]
 
    return pairs_ind

def bp2matrix(L, base_pairs):
    matrix = torch.zeros((L, L))
    # base pairs are 1-based
    bp = torch.tensor(base_pairs) - 1
    if len(bp.shape) == 2:
        matrix[bp[:, 0], bp[:, 1]] = 1
        matrix[bp[:, 1], bp[:, 0]] = 1

    return matrix

def get_embed_dim(loader):
    # grab an element from the loader, which is represented by a dictionary with keys
    # `seq_ids`, `seq_embs_pad`, `contacts`, `Ls`
    batch_elem = next(iter(loader))
    # query for `seq_embs_pad` key (containing the embedding representations of all the sequences in the batch)
    # whose size will be batch_size x L x d
    return batch_elem["seq_embs_pad"].shape[2]
