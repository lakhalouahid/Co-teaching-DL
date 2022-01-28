import torch
import torch.nn.functional as F

def loss_coteaching(y_1, y_2, t, num_remember):
    """Co-teaching loss exchange"""
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_1_sorted = torch.argsort(loss_1.data)
    ind_2_sorted = torch.argsort(loss_2.data)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember
