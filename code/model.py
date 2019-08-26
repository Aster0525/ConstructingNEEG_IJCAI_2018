import torch
import torch.nn as nn


class TensorComposition(nn.Module):
    def __init__(self, k, n1, n2):
        super(TensorComposition, self).__init__()
        self.k = k
        self.n1 = n1
        self.n2 = n2
        self.t = nn.Parameter(torch.FloatTensor(k, n1, n2))
        # torch.nn.init.xavier_uniform_(self.t, gain=1)
        torch.nn.init.normal_(self.t, std=0.01)

    def forward(self, a, b):
        '''
        a: (*, n1)
        b: (*, n2)
        '''
        k = self.k
        n1 = self.n1
        n2 = self.n2
        output_shape = tuple(a.size()[:-1] + (k,))  # (*, k)
        a = a.contiguous().view(-1, n1)             # (m, n1)
        b = b.contiguous().view(-1, n2)             # (m, n2)
        o = torch.einsum('ijk,cj,ck->ci', [self.t, a, b])
        return o.view(output_shape)


class LowRankTensorComposition(nn.Module):
    def __init__(self, k, n, r):
        super(LowRankTensorComposition, self).__init__()
        self.k = k
        self.n = n
        self.r = r
        self.t_l = nn.Parameter(torch.FloatTensor(k, n, r))
        self.t_r = nn.Parameter(torch.FloatTensor(k, r, n))
        self.t_diag = nn.Parameter(torch.FloatTensor(k, n))
        torch.nn.init.normal_(self.t_l, std=0.01)
        torch.nn.init.normal_(self.t_r, std=0.01)
        torch.nn.init.normal_(self.t_diag, std=0.01)
        # torch.nn.init.xavier_uniform_(self.t_l, gain=1)
        # torch.nn.init.xavier_uniform_(self.t_r, gain=1)
        # torch.nn.init.xavier_uniform_(self.t_diag, gain=1)

    def forward(self, a, b):
        '''
        a: (*, n)
        b: (*, n)
        '''
        k = self.k
        n = self.n
        output_shape = tuple(a.size()[:-1]) + (k,)
        # make t_diag
        t_diag = []
        for v in self.t_diag:
            t_diag.append(torch.diag(v))    # (n, n)
        t_diag = torch.stack(t_diag)        # (k, n, n)
        t = torch.bmm(self.t_l, self.t_r) + t_diag      # (k, n, n)
        a = a.contiguous().view(-1, n)      # (m, n)
        b = b.contiguous().view(-1, n)      # (m, n)
        o = torch.einsum('ijk,cj,ck->ci', [t, a, b])
        return o.view(output_shape)


class NeuralTensorNetwork(nn.Module):
    def __init__(self, k1, k2, emb_dim):
        super(NeuralTensorNetwork, self).__init__()
        self.subj_verb_comp = TensorComposition(k1, emb_dim, emb_dim)
        self.verb_obj_comp = TensorComposition(k1, emb_dim, emb_dim)
        self.final_comp = TensorComposition(k2, k1, k1)
        self.linear1 = nn.Linear(2 * emb_dim, k1)
        self.linear2 = nn.Linear(2 * emb_dim, k1)
        self.linear3 = nn.Linear(2 * k1, k2)
        self.tanh = nn.Tanh()

    def forward(self, subj_emb, verb_emb, obj_emb):
        '''
        subj_emb: (*, emb_dim)
        verb_emb: (*, emb_dim)
        obj_emb:  (*, emb_dim)
        '''
        # r1 = subj_verb_comp(subj, verb)
        tensor_comp = self.subj_verb_comp(subj_emb, verb_emb)   # (*, k1)
        cat = torch.cat((subj_emb, verb_emb), dim=-1)           # (*, 2*emb_dim)
        linear = self.linear1(cat)              # (*, k1)
        r1 = self.tanh(tensor_comp + linear)    # (*, k1)
        # r2 = verb_obj_comp(verb, obj)
        tensor_comp = self.verb_obj_comp(verb_emb, obj_emb)     # (*, k1)
        cat = torch.cat((verb_emb, obj_emb), dim=-1)            # (*, 2*emb_dim)
        linear = self.linear2(cat)              # (*, k1)
        r2 = self.tanh(tensor_comp + linear)    # (*, k1)
        # r3 = final_comp(r1, r2)
        tensor_comp = self.final_comp(r1, r2)   # (*, k2)
        cat = torch.cat((r1, r2), dim=-1)       # (*, 2*k1)
        linear = self.linear3(cat)              # (*, k2)
        r3 = self.tanh(tensor_comp + linear)    # (*, k2)
        return r3


class LowRankNeuralTensorNetwork(NeuralTensorNetwork):
    def __init__(self, k1, k2, r, emb_dim):
        super(NeuralTensorNetwork, self).__init__()
        self.subj_verb_comp = LowRankTensorComposition(k1, emb_dim, r)
        self.verb_obj_comp = LowRankTensorComposition(k1, emb_dim, r)
        self.final_comp = LowRankTensorComposition(k2, k1, r)
        self.linear1 = nn.Linear(2 * emb_dim, k1)
        self.linear2 = nn.Linear(2 * emb_dim, k1)
        self.linear3 = nn.Linear(2 * k1, k2)
        self.tanh = nn.Tanh()


class RoleFactoredTensorModel(nn.Module):
    def __init__(self, k, emb_dim):
        super(RoleFactoredTensorModel, self).__init__()
        self.k = k
        self.tensor_comp = TensorComposition(k, emb_dim, emb_dim)
        self.w = nn.Linear(2 * k, k, bias=False)

    def forward(self, subj_emb, verb_emb, obj_emb):
        '''
        subj_emb: (*, n)
        verb_emb: (*, n)
        obj_emb:  (*, n)
        '''
        vs = self.tensor_comp(verb_emb, subj_emb)       # (*, k)
        vo = self.tensor_comp(verb_emb, obj_emb)        # (*, k)
        cat = torch.cat((vs, vo), dim=-1)   # (*, 2*k)
        return self.w(cat)                  # (*, k)
