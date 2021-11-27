import torch

class CELoss(torch.nn.Module):
    def __init__(self, device):
        super(CELoss, self).__init__()
        self.device=device

    def forward(self, s_matrix):
        criterion = torch.nn.CrossEntropyLoss().cuda()
        spk_count = s_matrix.shape[0]
        seg_length = s_matrix.shape[1]

        out = s_matrix.reshape(spk_count * seg_length, spk_count)
        labels = torch.zeros((spk_count, seg_length), dtype=torch.long).to(self.device)
        for i in range(spk_count):
            labels[i] = torch.full((seg_length, ), i, dtype=torch.long)

        labels = labels.reshape(spk_count * seg_length)
        return criterion(out, labels)



def similarity_per_speaker(d_vectors):
    seg_count = d_vectors.shape[1]
    d_vects_t = torch.transpose(d_vectors, 0, 1)
    dimension = d_vectors.shape[2]
    spk_count = d_vectors.shape[0]
    d_vects_t = d_vects_t.repeat(spk_count,1,1)
    d_vects_t = d_vects_t.reshape(spk_count,seg_count,spk_count,dimension) 

    c = torch.mean(d_vectors, axis=1)
    c = c.unsqueeze(1)
    c = c.repeat(1, seg_count*spk_count ,1 )
    c = c.reshape(spk_count,seg_count,spk_count,dimension) 

    cos = torch.nn.CosineSimilarity(dim=3, eps=1e-6)
    return cos(d_vects_t, c)
