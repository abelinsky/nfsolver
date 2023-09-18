import torch


class HydraulicsLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, P, imbalance):
        psrc = data.x[..., -1]
        sidx = torch.where(psrc > 0)  # узлы с заданными значениями давления
        snidx = torch.where(psrc == 0)[0]  # узлы с незаданными значениями давления

        ploss = torch.nn.MSELoss()

        pl = ploss(psrc[sidx], P.view(-1)[sidx])
        ql = torch.square(imbalance[snidx]).mean()

        beta = 1
        return beta * ql + (1 - beta) * pl
