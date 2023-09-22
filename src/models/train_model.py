import torch


class HydraulicsLoss(torch.nn.Module):
    def __init__(self, name=""):
        super().__init__()
        self.name = name

    def forward(self, data, P, imbalance):
        """Вычисление критерия.

        Args:
            data (Data): Информация о графе.
            P (float): Значения давления газа.
            imbalance (float): Небалансы в узлах, shape [Количество слоев, Количество вершин, 1].

        Returns:
            float: Значение критерия.
        """
        # print(f"{imbalance.shape=}")
        psrc = data.x[..., -1]
        snidx = torch.where(psrc == 0)[0]  # узлы с незаданными значениями давления
        # print(f">> === {self.name} === loss calculating: {imbalance.shape}")
        # return torch.square(imbalance[snidx]).mean()
        return torch.square(imbalance[:, snidx, ...]).mean()


class MultiHydraulicsLoss(torch.nn.Module):
    def __init__(self, gamma: float, device):
        super().__init__()
        self.gamma = gamma
        self.device = device

    def forward(self, data, P, imbalances):
        K = len(imbalances)
        gammas = torch.tensor([self.gamma ** (K - k) for k in range(1, K + 1)]).to(
            self.device
        )

        # print(f">>> in multiloss {imbalances.shape=}")
        # print(f"{gammas=}")

        # losses = torch.tensor(
        #     [HydraulicsLoss()(data, P, imbalance) for imbalance in imbalances]
        # )
        # total_loss = torch.sum(gammas * losses)
        # print(f"{total_loss=}")

        # print(f"{imbalances.squeeze().shape=}")

        losses = HydraulicsLoss()(data, P, imbalances.squeeze())

        # print(f"{losses.shape=}")

        total_loss = torch.sum(gammas * losses)

        # print(f"{total_loss=}")

        return total_loss
