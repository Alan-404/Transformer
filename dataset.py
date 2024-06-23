from torch.utils.data import Dataset

class TransformerDataset(Dataset):
    def __init__(self, manifest_path: str) -> None:
        super().__init__()