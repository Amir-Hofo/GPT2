from packages import *

def dataset_loader_fn(repo_id= "roneneldan/TinyStories"):
    return load_dataset(repo_id)

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len):
        super().__init__()
        self.tokenizer, self.seq_len= tokenizer, seq_len+ 1
        self.data= list(chain.from_iterable(tokenizer.encode(text).ids for text in data["text"]))
        self.num_rows= (len(self.data) // self.seq_len)
        self.data= torch.Tensor(self.data[: self.num_rows * self.seq_len]).reshape(self.num_rows, self.seq_len)
    
    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, -1]