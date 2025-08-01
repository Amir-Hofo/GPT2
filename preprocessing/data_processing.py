from packages import *


def dataset_loader_fn(repo_id= "roneneldan/TinyStories"):
    return load_dataset(repo_id)



def tokenizer_fn(dataset, vocab_size= 10_000, min_frequency= 2, save_tokenizer= True):
    unk, eos= "|<unk>|", "<|endoftext|>"
    tokenizer= Tokenizer(models.BPE(unk_token= unk))
    tokenizer.pre_tokenizer= pre_tokenizers.ByteLevel(add_prefix_space= False)
    trainer= trainers.BpeTrainer(vocab_size= vocab_size, 
                                 min_frequency= min_frequency,
                                 special_tokens= [unk, eos] )

    tokenizer.train_from_iterator(dataset["train"]["text"], trainer)
    print(10 * "--", " vocab size ", 10 * "--")
    print(tokenizer.get_vocab_size())

    tokenizer.post_processor= processors.TemplateProcessing(
        single= f"{eos} $A",
        special_tokens= [(eos, tokenizer.token_to_id(eos))]
        )

    tokenizer.decoder= decoders.ByteLevel(add_prefix_space= False)

    if save_tokenizer: tokenizer.save(f"./preprocessing/custom_tokenizer_{vocab_size//1000}K.json")
    return tokenizer



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
    


def data_preparation(dataset, tokenizer, seq_len= 128, batch_size= 64, 
                     shuffle_st= True, all_workers= True):
    data= CustomDataset(dataset, tokenizer, seq_len)
    del dataset, tokenizer
    pin_memory= torch.cuda.is_available()
    if all_workers:
        data_loader= DataLoader(data, batch_size= batch_size, shuffle= shuffle_st, 
                                pin_memory= pin_memory, num_workers= os.cpu_count())
    else:
        data_loader= DataLoader(data, batch_size= batch_size, shuffle= shuffle_st, pin_memory= pin_memory)
    return data_loader
    

    
class IterCustomDataset(IterableDataset):

    def __init__(self, text_samples, tokenizer, group_size=10, seq_len=256):
        """
        Args:
            text_samples (List[str]): List of raw text strings (e.g., stories).
            tokenizer (Tokenizer): A HuggingFace-style tokenizer with .encode().ids support.
            group_size (int): Number of text samples to concatenate per chunk before tokenization.
            seq_len (int): Target length of each token sequence for training.
        """
        self.text_samples = text_samples
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.group_size = group_size

    def __iter__(self):
        buffer = []

        # Iterate through the dataset in chunks of `group_size`
        for i in range(0, len(self.text_samples), self.group_size):
            # Join multiple stories with a special token and tokenize the batch
            sample = ' <|endoftext|> '.join(self.text_samples[i:i + self.group_size]) + ' <|endoftext|>'
            tokens = self.tokenizer.encode(sample).ids
            buffer += tokens

            # Yield token chunks of size seq_len + 1 for input-target training pairs
            yield from (
                torch.from_numpy(np.array(buffer[j:j + self.seq_len+1], dtype=np.int64))
                for j in range(0, len(buffer) - self.seq_len, self.seq_len)
            )

            # Keep leftover tokens that didn't form a full chunk
            buffer = buffer[(len(buffer) // self.seq_len) * self.seq_len:]
