from packages import *

def print_title(title: str):
    s= (70-len(title)-1)//2
    e= 70-len(title)-s-2
    print()
    print(70*"#")
    print(s*"#", title, e*"#")
    print(70*"#")


def num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/ 1e6


def scales_dot_product_attention_fn(Q, K, V):
    scores= (Q @ K.transpose(-2, -1)) / math.sqrt(K.shape[-1])
    mask= torch.tril(torch.ones(scores.shape[-2:])).to(scores.device)
    scores= scores.masked_fill(mask == 0, float(-torch.inf))
    return scores.softmax(dim=-1) @ V