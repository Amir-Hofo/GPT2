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


def read_config(config_name: str, root: str) -> SimpleNamespace:
    with open(f"{root}config/{config_name}.json") as f:
        return SimpleNamespace(**json.load(f))