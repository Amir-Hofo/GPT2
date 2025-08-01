from packages import *

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads= config["num heads"]
        self.feature_dimension= config["feature dimension"]
        self.head_dimension= self.feature_dimension // self.num_heads
        self.fc1= nn.Linear(self.feature_dimension, 3* self.feature_dimension)
        self.fc2= nn.Linear(self.feature_dimension, self.feature_dimension)

    def forward(self, x):
        batch_size, seq_len= x.shape[0], x.shape[1]
        qkv= self.fc1(x).reshape(batch_size, seq_len, self.num_heads, 
                                 3*(self.head_dimension)).permute(0, 2, 1, 3)
        Q, K, V= qkv.split(self.head_dimension, dim= -1)
        del x, qkv
        sdpa= scaled_dot_product_attention(Q, K, V, is_causal= True)
        sdpa= sdpa.reshape(batch_size, seq_len, self.feature_dimension)
        del Q, K, V
        return self.fc2(sdpa)



class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config= config
        self.feature_dimension= self.config["feature dimension"]

        self.mha_layernorm= nn.LayerNorm(self.feature_dimension)
        self.masked_mha= MultiHeadAttention(self.config)
        self.mha_dropout= nn.Dropout(self.config["mha dropout"])

        self.ffnn_layernorm= nn.LayerNorm(self.feature_dimension)
        self.ffnn= nn.Sequential(nn.Linear(self.feature_dimension, 4*self.feature_dimension),
                                 nn.GELU(),
                                 nn.Linear(4*self.feature_dimension, self.feature_dimension))
        self.ffnn_dropout= nn.Dropout(self.config["ffnn dropout"])


    def forward(self, x):
        mha= self.mha_dropout(self.masked_mha(self.mha_layernorm(x))) + x
        ffnn= self.ffnn_dropout(self.ffnn(self.ffnn_layernorm(mha))) + mha
        return ffnn
    


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config= config
        self.token_embedding= nn.Embedding(self.config["vocab size"], self.config["feature dimension"])
        self.position_embedding= nn.Embedding(self.config["max position"], self.config["feature dimension"])
        self.dropout= nn.Dropout(self.config["dropout"])
        self.decoder_blocks= nn.ModuleList([DecoderBlock(self.config) for _ in range(self.config["num layers"])])
        self.layer_norm= nn.LayerNorm(self.config["feature dimension"])

    def forward(self, x):
        pos_emb= self.position_embedding(torch.arange(x.size(1), device= x.device))
        x= self.dropout(self.token_embedding(x) + pos_emb)
        for block in self.decoder_blocks: x= block(x)
        return self.layer_norm(x)