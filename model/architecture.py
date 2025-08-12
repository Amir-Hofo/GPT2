from packages import *
from model.utils import text_generation_fn


def scales_dot_product_attention_fn(Q, K, V):
    scores= (Q @ K.transpose(-2, -1)) / math.sqrt(K.shape[-1])
    mask= torch.tril(torch.ones(scores.shape[-2:])).to(scores.device)
    scores= scores.masked_fill(mask == 0, float(-torch.inf))
    return scores.softmax(dim=-1) @ V



class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads= config.num_heads
        self.feature_dimension= config.feature_dimension
        self.head_dimension= self.feature_dimension // self.num_heads
        self.fc1= nn.Linear(self.feature_dimension, 3* self.feature_dimension, bias= config.bias)
        self.fc2= nn.Linear(self.feature_dimension, self.feature_dimension, bias= config.bias)
        self.fc2.residual= True

    def forward(self, x):
        Q, K, V= self.fc1(x).view(x.shape[0], x.shape[1], 3*self.num_heads, self.head_dimension).transpose(1, 2).chunk(3, dim=-3)
        x_attention= scales_dot_product_attention_fn(Q, K, V).transpose(1, 2).contiguous().view(x.shape)
        # x_attention= nn.functional.scaled_dot_product_attention(Q, K, V, is_causal= True).transpose(1, 2).contiguous().view(x.shape)
        return self.fc2(x_attention)



class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config= config
        self.feature_dimension= self.config.feature_dimension
        self.ffnn_expand= self.config.ffnn_expand
        self.use_checkpoint= config.use_checkpoint

        self.mha_layernorm= nn.LayerNorm(self.feature_dimension)
        self.masked_mha= MultiHeadAttention(self.config)
        self.mha_dropout= nn.Dropout(self.config.mha_dropout)

        self.ffnn_layernorm= nn.LayerNorm(self.feature_dimension)
        self.ffnn= nn.Sequential(nn.Linear(self.feature_dimension, 
                                           int(self.ffnn_expand* self.feature_dimension), 
                                           bias= config.bias),
                                 nn.GELU(),
                                 nn.Linear(int(self.ffnn_expand* self.feature_dimension), 
                                           self.feature_dimension, bias= config.bias))
        self.ffnn[-1].residual= True
        self.ffnn_dropout= nn.Dropout(self.config.ffnn_dropout)

    def forward(self, x):
        if self.config.use_checkpoint:
            x= x+ self.mha_dropout(checkpoint(self._masked_mha_forward, self.mha_layernorm(x), use_reentrant= False))
            x= x+ self.ffnn_dropout(checkpoint(self._ffnn_forward, self.ffnn_layernorm(x), use_reentrant= False))
        else:
            x= x+ self.mha_dropout(self.masked_mha(self.mha_layernorm(x)))
            x= x+ self.ffnn_dropout(self.ffnn(self.ffnn_layernorm(x)))
        return x

    def _masked_mha_forward(self, x):
        return self.masked_mha(x)

    def _ffnn_forward(self, x):
        return self.ffnn(x)



class PytorchDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config= config
        self.use_checkpoint= config.use_checkpoint
        self.decoder_layer= nn.TransformerEncoderLayer(
                                        d_model= self.config.feature_dimension,
                                        nhead= self.config.num_heads,
                                        dim_feedforward= int(self.config.ffnn_expand * self.config.feature_dimension),
                                        dropout= self.config.decoder_dropout,
                                        activation= "gelu",
                                        batch_first= True)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers= self.config.num_layers)

    def forward(self, x):
        causal_mask= nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        if self.config.use_checkpoint:
            def checkpoint_wrapper(module, *inputs):
                return checkpoint(module, *inputs, use_reentrant= False)
            return checkpoint_wrapper(self.decoder, x, causal_mask)
        else:
            return self.decoder(x, mask= causal_mask)



class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config= config
        self.token_embedding= nn.Embedding(self.config.vocab_size, self.config.feature_dimension)
        self.position_embedding= nn.Embedding(self.config.seq_len, self.config.feature_dimension)
        self.dropout= nn.Dropout(self.config.dropout)
        
        if self.config.custom_decoder:
            self.decoder_blocks= nn.ModuleList([DecoderBlock(self.config) for _ in range(self.config.num_layers)])
        else: self.decoder= PytorchDecoder(self.config)

        self.layer_norm= nn.LayerNorm(self.config.feature_dimension)
        self.fc= nn.Linear(self.config.feature_dimension, self.config.vocab_size, bias= self.config.bias)
        if self.config.weight_tying: self.fc.weight= self.token_embedding.weight

        self.apply(self._init_weights)


    def _init_weights(self, module, std= 0.02):
        if isinstance(module, nn.Linear):
            if hasattr(module, 'residual'):
                std*= (2* self.config.num_layers)** -0.5

            nn.init.normal_(module.weight, mean= 0.0, std= std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean= 0.0, std= std)

    
    def text_generator(self, prompt, tokenizer, max_length= 128, temperature= 1.5, top_k= 5):
        return text_generation_fn(self, tokenizer, prompt, max_length, temperature, top_k)


    def forward(self, x):
        pos_emb= self.position_embedding(torch.arange(x.size(1), device= x.device))
        x= self.dropout(self.token_embedding(x) + pos_emb)
        if self.config.custom_decoder:
            for block in self.decoder_blocks: x= block(x)
        else: x= self.decoder(x)
        return self.fc(self.layer_norm(x))