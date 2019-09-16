import copy
import math

import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import (BertAttention,
                                              BertIntermediate,
                                              BertModel, BertOutput, BertLayerNorm,
                                              BertLayer
                                              )
from pytorch_pretrained_bert.modeling import BertSelfOutput, gelu
# from torch_scatter import scatter_add, scatter_mean


class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Lang self-att and FFN layer
        self.lang_self_att = BertAttention(config)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def cross_att(
        self, lang_input, lang_attention_mask, visn_input, visn_attention_mask
    ):
        # Cross Attention
        lang_att_output = self.visual_attention(
            lang_input, visn_input, ctx_att_mask=visn_attention_mask
        )
        visn_att_output = self.visual_attention(
            visn_input, lang_input, ctx_att_mask=lang_attention_mask
        )
        return lang_att_output, visn_att_output

    def self_att(
        self, lang_input, lang_attention_mask, visn_input, visn_attention_mask
    ):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output = self.cross_att(
            lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask
        )
        lang_att_output, visn_att_output = self.self_att(
            lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask
        )
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        return lang_output, visn_output


class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob=0.2, bias=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                x = gelu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x


class GCN(nn.Module):
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.05)

    def __init__(self, input_size):
        super(GCN, self).__init__()
        self.diffusion = nn.Linear(input_size, input_size, bias=False)
        self.retained = nn.Linear(input_size, input_size, bias=False)
        self.predict = MLP(input_sizes=(input_size, input_size, 1))
        self.apply(self.init_weights)

    def forward(self, A, x):
        layer1_diffusion = A.t().mm(gelu(self.diffusion(x)))
        x = gelu(self.retained(x) + layer1_diffusion)
        layer2_diffusion = A.t().mm(gelu(self.diffusion(x)))
        x = gelu(self.retained(x) + layer2_diffusion)
        return self.predict(x).squeeze(-1)

def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

class MPLayer(nn.Module):

    def __init__(self, hidden_size, config):
        super().__init__()
        self.layer1 = BertLayer(config)
        self.layer2 = BertLayer(config)
        self.xlayer = BertLayer(config)
        # self.dense = nn.Linear(hidden_size, hidden_size)
        # self.norm = BertLayerNorm(hidden_size, eps=1e-12)

    def forward(self, adj, semantics, attention_masks):
        '''
        adj: (n, n)
        semantics: (n, seq_len, hidden_size)
        attention_mask: (n, seq_len, hidden_size)
        '''
        n = semantics.shape[0]
        assert adj.shape[0] == n

        semantics = self.layer1(semantics, attention_masks)
        # semantics = self.layer2(semantics, attention_masks)

        attention_masks = torch.zeros(1, 1, 1, semantics.shape[0]).to(semantics.device)

        semantics = self.xlayer(semantics[:, 0].unsqueeze(0), attention_masks)
        semantics = semantics.squeeze()

        return semantics

    # def forward(self, adj, semantics, attention_masks):
    #     '''
    #     adj: (n, n)
    #     semantics: (n, seq_len, hidden_size)
    #     attention_mask: (n, seq_len, hidden_size)
    #     '''
    #     edge_list = adj.nonzero()
    #     n = semantics.shape[0]
    #     assert adj.shape[0] == n
    #     _, h = self.xlayer(semantics[edge_list[:, 0]], attention_masks[edge_list[:, 0]],
    #                        semantics[edge_list[:, 1]], attention_masks[edge_list[:, 1]])
    #     h = h[:, 0]

    #     # Message passing
    #     index = edge_list[:, 1].long()
    #     # src = self.dense(h)
    #     src = h
    #     h_sum = torch.zeros_like(semantics[:, 0])
    #     # h_sum = torch.zeros_like(semantics)
    #     # scatter_add(src, index, out=h_sum, dim=0)
    #     scatter_mean(src, index, out=h_sum, dim=0)

    #     # h_sum /= h_num

    #     return gelu(h_sum)
