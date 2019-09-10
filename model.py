import math
import copy
import pdb
import re

from torch_scatter import scatter_add, scatter_mean
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import (BertAttention, BertEncoder,
                                              BertIntermediate, BertLayerNorm,
                                              BertModel,
                                              BertOutput, BertPooler)
from pytorch_pretrained_bert.modeling import \
    BertPreTrainedModel as PreTrainedBertModel  # Thenized,
from pytorch_pretrained_bert.modeling import BertSelfOutput, gelu
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

from utils import bundle_part_to_batch, find_start_end_before_tokenized


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

    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # layer = BertXLayer(config)
    
    def reset_parameters(self):
        glorot(self.W)

    def forward(self, adj, semantics, attention_masks):
        '''
        adj: (n, n)
        semantics: (n, seq_len, hidden_size)
        attention_mask: (n, seq_len, hidden_size)
        '''
        edge_list = adj.nonzero()
        n = semantics.shape[0]
        assert adj.shape[0] == n
        h = torch.mm(semantics[:, 0], self.W)
        h_x = h[edge_list[:, 0]]
        # h_y = h[edge_list[:, 1], 0]

        # Message passing
        # index = edge_list[:, 1].long()
        # src = torch.ones_like(index).float()
        # h_num = torch.zeros(n, device=src.device)
        # scatter_add(src, index, out=h_num)

        index = edge_list[:, 1].long()
        src = h_x
        h_sum = torch.zeros_like(h)
        scatter_mean(src, index, out=h_sum, dim=0)

        # h_sum /= h_num

        return gelu(h_sum)


class XAttn(nn.Module):

    def __init__(self, input_size, config, n_layers=1):
        super(XAttn, self).__init__()
        layer = MPLayer(input_size)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.predict = MLP(input_sizes=(input_size, input_size, 1))

    def forward(self, adj, semantics, attention_masks):
        for layer_module in self.layer:
            semantics = layer_module(adj, semantics, attention_masks)
        return self.predict(semantics).squeeze(-1)


class BertEmbeddingsPlus(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, max_sentence_type=30):
        super(BertEmbeddingsPlus, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.sentence_type_embeddings = nn.Embedding(
            max_sentence_type, config.hidden_size
        )
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings((token_type_ids > 0).long())
        sentence_type_embeddings = self.sentence_type_embeddings(token_type_ids)

        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + sentence_type_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModelPlus(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddingsPlus(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(
        self, input_ids, token_type_ids=None, attention_mask=None, output_hidden=-4
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask, output_all_encoded_layers=True
        )
        sequence_output = encoded_layers[-1]
        # pooled_output = self.pooler(sequence_output)
        encoded_layers, hidden_layers = (
            encoded_layers[-1],
            encoded_layers[output_hidden],
        )
        return encoded_layers, hidden_layers


class BertForMultiHopQuestionAnswering(PreTrainedBertModel):
    def __init__(self, config):
        super(BertForMultiHopQuestionAnswering, self).__init__(config)
        self.bert = BertModelPlus(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 4)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        sep_positions=None,
        hop_start_weights=None,
        hop_end_weights=None,
        ans_start_weights=None,
        ans_end_weights=None,
        B_starts=None,
        allow_limit=(0, 0),
    ):
        """ Extract spans by System 1.
        
        Args:
            input_ids (LongTensor): Token ids of word-pieces. (batch_size * max_length)
            token_type_ids (LongTensor): The A/B Segmentation in BERTs. (batch_size * max_length)
            attention_mask (LongTensor): Indicating whether the position is a token or padding. (batch_size * max_length)
            sep_positions (LongTensor): Positions of [SEP] tokens, mainly used in finding the num_sen of supporing facts. (batch_size * max_seps)
            hop_start_weights (Tensor): The ground truth of the probability of hop start positions. The weight of sample has been added on the ground truth. 
                (You can verify it by examining the gradient of binary cross entropy.)
            hop_end_weights ([Tensor]): The ground truth of the probability of hop end positions.
            ans_start_weights ([Tensor]): The ground truth of the probability of ans start positions.
            ans_end_weights ([Tensor]): The ground truth of the probability of ans end positions.
            B_starts (LongTensor): Start positions of sentence B.
            allow_limit (tuple, optional): An Offset for negative threshold. Defaults to (0, 0).
        
        Returns:
            [type]: [description]
        """
        batch_size = input_ids.size()[0]
        device = input_ids.get_device() if input_ids.is_cuda else torch.device("cpu")
        sequence_output, hidden_output = self.bert(
            input_ids, token_type_ids, attention_mask
        )
        semantics = hidden_output
        # semantics = hidden_output[:, 0]
        # Some shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
        if sep_positions is None:
            return semantics  # Only semantics, used in bundle forward
        else:
            max_sep = sep_positions.size()[-1]
        if max_sep == 0:
            empty = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            return (
                empty,
                empty,
                semantics,
                empty,
            )  # Only semantics, used in eval, the same ``empty'' variable is a mistake in general cases but simple

        # Predict spans
        logits = self.qa_outputs(sequence_output)
        hop_start_logits, hop_end_logits, ans_start_logits, ans_end_logits = logits.split(
            1, dim=-1
        )
        hop_start_logits = hop_start_logits.squeeze(-1)
        hop_end_logits = hop_end_logits.squeeze(-1)
        ans_start_logits = ans_start_logits.squeeze(-1)
        ans_end_logits = ans_end_logits.squeeze(-1)  # Shape: [batch_size, max_length]

        if hop_start_weights is not None:  # Train mode
            lgsf = torch.nn.LogSoftmax(
                dim=1
            )  # If there is no targeted span in the sentence, start_weights = end_weights = 0(vec)
            hop_start_loss = -torch.sum(
                hop_start_weights * lgsf(hop_start_logits), dim=-1
            )
            hop_end_loss = -torch.sum(hop_end_weights * lgsf(hop_end_logits), dim=-1)
            ans_start_loss = -torch.sum(
                ans_start_weights * lgsf(ans_start_logits), dim=-1
            )
            ans_end_loss = -torch.sum(ans_end_weights * lgsf(ans_end_logits), dim=-1)
            hop_loss = torch.mean((hop_start_loss + hop_end_loss)) / 2
            ans_loss = torch.mean((ans_start_loss + ans_end_loss)) / 2
        else:
            # In eval mode, find the exact top K spans.
            K_hop, K_ans = 3, 1
            hop_preds = torch.zeros(
                batch_size, K_hop, 3, dtype=torch.long, device=device
            )  # (start, end, sen_num)
            ans_preds = torch.zeros(
                batch_size, K_ans, 3, dtype=torch.long, device=device
            )
            ans_start_gap = torch.zeros(batch_size, device=device)
            for u, (start_logits, end_logits, preds, K, allow) in enumerate(
                (
                    (
                        hop_start_logits,
                        hop_end_logits,
                        hop_preds,
                        K_hop,
                        allow_limit[0],
                    ),
                    (
                        ans_start_logits,
                        ans_end_logits,
                        ans_preds,
                        K_ans,
                        allow_limit[1],
                    ),
                )
            ):
                for i in range(batch_size):
                    if sep_positions[i, 0] > 0:
                        values, indices = start_logits[i, B_starts[i] :].topk(K)
                        for k, index in enumerate(indices):
                            if values[k] <= start_logits[i, 0] - allow:  # not golden
                                if u == 1:  # For ans spans
                                    ans_start_gap[i] = start_logits[i, 0] - values[k]
                                break
                            start = index + B_starts[i]
                            # find ending
                            for j, ending in enumerate(sep_positions[i]):
                                if ending > start or ending <= 0:
                                    break
                            if ending <= start:
                                break
                            ending = min(ending, start + 10)
                            end = torch.argmax(end_logits[i, start:ending]) + start
                            preds[i, k, 0] = start
                            preds[i, k, 1] = end
                            preds[i, k, 2] = j
        return (
            (hop_loss, ans_loss, semantics)
            if hop_start_weights is not None
            else (hop_preds, ans_preds, semantics, ans_start_gap)
        )


class CognitiveGNN(nn.Module):
    def __init__(self, hidden_size, config):
        super(CognitiveGNN, self).__init__()
        self.gcn = GCN(hidden_size)
        self.both_net = MLP((hidden_size, hidden_size, 1))
        self.select_net = MLP((hidden_size, hidden_size, 1))

    def forward(self, bundle, model, device):
        batch = bundle_part_to_batch(bundle)
        batch = tuple(t.to(device) for t in batch)
        hop_loss, ans_loss, semantics = model(
            *batch
        )
        attention_mask = batch[2]
        # Shape of semantics: [num_para, seq_len, hidden_size]
        # # Shape of semantics: [num_para, hidden_size]
        num_additional_nodes = len(bundle.additional_nodes)

        if num_additional_nodes > 0:
            max_length_additional = max([len(x) for x in bundle.additional_nodes])
            ids = torch.zeros(
                (num_additional_nodes, max_length_additional),
                dtype=torch.long,
                device=device,
            )
            segment_ids = torch.zeros(
                (num_additional_nodes, max_length_additional),
                dtype=torch.long,
                device=device,
            )
            input_mask = torch.zeros(
                (num_additional_nodes, max_length_additional),
                dtype=torch.long,
                device=device,
            )
            for i in range(num_additional_nodes):
                length = len(bundle.additional_nodes[i])
                ids[i, :length] = torch.tensor(
                    bundle.additional_nodes[i], dtype=torch.long
                )
                input_mask[i, :length] = 1
            additional_attention_mask = input_mask
            additional_semantics = model(ids, segment_ids, input_mask)

            if semantics.shape[1] > additional_semantics.shape[1]:
                zero_shape = list(additional_semantics.shape)
                zero_shape[1] = semantics.shape[1] - additional_semantics.shape[1]
                additional_semantics = torch.cat((additional_semantics, torch.zeros(zero_shape).to(device)), dim=1)
                additional_attention_mask = torch.cat((additional_attention_mask, torch.zeros(zero_shape[:-1], dtype=torch.long).to(device)), dim=1)
            elif semantics.shape[1] < additional_semantics.shape[1]:
                zero_shape = list(semantics.shape)
                zero_shape[1] = additional_semantics.shape[1] - semantics.shape[1]
                semantics = torch.cat((additional_semantics, torch.zeros(zero_shape).to(device)), dim=1)
                attention_mask = torch.cat((attention_mask, torch.zeros(zero_shape[:-1], dtype=torch.long).to(device)), dim=1)
            semantics = torch.cat((semantics, additional_semantics), dim=0)
            attention_mask = torch.cat((attention_mask, additional_attention_mask), dim=0)
            attention_mask = (1.0 - attention_mask) * -10000.0
            

        assert semantics.size()[0] == bundle.adj.size()[0]
        assert semantics.shape[0] == attention_mask.shape[0]

        if bundle.question_type == 0:  # Wh-
            pred = self.gcn(bundle.adj.to(device), semantics, attention_mask)
            ce = torch.nn.CrossEntropyLoss()
            final_loss = ce(
                pred.unsqueeze(0),
                torch.tensor([bundle.answer_id], dtype=torch.long, device=device),
            )
        else:
            x, y, ans = bundle.answer_id
            ans = torch.tensor(ans, dtype=torch.float, device=device)
            diff_sem = semantics[x][0] - semantics[y][0]
            classifier = self.both_net if bundle.question_type == 1 else self.select_net
            final_loss = 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(
                classifier(diff_sem).squeeze(-1), ans.to(device)
            )
        return hop_loss, ans_loss, final_loss


if __name__ == "__main__":
    BERT_MODEL = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    orig_text = "".join(
        [
            "Theatre Centre is a UK-based theatre company touring new plays for young audiences aged 4 to 18, founded in 1953 by Brian Way, the company has developed plays by writers including which British writer, dub poet and Rastafarian?",
            " It is the largest urban not-for-profit theatre company in the country and the largest in Western Canada, with productions taking place at the 650-seat Stanley Industrial Alliance Stage, the 440-seat Granville Island Stage, the 250-seat Goldcorp Stage at the BMO Theatre Centre, and on tour around the province.",
        ]
    )
    tokenized_text = tokenizer.tokenize(orig_text)
    print(len(tokenized_text))
