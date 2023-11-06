#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# The models 

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, LayerNorm, MSELoss

import math
import copy


#%%
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


#%%
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        #print("I am in the attention head")
        #print(hidden_states.shape)
        #torch.Size([10, 25, 768])

        mixed_query_layer = self.query(hidden_states)
        #print(mixed_query_layer.shape)
        #torch.Size([10, 25, 768])

        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        #print(query_layer.shape)
        #torch.Size([10, 2, 25, 384])

        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        #print(attention_output.shape)
        #torch.Size([10, 25, 768])
        #print("End of the attention head")
        return attention_output, weights


#%%
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        #print("I am in the MLP")
        x = self.fc1(x)
        #print(x.shape)
        x = self.act_fn(x)
        #print(x.shape)
        x = self.dropout(x)
        x = self.fc2(x)
        #print(x.shape)
        x = self.dropout(x)
        #print("End of the MLP")
        return x 


#%%
# Define your deep learning model
class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.hidden_size = config.hidden_size
        self.in_channels = config.in_channels
        patch_size = config.patches["size"]

        # self.patch_embeddings = nn.Conv1d(in_channels=self.in_channels, 
        #                                   out_channels=config.hidden_size*self.in_channels, 
        #                                   kernel_size=patch_size, 
        #                                   stride=patch_size, 
        #                                   groups=self.in_channels)

        self.patch_embeddings =  FeatureExtractor(config)
        #torch.Size([10, 512, 12])

        #n_patches = 24 #self.in_channels*(int((input_length - patch_size) / patch_size ) + 1)
        no_tokens = config.in_channels * 12 + 1
        #print(n_patches)
        self.position_embeddings = nn.Parameter(torch.zeros(1, no_tokens, config.hidden_size))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        #print("I am in the Embeding Layer")
        #print(x.shape)
        #torch.Size([10, 2, 30000])
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        #print(cls_tokens.shape)
        #torch.Size([10, 1, 768])

        x = self.patch_embeddings(x)
        #print(x.shape)
        #torch.Size([10, 1536, 12])

        x = x.view(x.shape[0], self.in_channels, self.hidden_size, x.shape[2])
        #print(x.shape)
        #torch.Size([10, 2, 768, 12])

        x = x.transpose(-2, -3)
        #print(x.shape)
        #torch.Size([10, 768, 2, 12])

        x = x.flatten(2)
        #print(x.shape)
        #torch.Size([10, 768, 24])

        x = x.transpose(-1, -2)
        #print(x.shape)
        #torch.Size([10, 24, 768])

        x = torch.cat((cls_tokens, x), dim=1)
        #print(x.shape)
        #torch.Size([10, 25, 768])

        #embeddings = self.position_embeddings(x)
        k = self.position_embeddings
        #print(k.shape)
        #torch.Size([1, 25, 768])

        embeddings = x + k
        #print(embeddings.shape)
        #torch.Size([10, 25, 768])
        embeddings = self.dropout(embeddings)
        #print("End of the Embeding Layer")

        return embeddings


#%%
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        #print("I am in the Block Lalyer")
        #print(x.shape)
        #torch.Size([10, 25, 768])

        x = self.attention_norm(x)
        #print(x.shape)
        #torch.Size([10, 25, 768])

        x, weights = self.attn(x)
        #print(x.shape)
        #torch.Size([10, 25, 768])

        x = x + h
        #print(x.shape)
        #torch.Size([10, 25, 768])

        h = x
        x = self.ffn_norm(x)
        #print(x.shape)
        #torch.Size([10, 25, 768])

        x = self.ffn(x)
        #print(x.shape)
        #torch.Size([10, 25, 768])

        x = x + h
        #print(x.shape)
        #torch.Size([10, 25, 768])
        #print("End of the Block Lalyer")
        return x, weights


#%%
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


#%%
class Transformer(nn.Module):
    def __init__(self, config, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


#%%
class VisionTransformer(nn.Module):
    def __init__(self, config, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()

        no_tokens = config.in_channels * 12 + 1
        self.num_classes = config.num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, vis)
        self.head = Linear(config.hidden_size*no_tokens, self.num_classes)
        self.regress = Linear(config.hidden_size*no_tokens, 1)

    def forward(self, x, labels=None, cpcs=None):
        #print(x.shape)
        x, attn_weights = self.transformer(x)
        #print('I am in the VisionTransformer')
        #print(x.shape)
        #torch.Size([1, n_takens, 768])
        #print(x[:, 0].shape)
        #torch.Size([1, 768])
        x = x.flatten(1)
        #print(x.shape)
        logits = self.head(x) #[:, 0])
        regression = self.regress(x) #[:, 0])
        #torch.Size([1, 1000])
        #print(logits.shape)
        #print("I am output: ", logits)
        #print('End of the VisionTransformer')

        if labels is not None and cpcs is not None:
            # if batch size = 9
            #print("I am in the loss")
            #print(labels.view(-1))
            #print(logits.view(-1, self.num_classes).shape)
            loss_fct = CrossEntropyLoss()
            l2_loss = MSELoss()
            loss1 = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            loss2 = l2_loss(regression.squeeze(1), cpcs)
            #loss = loss1 + loss2
            #print(loss)
            return loss1, loss2 
        else:
            return logits, regression, attn_weights


#%%
class GroupNormConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)
        self.activation = nn.GELU()
        self.layer_norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, eps=1e-05, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        return x

#%%   
class NoLayerNormConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


#%%
class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config.in_channels

        out_channels  = int((config.hidden_size)*in_channels)

        self.conv_layers = nn.ModuleList([
            GroupNormConvLayer(in_channels, out_channels, 10, 5, groups=in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 5, 3, groups=in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 5, 3, groups=in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 5, 3, groups=in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 5, 2, groups=in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 3, 3, groups=in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 3, 3, groups=in_channels)
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

#%%
class FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config.in_channels
        out_channels  = int((config.hidden_size/2)*in_channels)

        self.layer_norm = nn.LayerNorm(out_channels, eps=1e-05)
        self.projection = nn.Linear(out_channels, config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


#%%
######################################## 1D CNN Classifier ##########################################
class Classification_1DCNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config.in_channels
        self.num_classes = config.num_classes

        self.feature_extractor = FeatureExtractor(config)
        self.feature_projection = FeatureProjection(config)
        
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        
    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")
        return outputs

    def forward(self, inputs, labels=None, **kwargs):
        #print("="*10)
        #print(inputs.shape)
        #torch.Size([5, 660000])
        
        x = inputs #.unsqueeze(1)
        #print(x.shape)
        #torch.Size([5, 1, 660000])
        
        x = self.feature_extractor(x)
        #print(x.shape)
        #torch.Size([5, 512, 2062])
        
        x = x.transpose(1, 2)
        #print(x.shape)
        #torch.Size([5, 2062, 512])
        
        x = self.feature_projection(x)
        #print(x.shape)
        #torch.Size([5, 2062, 768])
        
        x = self.merged_strategy(x, mode="mean")
        #print(merged_features_proj.shape)
        #torch.Size([5, 768])
        
        logits = self.classifier(x)
        #print(logits.shape)
        #torch.Size([5, 10])

        if labels is not None:
            # if batch size = 9
            #print("I am in the loss")
            #print(labels.view(-1))
            #print(logits.view(-1, self.num_classes).shape)
            #loss_fct = BCEWithLogitsLoss()
            #loss =  loss_fct(logits.squeeze(), labels.float())#
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1)) #loss_fct(logits, labels) #
            #print(loss)
            return loss 
        else:
            return logits
