import logging
from unittest import result
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
import pdb

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class syncvisMultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                    newk1 = k.replace("static_query", "query_feat1")
                    newk2 = k.replace("static_query", "query_feat2")
                    newk3 = k.replace("static_query", "query_feat3")
                    newk4 = k.replace("static_query", "query_feat4")
                    newk5 = k.replace("static_query", "query_feat5")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    state_dict[newk1] = state_dict[k]
                    state_dict[newk2] = state_dict[k]
                    state_dict[newk3] = state_dict[k]
                    state_dict[newk4] = state_dict[k]
                    state_dict[newk5] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        syncvis_last_layer_num: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        self.FFN_1 =  FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm,)
        self.FFN_2 =  FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm,)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.multi_attn = nn.MultiheadAttention(256, 8)
        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_feat1 = nn.Embedding(num_queries, hidden_dim)
        self.query_feat2 = nn.Embedding(num_queries, hidden_dim)
        self.query_feat3 = nn.Embedding(num_queries, hidden_dim)
        self.query_feat4 = nn.Embedding(num_queries, hidden_dim)
        self.query_feat5 = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_embed1 = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.syncvis_last_layer_num = syncvis_last_layer_num

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["syncvis_last_layer_num"] = cfg.MODEL.syncvis.LAST_LAYER_NUM

        return ret

    def forward(self, x, mask_features, clip_mask_features, mask = None):
        # x is a list of multi-scale feature
        if not self.training:
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            query_embed1 = self.query_embed1.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            output1 = self.query_feat1.weight.unsqueeze(1).repeat(1, bs, 1)
            output2 = self.query_feat2.weight.unsqueeze(1).repeat(1, bs, 1)
            output3 = self.query_feat3.weight.unsqueeze(1).repeat(1, bs, 1)
            output4 = self.query_feat4.weight.unsqueeze(1).repeat(1, bs, 1)
            output5 = self.query_feat5.weight.unsqueeze(1).repeat(1, bs, 1)

            frame_queries = []
            predictions_class = []
            predictions_mask = []
            predictions_class1 = []
            predictions_mask1 = []


            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask, frame_query = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
            outputs_class1, outputs_mask1, attn_mask1, frame_query1= self.forward_prediction_heads(output1, clip_mask_features, attn_mask_target_size=size_list[0])
            # outputs_class5, outputs_mask5, attn_mask5, frame_query5 = self.forward_prediction_heads(output5, mask_features[4,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
            # outputs_class1, outputs_mask1, attn_mask1, frame_query1 = self.forward_prediction_heads(output1, mask_features[0,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
            # outputs_class2, outputs_mask2, attn_mask2, frame_query2 = self.forward_prediction_heads(output2, mask_features[1,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
            # outputs_class3, outputs_mask3, attn_mask3, frame_query3 = self.forward_prediction_heads(output3, mask_features[2,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
            # outputs_class4, outputs_mask4, attn_mask4, frame_query4 = self.forward_prediction_heads(output4, mask_features[3,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
            # outputs_class = (outputs_class1+outputs_class2+outputs_class3+outputs_class4+outputs_class5)/5
            # outputs_mask = torch.cat([outputs_mask1+outputs_mask2+outputs_mask3+outputs_mask4+outputs_mask5],dim=
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )
                
                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                # attn_mask1[torch.where(attn_mask1.sum(-1) == attn_mask1.shape[-1])] = False
                # # attention: cross-attention first
                # output1 = self.transformer_cross_attention_layers[i](
                #     output1, src[level_index],
                #     memory_mask=attn_mask1,
                #     memory_key_padding_mask=None,  # here we do not apply masking on padded region
                #     pos=pos[level_index], query_pos=query_embed1
                # )

                # output1 = self.transformer_self_attention_layers[i](
                #     output1, tgt_mask=None,
                #     tgt_key_padding_mask=None,
                #     query_pos=query_embed1
                # )
                
                # # FFN
                # output1 = self.transformer_ffn_layers[i](
                #     output1
                # )
                # outputs_class1, outputs_mask1, attn_mask1, frame_query1 = self.forward_prediction_heads(output1, clip_mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                # if i == 0:
                #     # pdb.set_trace()
                #     scores = F.softmax(outputs_class1, dim=2)[:, :,:-1]  #5,100,40
                #     # out_logits,logits_indices = torch.max(scores,dim=2)
                #     # pdb.set_trace()
                #     out_logits,logits_indices = scores.topk(k=2,dim=2) #5,100,k<40
                #     out_logits = torch.sum(out_logits,dim=2)
                #     select_logits, select_indices = out_logits.topk(k=10,dim=1,largest=True)  #5*10
                #     select_query_1 = output1[:,0,:][select_indices[0,:].unsqueeze(0).permute(1,0),:]
                #     select_query_2 = output1[:,1,:][select_indices[1,:].unsqueeze(0).permute(1,0),:]
                #     select_query_3 = output1[:,2,:][select_indices[2,:].unsqueeze(0).permute(1,0),:]
                #     select_query_4 = output1[:,3,:][select_indices[3,:].unsqueeze(0).permute(1,0),:]
                #     select_query_5 = output1[:,4,:][select_indices[4,:].unsqueeze(0).permute(1,0),:]
                #     ref_query = torch.cat([select_query_1,select_query_2,select_query_3,select_query_4,select_query_5],dim=1)
                #     # select_query_list.append(select_query)
                #     # ref_query = torch.cat([select_query_list[0],select_query_list[1],select_query_list[2],select_query_list[3]],dim=0)
                
                # output, weights = self.multi_attn(output,ref_query,ref_query)  
                outputs_class, outputs_mask, attn_mask, frame_query = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                
                frame_queries.append(frame_query)
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)

            assert len(predictions_class) == self.num_layers + 1

            out = {
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(
                    predictions_class if self.mask_classification else None, predictions_mask
                )
            }

            num_layer         = self.syncvis_last_layer_num if self.training else 1
            frame_queries     = torch.unsqueeze(frame_queries[-1],0).repeat(num_layer,1,1,1) # L x BT x fQ x 256

            # frame_queries     = torch.stack(frame_queries[-num_layer:]) # L x BT x fQ x 256

            return out, frame_queries, clip_mask_features
        
        else:
            results = []
            assert len(x) == self.num_feature_levels
            src_1, src_2, src_3, src_4, src_5 = [],[],[],[],[]
            src_6, src_7, src_8, src_9, src_10 = [],[],[],[],[]
            src = []
            pos, pos_1, pos_2, pos_3, pos_4= [],[],[],[],[]
            pos5, pos_6, pos_7, pos_8, pos_9= [],[],[],[],[]            
            size_list = []
            # idx_list=[[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
            # idx_list=[[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4],[0,1,2,3,4]]
            idx_list=[[0,1],[1,2],[2,3],[3,4]]
            pos_list=[[],[],[],[],[],[],[],[],[],[],[]]
            src_list=[[],[],[],[],[],[],[],[],[],[],[]]
            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                for ii in range(len(idx_list)):
                    pos_list[ii].append(self.pe_layer(x[i][idx_list[ii]], None).flatten(2))
                    src_list[ii].append(self.input_proj[i](x[i][idx_list[ii]]).flatten(2) + self.level_embed.weight[i][None, :, None])


                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)
                for ii in range(len(idx_list)):
                    pos_list[ii][-1] = pos_list[ii][-1].permute(2,0,1)
                    src_list[ii][-1] = src_list[ii][-1].permute(2,0,1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            query_embed1 = self.query_embed1.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            output1 = self.query_feat1.weight.unsqueeze(1).repeat(1, bs, 1)
            output2 = self.query_feat2.weight.unsqueeze(1).repeat(1, bs, 1)
            output3 = self.query_feat3.weight.unsqueeze(1).repeat(1, bs, 1)
            output4 = self.query_feat4.weight.unsqueeze(1).repeat(1, bs, 1)
            output5 = self.query_feat5.weight.unsqueeze(1).repeat(1, bs, 1)

            frame_queries = []
            predictions_class = []
            predictions_mask = []
            predictions_class1 = []
            predictions_mask1 = []

            for ii in range(len(idx_list)):
                predictions_class = []
                predictions_mask = []
                # pdb.set_trace()
                output_s = output[:,idx_list[ii]]
                output1_s = output1[:,idx_list[ii]]
                mask_features_s = mask_features[idx_list[ii],]
                clip_mask_features_s = clip_mask_features[idx_list[ii],]
                query_embed_s = query_embed[:,idx_list[ii]]
                src_tmp = src_list[ii]
                pos_tmp = pos_list[ii]                              
                # prediction heads on learnable query features
                outputs_class, outputs_mask, attn_mask, frame_query = self.forward_prediction_heads(output_s, mask_features_s, attn_mask_target_size=size_list[0])
                outputs_class1, outputs_mask1, attn_mask1, frame_query1= self.forward_prediction_heads(output1_s, clip_mask_features_s, attn_mask_target_size=size_list[0])
                # outputs_class5, outputs_mask5, attn_mask5, frame_query5 = self.forward_prediction_heads(output5, mask_features[4,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
                # outputs_class1, outputs_mask1, attn_mask1, frame_query1 = self.forward_prediction_heads(output1, mask_features[0,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
                # outputs_class2, outputs_mask2, attn_mask2, frame_query2 = self.forward_prediction_heads(output2, mask_features[1,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
                # outputs_class3, outputs_mask3, attn_mask3, frame_query3 = self.forward_prediction_heads(output3, mask_features[2,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
                # outputs_class4, outputs_mask4, attn_mask4, frame_query4 = self.forward_prediction_heads(output4, mask_features[3,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])

                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)

                
                for i in range(self.num_layers):
                    level_index = i % self.num_feature_levels
                    attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                    # attention: cross-attention first
                    output_s = self.transformer_cross_attention_layers[i](
                        output_s, src_tmp[level_index],
                        memory_mask=attn_mask,
                        memory_key_padding_mask=None,  # here we do not apply masking on padded region
                        pos=pos_tmp[level_index], query_pos=query_embed_s
                    )

                    output_s = self.transformer_self_attention_layers[i](
                        output_s, tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=query_embed_s
                    )
                    
                    # FFN
                    output_s = self.transformer_ffn_layers[i](
                        output_s
                    )

                    attn_mask1[torch.where(attn_mask1.sum(-1) == attn_mask1.shape[-1])] = False
                    # attention: cross-attention first
                    output1_s = self.transformer_cross_attention_layers[i](
                        output1_s, src[level_index],
                        memory_mask=attn_mask1,
                        memory_key_padding_mask=None,  # here we do not apply masking on padded region
                        pos=pos[level_index], query_pos=query_embed1
                    )

                    output1_s = self.transformer_self_attention_layers[i](
                        output1_s, tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=query_embed1
                    )
                    
                    # FFN
                    output1_s = self.transformer_ffn_layers[i](
                        output1_s
                    )
                    outputs_class1, outputs_mask1, attn_mask1, frame_query1 = self.forward_prediction_heads(output1_s, clip_mask_features_s, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                    
                    if i <= 2:
                        output1_tmp = self.FFN_1(output1_s)
                        # pdb.set_trace()
                        scores = F.softmax(outputs_class1, dim=2)[:, :,:-1]  #5,100,40
                        # out_logits,logits_indices = torch.max(scores,dim=2)
                        # pdb.set_trace()
                        out_logits,logits_indices = scores.topk(k=2,dim=2) #5,100,k<40
                        out_logits = torch.sum(out_logits,dim=2)
                        select_logits, select_indices = out_logits.topk(k=10,dim=1,largest=True)  #5*10
                        select_query_1 = output1_tmp[:,0,:][select_indices[0,:].unsqueeze(0).permute(1,0),:]
                        select_query_2 = output1_tmp[:,1,:][select_indices[1,:].unsqueeze(0).permute(1,0),:]
                        # select_query_3 = output1[:,2,:][select_indices[2,:].unsqueeze(0).permute(1,0),:]
                        # select_query_4 = output1[:,3,:][select_indices[3,:].unsqueeze(0).permute(1,0),:]
                        # select_query_5 = output1[:,4,:][select_indices[4,:].unsqueeze(0).permute(1,0),:]
                        ref_query = torch.cat([select_query_1,select_query_2],dim=1)
                        # select_query_list.append(select_query)
                        # ref_query = torch.cat([select_query_list[0],select_query_list[1],select_query_list[2],select_query_list[3]],dim=0)
                        output_s, weights = self.multi_attn(output_s,ref_query,ref_query)

                        outputs_class, outputs_mask, attn_mask, frame_query = self.forward_prediction_heads(output_s, mask_features_s, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                        output_tmp = self.FFN_2(output_s)
                        # pdb.set_trace()
                        scores1 = F.softmax(outputs_class, dim=2)[:, :,:-1]  #5,100,40
                        # out_logits,logits_indices = torch.max(scores,dim=2)
                        # pdb.set_trace()
                        out_logits1,logits_indices = scores1.topk(k=2,dim=2) #5,100,k<40
                        out_logits1 = torch.sum(out_logits1,dim=2)
                        select_logits, select_indices1 = out_logits1.topk(k=10,dim=1,largest=True)  #5*10
                        select_query_1 = output_tmp[:,0,:][select_indices1[0,:].unsqueeze(0).permute(1,0),:]
                        select_query_2 = output_tmp[:,1,:][select_indices1[1,:].unsqueeze(0).permute(1,0),:]
                        # select_query_3 = output1[:,2,:][select_indices[2,:].unsqueeze(0).permute(1,0),:]
                        # select_query_4 = output1[:,3,:][select_indices[3,:].unsqueeze(0).permute(1,0),:]
                        # select_query_5 = output1[:,4,:][select_indices[4,:].unsqueeze(0).permute(1,0),:]
                        ref_query = torch.cat([select_query_1,select_query_2],dim=1)
                        # select_query_list.append(select_query)
                        # ref_query = torch.cat([select_query_list[0],select_query_list[1],select_query_list[2],select_query_list[3]],dim=0)
                        output1_s, weights = self.multi_attn(output1_s,ref_query,ref_query)                        


                    outputs_class, outputs_mask, attn_mask, frame_query = self.forward_prediction_heads(output_s, mask_features_s, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                    
                    frame_queries.append(frame_query)
                    predictions_class.append(outputs_class)
                    predictions_mask.append(outputs_mask)

                assert len(predictions_class) == self.num_layers + 1
                
                results.append({
                    'pred_logits': predictions_class[-1],
                    'pred_masks': predictions_mask[-1],
                    'aux_outputs': self._set_aux_loss(
                        predictions_class if self.mask_classification else None, predictions_mask
                    )
                })

            predictions_class = []
            predictions_mask = []
            outputs_class, outputs_mask, attn_mask, frame_query = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
            outputs_class1, outputs_mask1, attn_mask1, frame_query1= self.forward_prediction_heads(output1, clip_mask_features, attn_mask_target_size=size_list[0])
            # outputs_class5, outputs_mask5, attn_mask5, frame_query5 = self.forward_prediction_heads(output5, mask_features[4,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
            # outputs_class1, outputs_mask1, attn_mask1, frame_query1 = self.forward_prediction_heads(output1, mask_features[0,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
            # outputs_class2, outputs_mask2, attn_mask2, frame_query2 = self.forward_prediction_heads(output2, mask_features[1,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
            # outputs_class3, outputs_mask3, attn_mask3, frame_query3 = self.forward_prediction_heads(output3, mask_features[2,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
            # outputs_class4, outputs_mask4, attn_mask4, frame_query4 = self.forward_prediction_heads(output4, mask_features[3,:,:,:].unsqueeze(0), attn_mask_target_size=size_list[0])
            predictions_class = []
            predictions_mask = []
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            
            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )
                
                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                # attn_mask1[torch.where(attn_mask1.sum(-1) == attn_mask1.shape[-1])] = False
                # # attention: cross-attention first
                # output1 = self.transformer_cross_attention_layers[i](
                #     output1, src[level_index],
                #     memory_mask=attn_mask1,
                #     memory_key_padding_mask=None,  # here we do not apply masking on padded region
                #     pos=pos[level_index], query_pos=query_embed1
                # )

                # output1 = self.transformer_self_attention_layers[i](
                #     output1, tgt_mask=None,
                #     tgt_key_padding_mask=None,
                #     query_pos=query_embed1
                # )
                
                # # FFN
                # output1 = self.transformer_ffn_layers[i](
                #     output1
                # )
                # outputs_class1, outputs_mask1, attn_mask1, frame_query1 = self.forward_prediction_heads(output1, clip_mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                # if i == 8:
                #     # pdb.set_trace()
                #     scores = F.softmax(outputs_class1, dim=2)[:, :,:-1]  #5,100,40
                #     # out_logits,logits_indices = torch.max(scores,dim=2)
                #     # pdb.set_trace()
                #     out_logits,logits_indices = scores.topk(k=2,dim=2) #5,100,k<40
                #     out_logits = torch.sum(out_logits,dim=2)
                #     select_logits, select_indices = out_logits.topk(k=10,dim=1,largest=True)  #5*10
                #     select_query_1 = output1[:,0,:][select_indices[0,:].unsqueeze(0).permute(1,0),:]
                #     select_query_2 = output1[:,1,:][select_indices[1,:].unsqueeze(0).permute(1,0),:]
                #     select_query_3 = output1[:,2,:][select_indices[2,:].unsqueeze(0).permute(1,0),:]
                #     select_query_4 = output1[:,3,:][select_indices[3,:].unsqueeze(0).permute(1,0),:]
                #     select_query_5 = output1[:,4,:][select_indices[4,:].unsqueeze(0).permute(1,0),:]
                #     ref_query = torch.cat([select_query_1,select_query_2,select_query_3,select_query_4,select_query_5],dim=1)
                #     # select_query_list.append(select_query)
                #     # ref_query = torch.cat([select_query_list[0],select_query_list[1],select_query_list[2],select_query_list[3]],dim=0)
                
                #     output, weights = self.multi_attn(output,ref_query,ref_query)

                outputs_class, outputs_mask, attn_mask, frame_query = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                frame_queries.append(frame_query)
            
            results.append({
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(
                    predictions_class if self.mask_classification else None, predictions_mask
                )
            })
            

            # out = {
            #     'pred_logits': predictions_class[-1],
            #     'pred_masks': predictions_mask[-1],
            #     'aux_outputs': self._set_aux_loss(
            #         predictions_class if self.mask_classification else None, predictions_mask
            #     )
            # }

            num_layer         = self.syncvis_last_layer_num if self.training else 1
            frame_queries     = torch.unsqueeze(frame_queries[-1],0).repeat(num_layer,1,1,1) # L x BT x fQ x 256

            # frame_queries     = torch.stack(frame_queries[-num_layer:]) # L x BT x fQ x 256

            return results, frame_queries, clip_mask_features


    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask, decoder_output

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
