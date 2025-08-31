import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch_geometric.nn as Gnn
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv
from typing import List, Dict, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence
from .AttentionModel import CustomMultiheadAttention

torch.backends.cudnn.benchmark = True
torch.cuda.amp.autocast(enabled=True)

class UniBIP(nn.Module):
    def __init__(self,
                 in_feat: int,
                 feature_in_feat: int,
                 hidden_feat: int,
                 num_classes: int = 7,
                 GCNELayer_num_layers: int = 2,
                 AuxiliaryGCNELayer_num_layers: int = 0,
                 Auxiliary_in_feat: int = 0,
                 grid_size: int = 4,
                 dr:float = 0.2,
                 EdgeFeatureAwareness: bool = True,
                 NodeFeatureAwareness: bool = True,
                 use_GCNELayer: bool = True,
                 use_AuxiliaryGCNELayer: bool = True,
                 texts=None,
                 device=None):

        super(UniBIP, self).__init__()
        if num_classes>1:
            classification = True
        else:
            classification = False
        if not use_GCNELayer:
            GCNELayer_num_layers = 0
        if not use_AuxiliaryGCNELayer:
            AuxiliaryGCNELayer_num_layers = 0
        self.GCNELayer_num_layers = GCNELayer_num_layers
        self.AuxiliaryGCNELayer_num_layers = AuxiliaryGCNELayer_num_layers
        self.classification = classification

        if GCNELayer_num_layers !=0:
                self.embedding_layer_GCNE = GCNELayer(in_channels=in_feat, out_channels=hidden_feat, num_layers=GCNELayer_num_layers)
        if AuxiliaryGCNELayer_num_layers !=0:
            self.Auxiliary_embedding_layer_GCNE = GCNELayer(in_channels=Auxiliary_in_feat, out_channels=hidden_feat, num_layers=AuxiliaryGCNELayer_num_layers)

        other_number = 0

        self.use_GEA = EdgeFeatureAwareness
        if self.use_GEA:
            other_number += 1
            self.embedding_layer_GEAE = GEAELayer(out_channels=hidden_feat, device=device)

        self.use_FNN = NodeFeatureAwareness
        if self.use_FNN:
            other_number += 1
            self.FNN = nn.Sequential(nn.LayerNorm(feature_in_feat), nn.Linear(feature_in_feat, hidden_feat), nn.LayerNorm(hidden_feat), nn.Linear(hidden_feat, hidden_feat))

        if texts is not None:
            self.attention_model = SequencePairEncoder(texts=texts, hidden_dim=hidden_feat, device=device)
            other_number += 1
        else:
            self.attention_model = None

        self.similarity = Similarity(hidden_feat * (other_number + GCNELayer_num_layers + AuxiliaryGCNELayer_num_layers),num_classes, activation_fn=torch.sigmoid)
        self.device = device
        self.to(device)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def save_model(self,path):
        torch.save(self.state_dict(), path)

    def forward(self, data):

        X_data, edge_index, edge_attr = data.x, data.edge_index, data.edge_feature
        x_feature = data.feature
        out_list = []

        if self.GCNELayer_num_layers != 0:
            X_init = self.embedding_layer_GCNE(X_data,edge_index)
            out_list.append(X_init)

        if self.AuxiliaryGCNELayer_num_layers != 0 and not data.auxiliary_data is None:
            auxiliary_data = data.auxiliary_data
            auxiliar_edge_index = torch.cat([edge_index, auxiliary_data.edge_index],dim=1)
            auxiliar_x = auxiliary_data.x
            X_Auxiliary = self.Auxiliary_embedding_layer_GCNE(auxiliar_x, auxiliar_edge_index)
            if hasattr(auxiliary_data, 'index_x_init') and not auxiliary_data.index_x_init is None:
                X_init = X_Auxiliary[auxiliary_data.index_x_init]
            else:
                X_init = X_Auxiliary[:X_data.shape[0]]
            out_list.append(X_init)
        if self.use_GEA:
            X_init = self.embedding_layer_GEAE(X_data,edge_index, edge_attr)
            out_list.append(X_init)
        if self.use_FNN:
            x_feature = self.FNN(x_feature.to(dtype=X_data.dtype))
            out_list.append(x_feature)

        return torch.cat(out_list, dim=1)

    def similarityfunc(self,X, edge_index, edge_attr=None):
        x1 = X[edge_index[0]]
        x2 = X[edge_index[1]]
        if self.attention_model is not None:
            edge_attr, *_ = self.attention_model(edge_index[0], edge_index[1])
        return self.similarity(x1,x2,edge_attr=edge_attr)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, random_seed=None):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if random_seed is not None:
            torch.manual_seed(random_seed)

    def forward(self, y_true, y_pred):
        y_true = y_true.float()
        device = y_true.device

        pos_mask = (y_true != 0)
        neg_mask = (y_true == 0)

        num_pos = pos_mask.sum(dim=0)
        num_neg = neg_mask.sum(dim=0)

        k_neg = torch.min(num_pos, num_neg)
        total_classes = y_true.size(1)

        if total_classes == 0 or num_pos.sum() == 0:
            print('Error...')
            return torch.tensor(0.0, device=device)

        rand_mat = torch.rand(y_true.shape, device=device)
        rand_mat = torch.where(pos_mask, float('inf'), rand_mat)

        max_k = k_neg.max().int().item()
        if max_k > 0:
            topk_vals, topk_idxs = torch.topk(
                rand_mat, k=max_k, dim=0, largest=False, sorted=False
            )

            valid_mask = (
                    torch.arange(max_k, device=device)[:, None]
                    < k_neg[None, :]
            )

            selected_neg_mask = torch.zeros_like(y_true, dtype=torch.bool)
            valid_rows = topk_idxs[valid_mask]
            valid_cols = torch.where(valid_mask)[1]
            selected_neg_mask[valid_rows, valid_cols] = True
        else:
            selected_neg_mask = torch.zeros_like(y_true, dtype=torch.bool)

        selected_mask = pos_mask | selected_neg_mask

        square_pred = torch.square(y_pred)
        margin_square = torch.square(
            torch.clamp(self.margin - y_pred, min=0.0)
        )

        element_loss = (1 - y_true) * square_pred + y_true * margin_square
        masked_loss = element_loss * selected_mask
        valid_counts = (num_pos + k_neg).clamp(min=1e-6)
        class_loss = masked_loss.sum(dim=0) / valid_counts
        loss = class_loss.sum() / total_classes

        return loss

class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        input: x [batch_size, hidden_dim, seq_len]
        output: [batch_size, hidden_dim]
        """
        attn_weights = self.attn(x)
        output = x * attn_weights
        return output

class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        attn_weights = F.softmax(self.attn(x), dim=1)
        return torch.sum(x * attn_weights, dim=1)


class Similarity(nn.Module):
    def __init__(self, input_dim, num_classes, activation_fn=None, edge_attr_fnn=False):
        super().__init__()
        self.FNN = nn.Linear(input_dim, num_classes)
        self.num_classes = num_classes
        self.attention = AttentionPool(num_classes)
        self.activation_fn = activation_fn
        self.edge_attr_fnn = edge_attr_fnn
        self.lp = torch.nn.Parameter(torch.tensor(1.0))
        self.edge_attr_fnn = Gnn.Linear(-1, 128)

    def forward(self, x1, x2, edge_attr=None, edge_attr_fnn=False):
        x = torch.pow(torch.abs(x1 - x2), torch.clamp(self.lp, min=1, max=4))
        if edge_attr is not None:
            if self.edge_attr_fnn == True:
                edge_attr = self.edge_attr_fnn(edge_attr)
            x = torch.cat([x, edge_attr], dim=1)
        x = self.FNN(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class GCNELayer(nn.Module):
    """
    Parameters:
    - in_channels: Number of input channels (features) for each node.
    - out_channels: Number of output channels (features) for each node in each GCNConv layer.
    - num_layers: Number of LayerGNN layers in the model.
    - num_channels: Number of parallel GCNConv layers within each LayerGNN layer.
    - dr: Dropout rate applied in each LayerGNN layer.
    - device: The device (CPU/GPU) on which the model is run.

    This class:
    - Initializes a series of GCNConv layers for initial processing of node features.
    - Stacks multiple LayerGNN layers for deeper feature extraction.
    - Applies residual connections to combine features from different layers.
    """

    def __init__(self, in_channels, out_channels=128, num_layers=8,device=None):
        super(GCNELayer, self).__init__()
        self.device = device
        self.initial_convs = nn.ModuleList([GCNConv(in_channels, out_channels) for _ in range(num_layers)])
        # Move the model to the specified device
        self.to(self.device)

    def forward(self,x,edge_index):
        """
        Parameters:
        - data: The input data containing node features (x) and graph structure (edge_index).

        Returns:
        - output: The combined node features after passing through all layers of the model.

        The function:
        - Applies LayerGT layers iteratively with residual connections.
        - Combines the outputs from all layers to form the final node representations.
        """
        initial_z = []
        # Apply initial GCNConv layers to process node features
        for conv in self.initial_convs:
            initial_z.append(F.sigmoid(conv(x, edge_index)))
        # Concatenate the outputs from all layers to form the final node representations
        return torch.cat(initial_z, dim=1) #out_channels * num_layers


class GEAELayer(nn.Module):
    """
    Parameters:
    - in_channels: Number of input channels (features) for each node.
    - out_channels: Number of output channels (features) for each node in each GCNConv layer.
    - num_layers: Number of LayerGNN layers in the model.
    - num_channels: Number of parallel GCNConv layers within each LayerGNN layer.
    - dr: Dropout rate applied in each LayerGNN layer.
    - device: The device (CPU/GPU) on which the model is run.

    This class:
    - Initializes a series of GCNConv layers for initial processing of node features.
    - Stacks multiple LayerGNN layers for deeper feature extraction.
    - Applies residual connections to combine features from different layers.
    """

    def __init__(self,out_channels=128, device=None):
        super(GEAELayer, self).__init__()
        self.device = device
        self.x_proj = Gnn.Linear(-1, out_channels)
        self.edge_proj = Gnn.Linear(-1, out_channels)
        self.msg_combine = Gnn.Linear(-1, out_channels)
        self.beta = nn.Parameter(torch.tensor(0.1,dtype=torch.float))
        self.to(self.device)

    def forward(self,x, edge_index, edge_attr):
        """
        Parameters:
        - data: The input data containing node features (x) and graph structure (edge_index).

        Returns:
        - output: The combined node features after passing through all layers of the model.

        The function:
        - Applies LayerGT layers iteratively with residual connections.
        - Combines the outputs from all layers to form the final node representations.
        """
        #x, edge_index = data.x, data.edge_index  # Extract node features and edge index
        return self._mult_attr_msg(x, edge_index, edge_attr)
        # Concatenate the outputs from all layers to form the final node representations

    def _mult_attr_msg(self,x, edge_index, edge_attr):
        x = self.x_proj(x)
        src, dst = edge_index.to(torch.int64)
        edge_features = self.edge_proj(edge_attr)
        src_features = x[src]
        msg = torch.cat([src_features, edge_features], dim=1)
        msg = self.msg_combine(msg)
        msg = F.leaky_relu(msg)
        aggregated_msg = scatter_add(msg, dst.to(torch.int64) , dim=0, dim_size=x.shape[0])

        return F.sigmoid(aggregated_msg) * F.relu(self.beta)

class SequencePairEncoder(nn.Module):
    def __init__(self, texts: List[str], embed_dim: int = 256, max_len=512,
                 num_heads: int = 2, hidden_dim: int = 128, device='cpu'):

        super().__init__()
        self.device = device

        texts = np.array(texts).reshape(-1, ).tolist()
        self.char_to_idx = self._build_vocab(texts)
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

        self.max_single_len = min(max(len(text) for text in texts) if texts else 1, max_len)
        self.max_len = self.max_single_len * 2 + 3

        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(self.max_len, embed_dim)
        self.seg_embedding = nn.Embedding(3, embed_dim)

        self.attention = CustomMultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.projection = nn.Linear(embed_dim, hidden_dim)

        self.cached_sequences = self._precache_sequences(texts)

        self.to(device)

    def _build_vocab(self, texts: List[str]) -> Dict[str, int]:
        chars = sorted(set(''.join(texts).lower()))
        char_to_idx = {char: i + 3 for i, char in enumerate(chars)}
        char_to_idx['[PAD]'] = 0
        char_to_idx['[CLS]'] = 1
        char_to_idx['[SEP]'] = 2

        return char_to_idx

    def _text_to_indices(self, text: str) -> torch.Tensor:
        indices = []
        for i, c in enumerate(text):
            if i >= self.max_single_len:
                break
            if c.isalpha():
                indices.append(self.char_to_idx.get(c.lower(), 0))
        return torch.tensor(indices)

    def _precache_sequences(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        cached = {
            'seq1': [],
            'seg1': [],
            'seq2': [],
            'seg2': [],
            'mask': []
        }

        for text in texts:
            indices = self._text_to_indices(text)
            seq1 = torch.cat([torch.tensor([1]), indices, torch.tensor([2])])
            seg1 = torch.cat([
                torch.tensor([1]),
                torch.ones(len(indices), dtype=torch.long),
                torch.tensor([1])
            ])

            seq2 = torch.cat([indices, torch.tensor([2])])
            seg2 = torch.cat([
                torch.full((len(indices),), 2, dtype=torch.long),
                           torch.tensor([2])
            ])

            def pad_to_max(tensor, max_len, pad_value=0):
                if len(tensor) < max_len:
                    return F.pad(tensor, (0, max_len - len(tensor)), value=pad_value)
                return tensor[:max_len]

            seq1_max_len = self.max_single_len + 2
            seq2_max_len = self.max_single_len + 1

            padded_seq1 = pad_to_max(seq1, seq1_max_len)
            padded_seg1 = pad_to_max(seg1, seq1_max_len)
            padded_seq2 = pad_to_max(seq2, seq2_max_len)
            padded_seg2 = pad_to_max(seg2, seq2_max_len)

            mask = torch.cat([
                torch.ones(len(seq1), dtype=torch.bool),
                torch.zeros(seq1_max_len - len(seq1), dtype=torch.bool),
                torch.ones(len(seq2), dtype=torch.bool),
                torch.zeros(seq2_max_len - len(seq2), dtype=torch.bool)
            ])

            cached['seq1'].append(padded_seq1)
            cached['seg1'].append(padded_seg1)
            cached['seq2'].append(padded_seq2)
            cached['seg2'].append(padded_seg2)
            cached['mask'].append(mask)

        for key in cached:
            cached[key] = torch.stack(cached[key]).to(self.device)

        return cached

    def forward(self, x1_idx: torch.Tensor, x2_idx: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:

        batch_size = x1_idx.size(0)

        seq1 = self.cached_sequences['seq1'][x1_idx]
        seg1 = self.cached_sequences['seg1'][x1_idx]

        if x2_idx is not None:
            seq2 = self.cached_sequences['seq2'][x2_idx]
            seg2 = self.cached_sequences['seg2'][x2_idx]
        else:
            seq2 = self.cached_sequences['seq2'][x1_idx]
            seg2 = self.cached_sequences['seg2'][x1_idx]

        padded_seq = torch.cat([seq1, seq2], dim=1)
        padded_seg = torch.cat([seg1, seg2], dim=1)

        pos_ids = torch.arange(padded_seq.size(1)).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        token_embed = self.token_embedding(padded_seq)
        pos_embed = self.pos_embedding(pos_ids)
        seg_embed = self.seg_embedding(padded_seg)
        embed = token_embed + pos_embed + seg_embed

        attn_output, _, attn_weights= self.attention(embed, embed, embed, need_weights=True, average_attn_weights=False)
        attn_output = self.layer_norm1(embed + attn_output)
        ffn_output = self.ffn(attn_output)
        ffn_output = self.layer_norm2(attn_output + ffn_output)

        cls_embedding = ffn_output[:, 0, :]
        embeddings = F.softmax(self.projection(cls_embedding))

        return embeddings, attn_weights

    def get_attention_weights_from_idx(self, x1_idx: int, x2_idx: Optional[int] = None) -> pd.DataFrame:
        x1_idx_tensor = torch.tensor([x1_idx]).to(self.device)
        x2_idx_tensor = torch.tensor([x2_idx]).to(self.device) if x2_idx is not None else None

        _, attn_weights = self.forward(x1_idx_tensor, x2_idx_tensor)
        attn_weights = attn_weights[0].detach()

        seq1 = self.cached_sequences['seq1'][x1_idx]
        valid_seq1 = seq1[seq1 != 0]
        x1_text = [self.idx_to_char.get(idx.item(), '[UNK]') for idx in valid_seq1
                   if idx.item() not in [0, 1, 2]]

        if x2_idx is not None:
            seq2 = self.cached_sequences['seq2'][x2_idx]
            valid_seq2 = seq2[seq2 != 0]
            x2_text = [self.idx_to_char.get(idx.item(), '[UNK]') for idx in valid_seq2
                       if idx.item() not in [0, 1, 2]]
        else:
            seq2 = self.cached_sequences['seq2'][x1_idx]
            valid_seq2 = seq2[seq2 != 0]
            x2_text = [self.idx_to_char.get(idx.item(), '[UNK]') for idx in valid_seq2
                       if idx.item() not in [0, 1, 2]]

        avg_weights = attn_weights.mean(dim=0).cpu().numpy()
        seq1_max_len = self.max_single_len + 2

        x1_start = 1
        x1_end = x1_start + len(x1_text)
        x2_start = seq1_max_len
        x2_end = x2_start + len(x2_text)
        weights_np = avg_weights[x1_start:x1_end, x2_start:x2_end]

        df = pd.DataFrame(weights_np, index=x1_text, columns=x2_text)
        return df

    def get_attention_weights_from_text(self, x1_str: str, x2_str: Optional[str] = None) -> pd.DataFrame:

        x1_str = x1_str[:self.max_single_len]
        x1_indices = self._text_to_indices(x1_str)
        x1_text = [self.idx_to_char.get(idx.item(), '[UNK]') for idx in x1_indices]

        if x2_str is not None:
            x2_str = x2_str[:self.max_single_len]
            x2_indices = self._text_to_indices(x2_str)
            x2_text = [self.idx_to_char.get(idx.item(), '[UNK]') for idx in x2_indices]
        else:
            x2_indices = None
            x2_text = []

        seq1 = torch.cat([torch.tensor([1]), x1_indices, torch.tensor([2])])
        seg1 = torch.cat([
            torch.tensor([1]),
            torch.ones(len(x1_indices), dtype=torch.long),
            torch.tensor([1])
        ])

        if x2_str is not None:
            seq2 = torch.cat([x2_indices, torch.tensor([2])])
            seg2 = torch.cat([
                torch.full((len(x2_indices),), 2, dtype=torch.long),
                torch.tensor([2])
            ])
        else:
            seq2 = torch.cat([x1_indices, torch.tensor([2])])
            seg2 = torch.cat([
                torch.full((len(x1_indices),), 2, dtype=torch.long),
                torch.tensor([2])
            ])
            x2_text = x1_text

        def pad_to_max(tensor, max_len, pad_value=0):
            if len(tensor) < max_len:
                return F.pad(tensor, (0, max_len - len(tensor)), value=pad_value)
            return tensor[:max_len]

        seq1_max_len = self.max_single_len + 2
        seq2_max_len = self.max_single_len + 1

        padded_seq1 = pad_to_max(seq1, seq1_max_len)
        padded_seg1 = pad_to_max(seg1, seq1_max_len)
        padded_seq2 = pad_to_max(seq2, seq2_max_len)
        padded_seg2 = pad_to_max(seg2, seq2_max_len)

        padded_seq = torch.cat([padded_seq1, padded_seq2]).unsqueeze(0)
        padded_seg = torch.cat([padded_seg1, padded_seg2]).unsqueeze(0)

        seq_len = padded_seq.size(1)
        pos_ids = torch.arange(seq_len).unsqueeze(0)

        padded_seq = padded_seq.to(self.device)
        padded_seg = padded_seg.to(self.device)
        pos_ids = pos_ids.to(self.device)

        token_embed = self.token_embedding(padded_seq)
        pos_embed = self.pos_embedding(pos_ids)
        seg_embed = self.seg_embedding(padded_seg)

        embed = token_embed + pos_embed + seg_embed

        with torch.no_grad():
            attn_output, _, attn_weights= self.attention(
                embed, embed, embed,
                need_weights=True,
                average_attn_weights=False
            )
        attn_weights = attn_weights.detach()
        avg_weights = attn_weights.mean(dim=0).cpu().numpy()

        x1_start = 1
        x1_end = x1_start + len(x1_text)
        x2_start = seq1_max_len
        x2_end = x2_start + len(x2_text)
        weights_np = avg_weights[x1_start:x1_end, x2_start:x2_end]

        df = pd.DataFrame(weights_np, index=x1_text, columns=x2_text)
        return df

