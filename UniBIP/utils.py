import pandas as pd
import json
import os
import torch
import numpy as np
import datetime
import random
import copy
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    accuracy_score, recall_score, precision_score,
    f1_score
)
from scipy import stats
from math import sqrt
from collections import OrderedDict
import joblib
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_networkx, negative_sampling, add_self_loops
import networkx as nx
from torch_geometric.data import Data
import copy
from sklearn.model_selection import ParameterGrid



def print_execution_time(start_time):
    """
    Print the execution time of the task.

    Parameters:
    start_time (datetime): The start time of the task.

    This function calculates the elapsed time since the start and prints it in a human-readable format (hours, minutes, seconds).
    """
    import datetime
    execution_time = datetime.datetime.now() - start_time
    hours, remainder = divmod(execution_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total execution time: {hours} hours, {minutes} minutes, {seconds} seconds")


def get_best_f1(y_true_micro,y_pred_micro):
    thresholds = np.arange(100) / 100.0
    binary_preds = (y_pred_micro[:, None] >= thresholds).astype(int)
    TP = np.sum((y_true_micro[:, None] == 1) & (binary_preds == 1), axis=0)
    FP = np.sum((y_true_micro[:, None] == 0) & (binary_preds == 1), axis=0)
    FN = np.sum((y_true_micro[:, None] == 1) & (binary_preds == 0), axis=0)
    f1_scores = 2 * TP / (2 * TP + FP + FN + 1e-9)

    micro_best_index = np.argmax(f1_scores)
    micro_best_threshold = thresholds[micro_best_index]
    micro_best_f1 = f1_scores[micro_best_index]
    return micro_best_f1, micro_best_threshold



def get_best_f1(y_true_micro, y_pred_micro):
    n = len(y_true_micro)
    total_positive = np.sum(y_true_micro)
    thresholds = np.arange(1000) / 1000.0

    sorted_pred_asc = np.sort(y_pred_micro)

    desc_order = np.argsort(y_pred_micro)[::-1]
    sorted_true_desc = y_true_micro[desc_order]

    cum_tp = np.concatenate([[0], np.cumsum(sorted_true_desc)])
    cum_fp = np.concatenate([[0], np.cumsum(1 - sorted_true_desc)])

    k = n - np.searchsorted(sorted_pred_asc, thresholds, side='left')

    TP_t = cum_tp[k]
    FP_t = cum_fp[k]
    FN_t = total_positive - TP_t

    f1_scores = (2 * TP_t) / (2 * TP_t + FP_t + FN_t + 1e-9)

    micro_best_index = np.argmax(f1_scores)
    micro_best_threshold = thresholds[micro_best_index]
    micro_best_f1 = f1_scores[micro_best_index]

    return micro_best_f1, micro_best_threshold


def get_best_f1(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    return best_f1, best_threshold

def get_metrics(true_score, predict_score, classification_mode=True):
    all_metrics = []
    all_names = []
    class_auc = {}
    best_threshold = None
    if classification_mode:
        for cls in range(true_score.shape[1]):
            y_true_cls = true_score[:, cls]
            y_pred_cls = predict_score[:, cls]

            if len(np.unique(y_true_cls)) < 2:
                print(f"Warning: Class {cls} has only one unique value. Skipping...")
                continue

            auc_score = roc_auc_score(y_true_cls, y_pred_cls)
            precision, recall, _ = precision_recall_curve(y_true_cls, y_pred_cls)
            aupr_score = auc(recall, precision)

            best_f1, best_threshold = get_best_f1(y_true_cls, y_pred_cls)

            binary_pred = (y_pred_cls > best_threshold).astype(int)
            acc = accuracy_score(y_true_cls, binary_pred)
            rec = recall_score(y_true_cls, binary_pred)
            spec = recall_score(1 - y_true_cls, 1 - binary_pred)
            prec = precision_score(y_true_cls, binary_pred)

            cls_metrics = [
                y_true_cls, y_pred_cls, auc_score, aupr_score,
                best_f1, acc, rec, spec, prec, best_threshold
            ]
            cls_names = [
                f'class_{cls}_y_true', f'class_{cls}_y_score',
                f'class_{cls}_auc', f'class_{cls}_prc',
                f'class_{cls}_f1', f'class_{cls}_acc',
                f'class_{cls}_recall', f'class_{cls}_specificity',
                f'class_{cls}_precision', f'class_{cls}_best_threshold'
            ]
            class_auc[cls] = auc_score
            all_metrics.extend(cls_metrics)
            all_names.extend(cls_names)

            print(f"Class {cls}: auc:{auc_score:.4f}, aupr:{aupr_score:.4f}, "
                  f"f1:{best_f1:.4f}, acc:{acc:.4f}, recall:{rec:.4f}, "
                  f"specificity:{spec:.4f}, precision:{prec:.4f}, "
                  f"best_threshold:{best_threshold:.2f}")

        y_true_micro = true_score.ravel()
        y_pred_micro = predict_score.ravel()

        micro_auc = roc_auc_score(y_true_micro, y_pred_micro)
        precision_micro, recall_micro, _ = precision_recall_curve(y_true_micro, y_pred_micro)
        micro_aupr = auc(recall_micro, precision_micro)

        micro_best_f1, micro_best_threshold = get_best_f1(y_true_micro, y_pred_micro)
        binary_pred_micro = (y_pred_micro > micro_best_threshold).astype(int)
        micro_acc = accuracy_score(y_true_micro, binary_pred_micro)
        micro_f1 = micro_best_f1
        micro_recall = recall_score(y_true_micro, binary_pred_micro)
        micro_specificity = recall_score(1 - y_true_micro, 1 - binary_pred_micro)
        micro_precision = precision_score(y_true_micro, binary_pred_micro)

        if not np.any(np.sum(true_score, axis=1) > 1) and true_score.shape[-1] > 2:
            y_true_label = true_score.argmax(axis=1)
            y_pred_label = predict_score.argmax(axis=1)
            micro_acc = accuracy_score(y_true_label, y_pred_label)

        def get_macro_metric(suffix):
            return np.mean([m for n, m in zip(all_names, all_metrics)
                            if n.startswith('class_') and n.endswith(f'_{suffix}')])

        all_metrics.extend([
            y_true_micro, y_pred_micro,
            micro_auc, micro_aupr, micro_f1, micro_acc, micro_recall, micro_specificity, micro_precision
        ])
        all_names.extend([
            'true_score', 'predict_score',
            'auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision'
        ])

        print("\nMicro Averages:")
        print(f"AUC:{micro_auc:.4f}, AUPR:{micro_aupr:.4f}, F1:{micro_f1:.4f},"
              f"Acc:{micro_acc:.4f}, Recall:{micro_recall:.4f}, "
              f"Specificity:{micro_specificity:.4f}, Precision:{micro_precision:.4f}")

    else:
        for cls in range(true_score.shape[1]):
            y_true_cls = true_score[:, cls]
            y_pred_cls = predict_score[:, cls]

            rmse_score = sqrt(((y_true_cls - y_pred_cls) ** 2).mean())
            mse_score = ((y_true_cls - y_pred_cls) ** 2).mean()
            pearson_score = np.corrcoef(y_true_cls, y_pred_cls)[0, 1]
            spearman_score = stats.spearmanr(y_true_cls, y_pred_cls)[0]

            threshold = np.median(y_true_cls)
            y_true_binary = (y_true_cls > threshold).astype(int)

            if len(np.unique(y_true_binary)) < 2:
                print(f"Warning: Class {cls} has only one unique value after binarization. Skipping AUC/PRC...")
                auc_score = float('nan')
                aupr_score = float('nan')
            else:
                auc_score = roc_auc_score(y_true_binary, y_pred_cls)
                precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_cls)
                aupr_score = auc(recall, precision) if len(recall) > 1 and len(precision) > 1 else float('nan')

            cls_metrics = [
                y_true_cls, y_pred_cls, auc_score, aupr_score,
                rmse_score, mse_score, pearson_score, spearman_score
            ]
            cls_names = [
                f'class_{cls}_y_true', f'class_{cls}_y_score',
                f'class_{cls}_auc', f'class_{cls}_prc',
                f'class_{cls}_rmse', f'class_{cls}_mse',
                f'class_{cls}_pearson', f'class_{cls}_spearman'
            ]
            all_metrics.extend(cls_metrics)
            all_names.extend(cls_names)

            print(f"Class {cls}: AUC:{auc_score:.4f}, PRC:{aupr_score:.4f}, "
                  f"RMSE:{rmse_score:.4f}, MSE:{mse_score:.4f}, "
                  f"Pearson:{pearson_score:.4f}, Spearman:{spearman_score:.4f}")

        y_true_micro = true_score.ravel()
        y_pred_micro = predict_score.ravel()

        micro_rmse = sqrt(((y_true_micro - y_pred_micro) ** 2).mean())
        micro_mse = ((y_true_micro - y_pred_micro) ** 2).mean()
        micro_pearson = np.corrcoef(y_true_micro, y_pred_micro)[0, 1]
        micro_spearman = stats.spearmanr(y_true_micro, y_pred_micro)[0]

        micro_threshold = np.median(y_true_micro)
        y_true_micro_binary = (y_true_micro > micro_threshold).astype(int)

        if len(np.unique(y_true_micro_binary)) < 2:
            print("Warning: Micro labels have only one unique value after binarization. Skipping AUC/PRC...")
            micro_auc = float('nan')
            micro_aupr = float('nan')
        else:
            micro_auc = roc_auc_score(y_true_micro_binary, y_pred_micro)
            precision_micro, recall_micro, _ = precision_recall_curve(y_true_micro_binary, y_pred_micro)
            micro_aupr = auc(recall_micro, precision_micro) if len(recall_micro) > 1 and len(
                precision_micro) > 1 else float('nan')

        all_metrics.extend([
            y_true_micro, y_pred_micro,
            micro_auc, micro_aupr, micro_rmse, micro_mse, micro_pearson, micro_spearman
        ])
        all_names.extend([
            'true_score', 'predict_score',
            'auc', 'prc', 'rmse', 'mse', 'pearson', 'spearman'
        ])

        print("\nMicro Averages:")
        print(f"AUC:{micro_auc:.4f}, PRC:{micro_aupr:.4f}, "
              f"RMSE:{micro_rmse:.4f}, MSE:{micro_mse:.4f}, "
              f"Pearson:{micro_pearson:.4f}, Spearman:{micro_spearman:.4f}")

    return all_metrics, all_names,best_threshold, class_auc if classification_mode else 0


def calculate_accuracy(y_true, y_pred):
    valid_indices = np.sum(y_true, axis=1) > 0
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    y_true_cls = np.argmax(y_true_valid, axis=1)
    y_pred_cls = np.argmax(y_pred_valid, axis=1)

    if len(y_true_cls) == 0:
        return 0.0
    else:
        return accuracy_score(y_true_cls, y_pred_cls)


def process_dataframe_with_missing_indices(df, x_name, map_dict, fill_strategy='random'):
    if x_name == 'x':
        sorted_indices = list(map_dict['src'].keys())
    elif x_name == 'x_src':
        sorted_indices = list(map_dict['src'].keys())
    elif x_name == 'x_dst':
        sorted_indices = list(map_dict['dst'].keys())
    else:
        raise ValueError(f"Invalid x_name: {x_name}")

    row_names = df.index.tolist()
    missing_indices = [idx for idx in sorted_indices if idx not in row_names]

    if missing_indices:
        columns = df.columns
        n_cols = len(columns)

        if fill_strategy == 'random':
            fill_data = np.random.uniform(low=-1.0, high=1.0, size=(len(missing_indices), n_cols))
        elif fill_strategy == 'zeros':
            fill_data = np.zeros((len(missing_indices), n_cols))
        else:
            raise ValueError(f"Unsupported fill_strategy: {fill_strategy}")

        missing_df = pd.DataFrame(fill_data, index=missing_indices, columns=columns)
        combined_df = pd.concat([df, missing_df])
        combined_df = combined_df.loc[sorted_indices].reset_index(drop=True)

        return combined_df
    else:
        return df.loc[sorted_indices].reset_index(drop=True)


def extract_id_rdkit2d(data_dict: dict) -> pd.DataFrame:
    data_list = []

    for id, entry in data_dict.items():
        id = id
        rdkit2d = entry.get('rdkit2d', None)

        if id is None or rdkit2d is None:
            continue

        if isinstance(rdkit2d, np.ndarray):
            rdkit2d_flat = rdkit2d.flatten().tolist()
        else:
            rdkit2d_flat = list(rdkit2d)

        row_data = {'id': id}
        row_data.update({f'feature_{i}': val for i, val in enumerate(rdkit2d_flat)})

        data_list.append(row_data)

    if not data_list:
        return pd.DataFrame()
    df = pd.DataFrame(data_list).set_index('id')
    df.index.name = 'id'

    return df

def split_attr_column(df):
    split_data = df.str.split(',', expand=True).astype(int)
    return split_data

def feature_vector(feature_name, df):
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)

    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = pd.DataFrame(feature_matrix, columns=all_feature)

    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1

    df_feature = np.array(df_feature)
    sim_matrix = np.array(Jaccard(df_feature))

    print(feature_name + " len is:" + str(len(sim_matrix[0])))
    return sim_matrix


def get_feature(df_drug, feature_list):
    d_feature = {}
    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  # vector=[]
    for i in feature_list:
        tempvec = feature_vector(i, df_drug)
        vector = np.hstack((vector, tempvec))
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]
    return pd.DataFrame(d_feature).T

def create_aligned_vector(g_map_dict, aux_map_dict, is_aux_hetero):
    temp_list = []

    for node_type in g_map_dict.keys():
        if is_aux_hetero:
            lookup_type = node_type
        else:
            if node_type == 'dst':
                continue
            lookup_type = 'src'

        for orig_name, g_index in g_map_dict[node_type].items():
            if orig_name in aux_map_dict.get(lookup_type, {}):
                aux_value = aux_map_dict[lookup_type][orig_name]
                temp_list.append((g_index, aux_value))
            else:
                continue

    temp_list.sort(key=lambda x: x[0])
    aligned_vector = np.array([item[1] for item in temp_list])

    return aligned_vector


def is_affinity_edge_attr(edge_attr):
    if not isinstance(edge_attr, (torch.Tensor, np.ndarray, list, tuple)):
        return False

    if isinstance(edge_attr, torch.Tensor):
        edge_attr = edge_attr.cpu().numpy()
    elif isinstance(edge_attr, (list, tuple)):
        edge_attr = np.array(edge_attr)
    try:
        edge_attr = edge_attr.astype(float)
    except:
        edge_attr = edge_attr

    if not np.issubdtype(edge_attr.dtype, np.number):
        return False

    return np.all((edge_attr >= 0) & (edge_attr <= 1))


def extract_dta(data_path) -> None:
    with open(data_path + "ligands_can.txt") as f:
        ligands = json.load(f, object_pairs_hook=OrderedDict)
        drug_smi = pd.DataFrame(ligands.items(), columns=["cid", "smi"])
    with open(data_path + "proteins.txt") as f:
        proteins = json.load(f, object_pairs_hook=OrderedDict)
        tar_seq = pd.DataFrame(proteins.items(), columns=["pid", "seq"])
    return drug_smi, tar_seq

def  mask_func(*array, mask_sp='src', test_rate=0.05, test_num=None, seed=42,
              is_hetero=None, undirected=None, node_map = None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    edge_index_copy = copy.deepcopy(array[0])
    src_node = list(node_map['src'].values())
    dst_node = list(node_map['dst'].values())
    if mask_sp == 'src':
        src_set = set(edge_index_copy[0].tolist())
        if is_hetero == True and undirected == True:
            src_set = set([x for x in src_set if x in src_node])

        if test_num != None:
            test_num = test_num
        else:
            test_num = int(len(src_set) * test_rate)

        test_src = np.random.choice(list(src_set), size=test_num, replace=False)
        test_idx = np.isin(edge_index_copy[0], test_src)
        if undirected == True:
            test_idx_2 = np.isin(edge_index_copy[1], test_src)
            test_idx = test_idx | test_idx_2

        train_idx = ~test_idx

    elif mask_sp == 'dst':
        dst_set = set(edge_index_copy[1].tolist())
        if is_hetero == True and undirected == True:
            dst_set = set([x for x in dst_set if x in dst_node])

        if test_num != None:
            test_num = test_num
        else:
            test_num = int(len(dst_set) * test_rate)
        test_gene = np.random.choice(list(dst_set), size=test_num, replace=False)

        test_idx = np.isin(edge_index_copy[1], test_gene)
        if undirected == True:
            test_idx_2 = np.isin(edge_index_copy[0], dst_set)
            test_idx = test_idx | test_idx_2

        train_idx = ~test_idx

    elif mask_sp == 'src_dst':
        src_set = set(edge_index_copy[0].tolist())
        if is_hetero == True and undirected == True:
            src_set = set([x for x in src_set if x in src_node])
        if test_num != None:
            test_num = test_num
        else:
            test_num = int(len(src_set) * test_rate/2)

        test_src = np.random.choice(list(src_set), size=test_num, replace=False)
        test_idx1 = np.isin(edge_index_copy[0], test_src)

        if undirected == True:
            test_idx_2 = np.isin(edge_index_copy[1], test_src)
            test_idx1 = test_idx1 | test_idx_2

        dst_set = set(edge_index_copy[1].tolist())
        if is_hetero == True and undirected == True:
            dst_set = set([x for x in dst_set if x in dst_node])
        if test_num != None:
            test_num = test_num
        else:
            test_num = int(len(dst_set) * test_rate/2)
        test_dst = np.random.choice(list(dst_set), size=test_num, replace=False)
        test_idx2 = np.isin(edge_index_copy[1], test_dst)

        if undirected == True:
            test_idx_2 = np.isin(edge_index_copy[0], dst_set)
            test_idx2 = test_idx2 | test_idx_2
        # test_idx = np.concatenate((test_idx1, test_idx2), axis=-1)
        test_idx = test_idx1 | test_idx2
        train_idx = ~test_idx
    else:
        raise ValueError("Invalid mask_sp value. Should be 'src' or 'dst' or 'src_dst'.")
    return_list = [array[0][:,train_idx].T,array[0][:,test_idx].T]
    for arr in array[1:]:
        return_list.append(arr[train_idx])
        return_list.append(arr[test_idx])

    return (x for x in return_list)


def add_self_loops_with_features(data):
    new_data = Data()
    for key in data.keys():
        setattr(new_data, key, getattr(data, key))

    if not hasattr(data, 'edge_index'):
        raise ValueError("Input data must contain 'edge_index'")
    if not hasattr(data, 'edge_feature'):
        raise ValueError("Input data must contain 'edge_feature'")

    edge_index, _ = add_self_loops(new_data.edge_index,
                                   num_nodes=new_data.num_nodes)

    num_self_loops = new_data.num_nodes
    feature_dim = new_data.edge_feature.size(1)
    self_loop_features = torch.ones((num_self_loops, feature_dim),
                                    dtype=new_data.edge_feature.dtype,
                                    device=new_data.edge_feature.device)

    new_data.edge_index = edge_index
    new_data.edge_feature = torch.cat([new_data.edge_feature,
                                       self_loop_features], dim=0)

    return new_data



def apply_masking(data_x, mask = 'src',ratio=0.05, mode='I'):
    masked_data_x = data_x.copy()
    if mode == 'I':
        mask == 'src_dst'

    if mask == 'src':
        for i in range(masked_data_x.shape[0]):
            masked_cols = np.random.choice(
                masked_data_x.shape[1],
                size=int(masked_data_x.shape[1] * ratio),
                replace=False
            )
            masked_data_x.iloc[i, masked_cols] = 0
    elif mask == 'dst':
        for j in range(masked_data_x.shape[1]):
            masked_rows = np.random.choice(
                masked_data_x.shape[0],
                size=int(masked_data_x.shape[0] * ratio),
                replace=False
            )
            masked_data_x.iloc[masked_rows, j] = 0
    else:
        total_elements = masked_data_x.shape[0] * masked_data_x.shape[1]
        num_elements_to_mask = int(total_elements * ratio)

        flat_indices = np.arange(total_elements)
        masked_flat_indices = np.random.choice(
            flat_indices,
            size=num_elements_to_mask,
            replace=False
        )
        row_indices, col_indices = np.unravel_index(masked_flat_indices, masked_data_x.shape)

        for i, j in zip(row_indices, col_indices):
            masked_data_x.iloc[i, j] = 0
    return masked_data_x