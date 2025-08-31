import os.path
from torch_geometric.utils import add_self_loops
import torch
from .model import UniBIP, ContrastiveLoss
import numpy as np
import setproctitle
import copy
import multiprocessing
import datetime
import warnings
import pandas as pd
from tqdm import tqdm
from .utils import *
from .preprocess import *
contrastiveLoss = ContrastiveLoss()

def training_params(Gdata, params):
    print(f"Random check - Torch: {torch.rand(1)}, Numpy: {np.random.rand(1)}")
    import datetime
    start_time = datetime.datetime.now()  # Record start time for tracking execution time
    warnings.filterwarnings('ignore', message='TypedStorage is deprecated.')  # Suppress specific warnings
    # params.save_file = save_path
    save_path, save_file_name = get_save_file(params)
    model, Gdata,best_result_df,results_df = main_training(Gdata, params)  # Perform cross-validation training
    best_result_df.to_feather(os.path.join(save_path, f"best_result.feather"))
    if Gdata.config.True_edge_matrix == True and params.repeat == 0:
        Gdata.save(os.path.join(save_path, f"gdata.Gdata"))
        results_df.to_feather(os.path.join(save_path, f"all_results.feather"))
    print(f'-----Task {save_file_name}_{params.repeat} completed-----')  # Log the completion of the task
    print_execution_time(start_time)  # Print the execution time for this task
    return model, Gdata, best_result_df,results_df


def get_save_file(params):
    if params.grid_search:
        # 保持与原始代码一致
        save_path = params.save_file + '/' + 'repeat' + '_' + str(params.repeat)
        save_file_name = params.data_name
        print(f'-----Task grid_search {save_path}_{params.repeat}started-----')  # Log the start of the task
    else:
        # 如果 params.data_name 存在，则使用它；否则，省略这部分
        if params.data_name is not None:
            save_file_name = params.data_name
            save_path = params.data_name + '_' + 'results' + '/' + 'repeat' + '_' + str(params.repeat)
        else:
            save_file_name = 'results'
            save_path = save_file_name + '/' + 'repeat' + '_' + str(params.repeat)

        params.save_file = save_path
        print(f'-----Task {save_file_name}_{params.repeat} started-----')  # Log the start of the task

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    return save_path, save_file_name

def calculate_class_weights(pos_labels):
    num_pos = pos_labels.sum(dim=0)
    num_neg = pos_labels.shape[0] - num_pos
    weights = (num_neg / (num_pos + 1e-8)).clamp(max=10.0)
    return weights.to(pos_labels.device)

def train_batch(model, optimizer, train_data, train_pos_edges, train_neg_edges, train_pos_labels, train_neg_labels,
                batch_size=512):
    total_loss = 0
    if train_neg_labels is not None:
        true_value = torch.cat([train_pos_labels, train_neg_labels], dim=0).float()
        edges = torch.cat([train_pos_edges, train_neg_edges], dim=1)
    else:
        true_value = train_pos_labels
        edges = train_pos_edges

    num_samples = true_value.size(0)
    indices = torch.randperm(num_samples)
    true_value = true_value[indices]
    edges = edges[:, indices]

    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        edges_batch = edges[:, start_idx:end_idx]
        true_value_batch = true_value[start_idx:end_idx]
        model.train()
        optimizer.zero_grad()
        with torch.no_grad():
            z = model(train_data)
        pos_pred = model.similarityfunc(z, edges_batch)
        loss = contrastiveLoss(true_value_batch.float(), pos_pred)
        print(f'Batch: {batch_idx}/{num_batches}')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return model, total_loss / num_batches


def test_batch(model, data, pos_edge_index, pos_labels, neg_edge_index=None,  batch_size = 512):
    model.eval()
    with torch.no_grad():
        z = model(data)
        total_loss = 0
        if neg_edge_index is not None:
            neg_labels = torch.zeros_like(neg_edge_index)
            true_value = torch.cat([pos_labels, neg_labels], dim=0).float()
            edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        else:
            true_value = pos_labels
            edges = pos_edge_index

        num_samples = true_value.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size

        pred_all = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            edges_batch = edges[:, start_idx:end_idx]
            pos_pred = model.similarityfunc(z, edges_batch)
            pred_all.append(pos_pred)
        all_probs = torch.cat(pred_all, dim=0)

        return true_value.cpu().numpy(), all_probs.cpu().numpy()

def train(model,optimizer, train_data, train_pos_edges, train_neg_edges, train_pos_labels, train_neg_labels):
    model.train()
    optimizer.zero_grad()
    z = model(train_data)
    pos_pred = model.similarityfunc(z, train_pos_edges)
    if train_neg_edges is None:
        loss = contrastiveLoss(
            train_pos_labels.float(),
            pos_pred
        )
    else:
        neg_pred = model.similarityfunc(z, train_neg_edges)
        loss = contrastiveLoss(
            torch.cat([train_pos_labels, train_neg_labels], dim=0).float(),
            torch.cat([pos_pred, neg_pred], dim=0)
        )
    loss.backward()
    optimizer.step()
    return model, loss.item()

def test(model, data, pos_edge_index, pos_labels, neg_edge_index=None, eval_data=None):
    if not eval_data is None:
        pos_edge_index = torch.tensor(eval_data['edge_idx'][:, eval_data['label']==1]).to(model.device)
        neg_edge_index = torch.tensor(eval_data['edge_idx'][:, eval_data['label']==0]).to(model.device)
        pos_labels = eval_data['class'][eval_data['label']==1]
        neg_labels = eval_data['class'][eval_data['label']==0]
        pos_ = eval_data['label'][eval_data['label']==1]
        neg_ = eval_data['label'][eval_data['label']==0]
        pos_labels = torch.tensor(np.concatenate([pos_labels, pos_.reshape((-1,1))], axis=1))
        neg_labels = torch.tensor(np.concatenate([neg_labels, neg_.reshape((-1,1))], axis=1))
    model.eval()
    with torch.no_grad():
        z = model(data)
        pos_pred = model.similarityfunc(z,pos_edge_index)
        if neg_edge_index is None:
            return pos_labels.cpu().numpy(), pos_pred.cpu().numpy()
        neg_pred = model.similarityfunc(z,neg_edge_index)
        if not eval_data is None:
            all_labels = torch.cat([pos_labels, neg_labels], dim=0).cpu()
            all_probs = torch.cat([pos_pred, neg_pred], dim=0).cpu()
            return all_labels.numpy(), all_probs.numpy()
        neg_labels = torch.zeros_like(neg_pred)
        all_labels = torch.cat([pos_labels,neg_labels],dim = 0).cpu()
        all_probs = torch.cat([pos_pred,neg_pred],dim = 0).cpu()
        return all_labels.numpy(), all_probs.numpy()

def predict(model, data, edge_index):
    model.eval()
    with torch.no_grad():
        z = model(data)
        pred = model.similarityfunc(z,edge_index)
        return pred.cpu()

def main_training(Gdata, params):
    device = params.device
    (data,
     train_data,
     _,
     train_pos_edges,
     train_neg_edges,
     test_pos_edges,
     test_neg_edges,
     train_pos_labels,
     train_neg_labels,
     test_pos_labels,
     test_neg_labels) = Gdata.train_test_split_process(params.val_size, params.neg_ratio, mask_sp=params.mask_sp)
    results_list = []
    data = add_self_loops_with_features(data)
    train_data = add_self_loops_with_features(data)
    model = UniBIP(
        in_feat=data.num_features,
        feature_in_feat=data.feature.shape[1],
        hidden_feat=params.hidden_feat,
        num_classes=data.edge_labels.shape[1],
        GCNELayer_num_layers=params.GCNELayer_num_layers,
        AuxiliaryGCNELayer_num_layers=params.AuxiliaryGCNELayer_num_layers,
        Auxiliary_in_feat=data.auxiliary_data.num_features if hasattr(data, 'auxiliary_data') else 0,
        NodeFeatureAwareness = params.NodeFeatureAwareness,
        grid_size=params.grid_size,
        dr=params.dr,
        texts=Gdata.GraphDict['x_sequence'],
        device=device,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.lr,
        weight_decay=params.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=params.scheduler_factor,patience = params.scheduler_patience
    )
    best_metrics = {'best_index': -np.inf ,'epoch':0}
    class_auc = None
    class_auc_if = False
    train_pos_edges_mask = None
    train_pos_labels_mask = None
    if Gdata.GraphDict['x_sequence'] is not None:
        global train
        global test
        train = train_batch
        test = test_batch
    else:
        pass
    for epoch in tqdm(range(params.epochs+1), desc="Training"):
        if not class_auc is None and (params.epochs-epoch)>params.epochs*0.2 and epoch > params.epochs*0.1 and train_pos_labels.shape[1]>=2:
            if class_auc_if:
                class_auc_if=False
                filtered_classes = [cls for cls, auc in class_auc.items() if auc < 0.5]
                if filtered_classes:
                    filtered_indices = filtered_classes
                    mask = torch.ones(train_pos_labels.shape[0], dtype=torch.bool,device=train_pos_edges.device)
                    for idx in filtered_indices:
                        mask &= (train_pos_labels[:, idx] == 1)
                    train_pos_edges_mask = train_pos_edges[:,mask]
                    train_pos_labels_mask = train_pos_labels[mask]
                    for e in range(min(int(params.epochs * 0.05), 10000) + 1):
                        print(f"\r Subclass: {e}/{min(int(params.epochs * 0.05), 10000)}", end="", flush=True)
                        model, loss = train(model,
                                            optimizer,
                                            train_data,
                                            train_pos_edges_mask,
                                            train_neg_edges,
                                            train_pos_labels_mask,
                                            train_neg_labels,
                                            )

        model, loss = train(model,
                                optimizer,
                                train_data,
                                train_pos_edges,
                                train_neg_edges,
                                train_pos_labels,
                                train_neg_labels,
                                )

        if epoch % params.print_epoch == 0 or epoch == 0:
            print(f"\nEpoch {epoch}: Loss={loss:.4f}")
            y_true, y_pred = test(
                model,
                train_data,
                test_pos_edges,
                test_pos_labels,
                test_neg_edges,
                Gdata.eval_data,
            )
            if Gdata.eval_data:
                pred_class={}
                class_number = y_true.shape[-1]-1
                y_ture_new = []
                y_pred_new = []
                for r in range(class_number):
                    index = y_true[:, r] > 0
                    pred_class[r] = {'true':list(y_true[index, -1]),
                                     'score': list(y_pred[index, r]),
                                     }
                    y_ture_new.append(y_true[index, -1])
                    y_pred_new.append(y_pred[index, r])
                y_ture_new = np.concatenate(y_ture_new)
                y_pred_new = np.concatenate(y_pred_new)
                acc = calculate_accuracy(y_ture_new.reshape((-1,1)), y_pred_new.reshape((-1,1))>0.5)
                print(f'max acc={acc}')
                all_metrics1 = []
                all_metrics2 = []
                all_metric_names = []
                class_auc_dict = {}
                macro_metrics = []
                for cls, pred_data in pred_class.items():
                    y_t, y_p = pred_data['true'], pred_data['score']

                    cls_metrics, metric_names, best_threshold, class_auc = get_metrics(
                        np.array(y_t).reshape(-1, 1),
                        np.array(y_p).reshape(-1, 1),
                        not Gdata.affinity
                    )
                    all_metrics1.append(cls_metrics[2:10])
                    all_metrics2.append(cls_metrics[12:])
                all_metrics1 = np.vstack(all_metrics1).mean(0)
                all_metrics2 = np.vstack(all_metrics2).mean(0)
                metrics = np.array([y_ture_new,y_pred_new,*[_x for _x in all_metrics1],y_ture_new,y_pred_new, *[_x for _x in all_metrics2]])
            else:
                acc = calculate_accuracy(y_true, y_pred)
                print(f'max acc={acc}')
                metrics, metric_names,best_threshold,class_auc = get_metrics(y_true, y_pred, not Gdata.affinity)

            class_auc_if=True
            if not Gdata.affinity:
                current_metrics = metrics[metric_names.index('auc')]
            else:
                # current_metrics = -metrics[metric_names.index('mse')]
                current_metrics = metrics[metric_names.index('pearson')]

            if params.use_scheduler:
                scheduler.step(current_metrics)
            log_entry = [
                *metrics,
                best_threshold,
                loss,
                epoch,
                *params.other_args['arg_value']
            ]
            log_name = [
                *metric_names,
                'best_threshold',
                'loss',
                'epoch',
                *params.other_args['arg_name']
            ]

            if current_metrics > best_metrics['best_index']:
                best_metrics = {
                    'best_index': current_metrics,
                    # 'macro_auc': current_auc,
                    'epoch': epoch
                }
                best_log_entry = log_entry
                model.save_model(path=params.save_file + '/best_model.pth')

            save_list = [best_log_entry]
            results_list.append(log_entry)

    print(f"\nBest Metrics at Epoch {best_metrics['epoch']}:")
    print(f"Best_index: {best_metrics['best_index']:.4f} ")
    print(f'Best_Metrix:{best_metrics}')
    torch.cuda.empty_cache()
    best_result_array = np.array(save_list, dtype=object)
    best_result_df = pd.DataFrame(best_result_array, columns=log_name)
    results_array = np.array(results_list, dtype=object)
    results_df = pd.DataFrame(results_array, columns=log_name)
    Gdata = predict_Gdata_from_data(Gdata, data, model, model_file=params.save_file + '/best_model.pth', batch_size=params.batch_size)
    return model, Gdata, best_result_df, results_df


def predict_Gdata(Gdata, params, model_file=None, device=None,):
    if device is None:
        device = params.device
    else:
        params.device = device
    (data,
    train_data,
    _,
    train_pos_edges,
    train_neg_edges,
    test_pos_edges,
    test_neg_edges,
    train_pos_labels,
    train_neg_labels,
    test_pos_labels,
    test_neg_labels) = Gdata.train_test_split_process(params.val_size, params.neg_ratio, mask_sp=params.mask_sp, device=device)
    model = UniBIP(
        in_feat = data.num_features,
        feature_in_feat = data.feature.shape[1],
        hidden_feat = params.hidden_feat,
        num_classes = data.edge_labels.shape[1],
        GCNELayer_num_layers = params.GCNELayer_num_layers,
        AuxiliaryGCNELayer_num_layers = params.AuxiliaryGCNELayer_num_layers,
        Auxiliary_in_feat = data.auxiliary_data.num_features if hasattr(data, 'auxiliary_data') else 0,
        NodeFeatureAwareness = params.NodeFeatureAwareness,
        grid_size = params.grid_size,
        dr = params.dr,
        texts = Gdata.GraphDict['x_sequence'],
        device = device,
    ).to(device)
    if model_file is None:
        model_file = params.save_file + '/best_model.pth'
    Gdata = predict_Gdata_from_data(Gdata, data, model,model_file,True_edge_matrix=True, batch_size=params.batch_size)
    return Gdata,model

def predict_Gdata_from_data(Gdata,data, model, model_file, True_edge_matrix=False, batch_size=10000):
    model.load_model(model_file)

    model.eval()
    with torch.no_grad():
        z = model(data)
        Gdata['embedding'] = z.cpu().detach().numpy()
        if Gdata.config.True_edge_matrix == True or True_edge_matrix==True:
            z = model(data)
            Gdata['embedding'] = z.cpu().detach().numpy()
            src_edge = np.array(list(Gdata.map_dict['src'].values()))
            dst_edge = np.array(list(Gdata.map_dict['dst'].values()))
            edge_index = []
            for s in src_edge:
                for d in dst_edge:
                    if Gdata.is_hetero:
                        edge_index.append([s, d])
                    else:
                        if d >= s:
                            edge_index.append([s, d])
            edge_index = torch.tensor(np.array(edge_index).T, device=z.device)
            pred_list = []
            for i in range(0, edge_index.shape[-1], batch_size):
                end_i = min(i + batch_size, edge_index.shape[-1])
                edge_index_batch = edge_index[:, i:end_i]
                values = model.similarityfunc(z, edge_index_batch)
                pred_list.append(values.cpu().detach())
            edge_labels = torch.cat(pred_list, dim=0)
            if Gdata.is_hetero == True:
                edge_index[1] -= (edge_index[0].max() + 1)
            Gdata['EdgeScores'] = Gdata.edge2matrix(edge_index=edge_index.cpu().detach().numpy(),
                                                    edge_labels=edge_labels.cpu().detach().numpy())
            Gdata['True_edge_matrix'] = Gdata.GraphDict['True_edge_matrix']
            src_emb = z[torch.tensor(src_edge, device=z.device)]
            dst_emb = z[torch.tensor(dst_edge, device=z.device)]
            Gdata['src_embedding'] = src_emb.cpu().detach().numpy()
            Gdata['dst_embedding'] = dst_emb.cpu().detach().numpy()
    return Gdata