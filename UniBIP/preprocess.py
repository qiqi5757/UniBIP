import copy
import os.path
import numpy as np
import pandas as pd
import joblib
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_networkx, negative_sampling, add_self_loops
from sklearn.model_selection import ParameterGrid
from .utils import *

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

    return not np.all(np.isin(edge_attr, [0, 1])) and np.all((edge_attr >=0) & (edge_attr <= 1))

class Config:
    """Configuration class to store model and training parameters."""
    def __init__(self, gpu_id=None,epochs=1000, print_epoch=1,repeats=10,):
        # File paths for data and saving results
        self.datapath = './data/'  # Path where datasets are stored
        self.save_file = './save_result/'  # Directory to save results
        # if not os.path.exists(self.save_file):
        #     os.mkdir(self.save_file)
        # Training parameters
        self.epochs = epochs # Number of training epochs
        self.print_epoch = print_epoch
        # self.print_epoch = int(self.epochs/1) # Frequency of printing training status
        # Hyperparameters for training the model
        self.lr = 0.003  # Learning rate
        self.weight_decay = 0  # Regularization parameter
        self.scheduler_factor = 0.99  # Learning rate decay factor
        self.scheduler_patience = 5  # Step size for learning rate decay
        self.early_stop_patience = 30  # Patience for early stopping

        # Hyperparameters for model architecture
        self.hidden_feat = 128
        self.out_channels = 128  # Number of output channels in the model
        self.num_layers = 8  # Number of layers in the model
        self.num_channels = 8  # Number of channels in each layer
        self.dr = 0.2  # Dropout rate
        self.grid_size = 6

        self.repeats = repeats
        self.repeat = 1

        self.cls = 7
        self.embeddingLayer_num_layers = 4
        self.GCNELayer_num_layers=6
        self.AuxiliaryGCNELayer_num_layers=0
        self.Auxiliary_in_feat=0
        self.EdgeFeatureAwareness = True
        self.NodeFeatureAwareness = True

        self.batch_size = 512
        self.val_size=0.1
        self.neg_ratio = 1.0

        self.data_name = None
        self.file_path = None
        self.features_file_src = None
        self.features_file_dst = None

        self.use_scheduler = True

        self.param_search = {}
        self.grid_search = False
        self.other_args = {'arg_name': [], 'arg_value': []}

        self.mask_sp = None

        self.True_edge_matrix = False
        if gpu_id is None:
            self.device = 'cpu'
        else:
            self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')  # Select device: GPU or CPU

    def set_random(self, seed=42):
        seed = seed + self.repeat
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)  # Set seed for PyTorch
        np.random.seed(seed)  # Set seed for NumPy
        os.environ['PYTHONHASHSEED'] = str(seed)  # Set seed for Python environment (hash functions)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # Set seed for all CUDA devices
            # Ensure deterministic operations for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def yield_attr(self,param_search):
        yield from self._set_attr(param_search)

    def get_list_attr(self,param_search):
        config_generator = self._set_attr(param_search)
        config_list = []
        for c in config_generator:
            config_list.append(c)
        return config_list

    def _set_attr(self, param_search):
        """
        Generate configurations based on the parameter grid for hyperparameter search.

        Parameters:
        config (Config): The initial configuration object.
        param_search (dict): A dictionary defining the grid of parameters to search.

        Yields:
        Config: A new configuration object with specific parameters set according to the grid search.
        """
        # Create a list of all possible parameter combinations
        if 'repeat' not in param_search.keys():
            param_search['repeat'] = list(range(self.repeats))
        self.param_search = param_search
        param_grid_list = list(ParameterGrid(param_search))

        for param in param_grid_list:
            # Make a deep copy of the base configuration to avoid modifying the original
            new_config = copy.deepcopy(self)
            new_config.other_args = {'arg_name': [],
                                     'arg_value': []}  # Store additional arguments for logging or debugging

            # Iterate over each parameter and set it in the new configuration
            if self.data_name is None:
                save_file = new_config.save_file
            else:
                save_file = str(self.data_name) + '_'+ 'results' + '_'+ 'grid_search'+'/'
            for key, value in param.items():
                setattr(new_config, key, value)  # Dynamically set the attribute in the config
                new_config.other_args['arg_name'].append(key)  # Store the name of the argument
                new_config.other_args['arg_value'].append(value)  # Store the value of the argument
                print(f"{key}: {value}")  # Print the parameter and its value for tracking
                if key != 'repeat':
                    save_file += key+'_'+str(value)+'_'
            new_config.save_file = save_file + 'results'
            # new_config.save_file = save_file + '/' + f"repeat{param['repeat']}/"
            if not os.path.exists(new_config.save_file) and self.grid_search:
                os.makedirs(new_config.save_file, exist_ok=True)
            yield new_config  # Yield the newly created configuration

    def copy(self):
        return copy.deepcopy(self)

class BioGraphData():
    def __init__(self,is_undirected=True, mode='Isomorphic', config = Config(), main_Gdata=None, Auxiliary_Gdata=None,):
        assert mode in ['Isomorphic', 'Heterogeneous'] or mode in ['I', 'H','i','h']
        device = config.device
        self.undirected=is_undirected
        self.GraphDict = {}
        self.eval_data = None

        if mode in ['Heterogeneous', 'H', 'h']:
            self.is_hetero = True
            self.GraphDict = {
                'x_src': None,
                'x_dst': None,
                'edge_index':None,
                'edge_labels':None,
                'edge_attr':None,
                'x_sequence': None,
                'meta':{'mode':'Heterogeneous'}
            }
        else:
            self.is_hetero = False
            self.GraphDict = {
                'x': None,
                'edge_index': None,
                'edge_labels': None,
                'edge_attr': None,
                'x_sequence':None,
                'meta': {'mode': 'Isomorphic'}
            }
        self.device = device
        self.data = None
        self.main_Gdata = main_Gdata
        self.Auxiliary_Gdata = Auxiliary_Gdata
        self.isAuxiliary = False
        self._index_x_init = None
        self.config = config
        self.affinity = False
        True_edge_matrix = self.config.True_edge_matrix

        if not main_Gdata is None:
            assert isinstance(main_Gdata, BioGraphData)
            self.isAuxiliary = True
            self.device = main_Gdata.device
            self.config = main_Gdata.config.copy()
            self._set_config(config.copy())
            main_Gdata.Auxiliary_Gdata = self
            self.config.True_edge_matrix = False

        if not Auxiliary_Gdata is None:
            assert isinstance(Auxiliary_Gdata, BioGraphData)
            self.Auxiliary_Gdata.isAuxiliary = True
            self.Auxiliary_Gdata.device = device
            self.Auxiliary_Gdata.config = config.copy()
            self.Auxiliary_Gdata._set_config(config.copy())
            self.Auxiliary_Gdata.main_Gdata = self
            self.Auxiliary_Gdata.config.True_edge_matrix = False

        if self.isAuxiliary == True:
            self.config.True_edge_matrix = False
        else:
            self.config.True_edge_matrix = True_edge_matrix

    def _set_config(self,config):
        config.AuxiliaryGCNELayer_num_layers = 1
        config.Auxiliary_in_feat = 1

    def _map(self, edge_index, all_edge):
        edge_index = edge_index.astype(str)
        all_edge = all_edge.astype(str)
        if self.GraphDict['meta']['mode'] == 'Isomorphic':
            key = pd.unique(np.concatenate([all_edge[0],all_edge[1]],axis = 0))
            self.node_number = len(key)
            self.map_dict = {
                'src':{val: idx for idx, val in enumerate(key)}
                }
            self.reverse_map_dict = {
                'src': {idx: val for val, idx in self.map_dict['src'].items()}
                }
            self.mapped_edge_index = np.array([
                [self.map_dict['src'][node] for node in edge_index[0]],
                [self.map_dict['src'][node] for node in edge_index[1]]
            ])
            self.map_dict['dst'] = self.map_dict['src']
            self.reverse_map_dict['dst'] = self.reverse_map_dict['src']
        else:
            key_src = pd.unique(all_edge[0])
            key_dst = pd.unique(all_edge[1])
            self.src_number = len(list(key_src))
            self.dst_number = len(list(key_dst))
            self.node_number = self.src_number + self.dst_number
            number_src = key_src.shape[0]

            self.map_dict = {
                'src': {val: idx for idx, val in enumerate(key_src)},
                'dst': {val: idx+number_src for idx, val in enumerate(key_dst)}
            }
            self.reverse_map_dict = {
                'src': {idx: val for val, idx in self.map_dict['src'].items()},
                'dst': {idx: val for val, idx in self.map_dict['dst'].items()}
            }
            self.mapped_edge_index = np.array([
                [self.map_dict['src'][node] for node in edge_index[0]],
                [self.map_dict['dst'][node] for node in edge_index[1]]
            ])

        node_name = []
        for k,v in self.map_dict.items():
            for name in v:
                name = str(name)
                if name in node_name:
                    name += str(k)
                node_name.append(name)
        self.node_name = np.array(node_name)
        self.GraphDict['node_name'] = self.node_name
        self.tests=None
        return self.mapped_edge_index

    def apply_map(self, edge_index):
        if self.GraphDict['meta']['mode'] == 'Isomorphic':
            mapped_edge_index = np.array([
                [self.map_dict['src'][node] for node in edge_index[0]],
                [self.map_dict['src'][node] for node in edge_index[1]]
            ])
        else:
            mapped_edge_index = np.array([
                [self.map_dict['src'][node] for node in edge_index[0]],
                [self.map_dict['dst'][node] for node in edge_index[1]]
            ])
        return mapped_edge_index

    def reverse_map(self, mapped_edge_index):
        if self.GraphDict['meta']['mode'] == 'Isomorphic':
            original_edge_index = np.array([
                [self.reverse_map_dict['src'][node] for node in mapped_edge_index[0]],
                [self.reverse_map_dict['src'][node] for node in mapped_edge_index[1]]
            ])
        else:
            original_edge_index = np.array([
                [self.reverse_map_dict['src'][node] for node in mapped_edge_index[0]],
                [self.reverse_map_dict['dst'][node] for node in mapped_edge_index[1]]
            ])
        return original_edge_index

    def reverse_map_for_I(self, mapped_edge_index):
        original_edge_index = np.array([
            [self.reverse_map_dict['src'][node] if node in self.reverse_map_dict['src'].keys() else
             self.reverse_map_dict['dst'][node] for node in mapped_edge_index[0]],
            [self.reverse_map_dict['src'][node] if node in self.reverse_map_dict['src'].keys() else
             self.reverse_map_dict['dst'][node] for node in mapped_edge_index[1]]
        ])
        return original_edge_index

    def load_edge(self, file_path=None,edge=None,edge_attr=None, index_col=None, header=None, sep=' ', main_Gdata=None):
        edge_labels = edge_attr
        if not file_path is None:
            df = pd.read_csv(file_path, index_col=index_col, header=header, sep=sep)
        else:
            df = pd.DataFrame(edge)
        if df.shape[-1]>10:
            edge_matrix = np.array([
                (row_idx, col, value)
                for row_idx, row in df.iterrows()
                for col, value in row.items()])
        else:
            edge_matrix = df.values
            if edge_matrix.shape[-1] < 3:
                edge_matrix = np.concatenate([edge_matrix, np.ones((edge_matrix.shape[0], 1))], axis=1)
        if not main_Gdata is None:
            self.main_Gdata = main_Gdata
        if self.isAuxiliary:
            assert not self.main_Gdata is None
            main_edge_matrix = self.main_Gdata.edge_matrix
            assert isinstance(main_edge_matrix, np.ndarray)
            main_edge_matrix = copy.deepcopy(main_edge_matrix)
            main_edge_matrix[:,-1] = 0
            edge_matrix = np.concatenate([main_edge_matrix, edge_matrix], axis=0)
        all_edge = edge_matrix[:,:2]
        try:
            mask = edge_matrix[:,-1].astype(float) != 0
            edge_matrix[:, -1] = edge_matrix[:, -1].astype(float)
        except:
            mask = edge_matrix[:,-1] != 0

        self.edge_matrix=edge_matrix
        edge_index = edge_matrix[:,:2][mask].T
        try:
            edge_index = edge_index.astype(int)
            all_edge = all_edge.astype(int)
        except:
            pass
        edge_index = edge_index.astype(str)
        if not edge_labels is None:
            edge_labels = edge_labels[mask]
        else:
            edge_labels = edge_matrix[:,-1][mask].reshape((-1,1))
        edge_index = self._map(edge_index, all_edge.T)
        if self.undirected==True:
            self.GraphDict['edge_index'] = np.concatenate([edge_index,edge_index[::-1]], axis=1).astype(int)
            self.GraphDict['edge_labels'] = np.concatenate([edge_labels,edge_labels], axis=0)
        else:
            self.GraphDict['edge_index'] = edge_index.astype(int)
            self.GraphDict['edge_labels'] = edge_labels

        if edge_labels.shape[-1] > 1:
            self.GraphDict['edge_labels'] = self.GraphDict['edge_labels']
            self.GraphDict['meta']['class_mode'] = 'Multiclass'
            self.GraphDict['meta']['edge_class'] = edge_labels.shape[-1]
        else:
            if is_affinity_edge_attr(self.GraphDict['edge_labels']):
                edge_class = [1]
                self.affinity = True
            else:
                edge_class = np.unique(self.GraphDict['edge_labels'])
            self.GraphDict['meta']['edge_class'] = edge_class
            if len(edge_class)>1:
                self.OneHotencoder = OneHotEncoder(sparse_output=False)
                self.GraphDict['edge_labels'] = self.OneHotencoder.fit_transform(self.GraphDict['edge_labels']).astype(float)
                edge_labels = self.OneHotencoder.fit_transform(edge_labels).astype(float)
                self.GraphDict['meta']['class_mode'] = 'Multiclass'
            else:
                self.GraphDict['edge_labels'] = self.GraphDict['edge_labels'].astype(float)
                edge_labels = edge_labels.astype(float)
                self.GraphDict['meta']['class_mode'] = 'Binaryclass'
        if self.isAuxiliary:
            assert not self.main_Gdata is None
            self.create_aligned_vector(self.main_Gdata)

        if self.config.True_edge_matrix == True and not self.isAuxiliary:
            if self.is_hetero == True:
                edge_index[1] -= (edge_index[0].max() + 1)
            self.GraphDict['True_edge_matrix'] = self.edge2matrix(edge_index, edge_labels)
        self.GraphDict['edge_attr'] = self.GraphDict['edge_labels']

    def load_edge_featrue(self, edge_attr=None):
        assert edge_attr.shape[0] == self.GraphDict['edge_index'].shape[0]
        self.GraphDict['edge_attr'] = edge_attr

    def edge2matrix(self, edge_index, edge_labels):
        if len(edge_labels.shape) == 1:
            edge_labels = edge_labels.reshape(-1, 1)
        if isinstance(edge_labels, pd.DataFrame):
            edge_labels = edge_labels.values

        src_names = [k for k in sorted(self.map_dict['src'], key=lambda x: self.map_dict['src'][x])]
        dst_names = [k for k in sorted(self.map_dict['dst'], key=lambda x: self.map_dict['dst'][x])]

        matrix_dict = {}
        for i in range(edge_labels.shape[-1]):
            adj_df = pd.DataFrame(
                0,
                index=src_names,
                columns=dst_names,
                dtype=edge_labels.dtype
            )

            try:
                src_indices = edge_index[0].astype(int)
                dst_indices = edge_index[1].astype(int)
                values = edge_labels[:, i]

                for src_idx, dst_idx, val in zip(src_indices, dst_indices, values):
                    src_name = src_names[src_idx]
                    dst_name = dst_names[dst_idx]
                    adj_df.loc[src_name, dst_name] = val
            except:
                src_indices = edge_index[0]
                dst_indices = edge_index[1]
                values = edge_labels[:, i]
                for src_idx, dst_idx, val in zip(src_indices, dst_indices, values):
                    src_name = src_idx
                    dst_name = dst_idx
                    adj_df.loc[src_name, dst_name] = val
            matrix_dict[i] = adj_df
        return matrix_dict

    def load_features_matrix(self,file_path=None,data=None, x_name='x', index_col=False, header=None, sep=' ', norm=True):
        assert x_name in self.GraphDict.keys()
        if not file_path is None:
            df = pd.read_csv(file_path, index_col=index_col, header=header, sep=sep)
        else:
            df = pd.DataFrame(data)
        df.index = df.index.astype(str)
        if not (index_col is None or index_col is False):
            row_names = df.index.tolist()
            if x_name == 'x':
                sorted_indices = list(self.map_dict['src'].keys())
            if x_name == 'x_src':
                sorted_indices = list(self.map_dict['src'].keys())
            if x_name == 'x_dst':
                sorted_indices = list(self.map_dict['dst'].keys())
            missing_indices = [idx for idx in sorted_indices if idx not in row_names]
            assert not missing_indices, f"Missing indices in data: {missing_indices}"
            sorted_indices = np.array(sorted_indices)
            df = df.loc[sorted_indices].reset_index(drop=True)
        else:
            if x_name == 'x':
                assert df.shape[0] == self.node_number
            if x_name == 'x_src':
                assert df.shape[0] == self.src_number
            if x_name == 'x_dst':
                assert df.shape[0] == self.dst_number

        similarity_matrix = df.values.astype(float)
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)
        if norm==True:
            similarity_matrix = softmax(similarity_matrix)
        self.GraphDict[x_name] = similarity_matrix


    def load_sequence_matrix(self,file_path=None,data=None, x_name='x', index_col=None, header=None, sep=' ', norm=True):
        assert x_name in self.GraphDict.keys()
        if not file_path is None:
            df = pd.read_csv(file_path, index_col=index_col, header=header, sep=sep)
        else:
            df = pd.DataFrame(data)

        if not index_col is None:
            row_names = df.index.tolist()
            if x_name == 'x':
                sorted_indices = list(self.map_dict['src'].keys())
            if x_name == 'x_src':
                sorted_indices = list(self.map_dict['src'].keys())
            if x_name == 'x_dst':
                sorted_indices = list(self.map_dict['dst'].keys())
            missing_indices = [idx for idx in sorted_indices if idx not in row_names]
            assert not missing_indices, f"Missing indices in data: {missing_indices}"
            sorted_indices = np.array(sorted_indices)
            df = df.loc[sorted_indices].reset_index(drop=True)
        else:
            if x_name == 'x':
                assert df.shape[0] == self.node_number
            if x_name == 'x_src':
                assert df.shape[0] == self.src_number
            if x_name == 'x_dst':
                assert df.shape[0] == self.dst_number
        similarity_matrix = df.values.astype(str)
        self.GraphDict[x_name + '_sequence'] = similarity_matrix

    def load_eval_data(self,edge_idx, label, class_label=None):
        try:
            edge_idx = np.array(edge_idx).astype(int).astype(str)
        except:
            edge_idx = np.array(edge_idx).values().astype(str)
        edge_idx = self.apply_map(edge_idx)
        assert len(label.shape) == 1
        if class_label is None:
            if label.shape[1]==1:
                oh = OneHotEncoder()
                class_label = oh.fit_transform(label)
        self.eval_data =  {'edge_idx':edge_idx,
                            'class': class_label,
                            'label': label
                            }

    def generate_random_feature(self, feature_dim=128):
        if self.GraphDict['meta']['mode'] == 'Heterogeneous':
            num_src_nodes = len(list(self.map_dict['src'].keys()))
            num_dst_nodes = len(list(self.map_dict['dst'].keys()))

            self.GraphDict['x_src'] = np.random.rand(num_src_nodes, feature_dim).astype(np.float32)
            self.GraphDict['x_dst'] = np.random.rand(num_dst_nodes, feature_dim).astype(np.float32)
            self.ConcatPad_feature()
        else:
            num_nodes = len(list(self.map_dict['src'].keys()))
            self.GraphDict['x'] = np.random.rand(num_nodes, feature_dim).astype(np.float32)

    def ConcatPad_feature(self,one_hot=True):
        if self.GraphDict['meta']['mode'] == 'Heterogeneous':
            assert (not self.GraphDict['x_src'] is None) and (not self.GraphDict['x_dst'] is None)
            feature_1 = self.GraphDict['x_src'].astype(np.float32)
            feature_2 = self.GraphDict['x_dst'].astype(np.float32)
            feature = np.zeros((feature_1.shape[0] + feature_2.shape[0], max(feature_1.shape[1], feature_2.shape[1])),
                               dtype=np.float32)
            feature[:feature_1.shape[0], :feature_1.shape[1]] = feature_1
            feature[feature_1.shape[0]:, :feature_2.shape[1]] = feature_2
            if one_hot == True:
                num_src_nodes = feature_1.shape[0]
                num_dst_nodes = feature_2.shape[0]
                one_hot = np.zeros((num_src_nodes + num_dst_nodes, 2), dtype=np.float32)
                one_hot[:num_src_nodes, 0] = 1
                one_hot[num_src_nodes:, 1] = 1
                feature = np.concatenate([feature, one_hot], axis=1)
            self.GraphDict['x'] = feature
        else:
            raise ValueError('GraphDict is Isomorphic..')

    def Concat_sequence(self, max_len=512):
        if self.GraphDict['meta']['mode'] == 'Heterogeneous':
            assert (not self.GraphDict['x_src' + '_sequence'] is None) and (
                not self.GraphDict['x_dst' + '_sequence'] is None)
            truncate_str = np.vectorize(lambda x: x[:max_len])

            feature_1 = self.GraphDict['x_src' + '_sequence'].astype(str)
            feature_2 = self.GraphDict['x_dst' + '_sequence'].astype(str)

            feature_1_truncated = truncate_str(feature_1)
            feature_2_truncated = truncate_str(feature_2)

            self.GraphDict['x' + '_sequence'] = np.concatenate([feature_1_truncated, feature_2_truncated])
        else:
            raise ValueError('GraphDict is Isomorphic..')


    def load_I2H_features_matrix(self, file_path=None, data=None, index_col=None, header=None, sep=' ', norm=True):
        self.load_features_matrix(file_path=file_path, data=data, x_name='x_src', index_col=index_col, header=header,
                                  sep=sep, norm=norm)
        self.load_features_matrix(file_path=file_path, data=data, x_name='x_dst', index_col=index_col, header=header,
                                  sep=sep, norm=norm)
        self.ConcatPad_feature()


    def self_to_torch_geometric_data(self,device=None,GraphEncoding='eye'):
        data = Data()
        data.num_nodes = self.node_number
        if GraphEncoding == 'eye':
            data.x = torch.eye(data.num_nodes).float()
        elif GraphEncoding == 'random':
            data.x = torch.rand((data.num_nodes, 128)).float()
        if self.GraphDict['x'] is None:
            self.generate_random_feature()
        data.feature =  torch.tensor(self.GraphDict['x']).float()
        data.edge_index = torch.tensor(self.GraphDict['edge_index'].astype(np.int32))
        data.edge_labels = torch.tensor(self.GraphDict['edge_labels'].astype(np.float32))
        data.edge_feature = torch.tensor(self.GraphDict['edge_attr'].astype(np.float32))

        if not self._index_x_init is None:
            data.index_x_init = torch.tensor(self._index_x_init.astype(np.int32))
        if not device is None:
            self.device = device
        self.data = data.to(self.device)
        if not self.Auxiliary_Gdata is None:
            Auxiliary_data = self.Auxiliary_Gdata.self_to_torch_geometric_data(device=device, GraphEncoding=GraphEncoding)
            data.auxiliary_data = Auxiliary_data
        return data

    def train_data_process(self,neg_ratio=1.0):
        edges = self.GraphDict['edge_index']
        edge_labels = self.GraphDict['edge_labels']
        num_classes = edge_labels.shape[-1]
        if self.data is None:
            self.self_to_torch_geometric_data()
        pos_neg_edges = self._prepare_edges(edges.T, None, neg_ratio)
        train_pos_labels = torch.from_numpy(edge_labels).float()
        data = self.data
        train_data = data.clone()
        train_data.edge_index = pos_neg_edges['train'][0]
        train_data.edge_labels = train_pos_labels

        components = [
            data,
            train_data,
            pos_neg_edges['train'][0],
            pos_neg_edges['train'][1],
            train_pos_labels,
            torch.zeros(len(pos_neg_edges['train'][1].T), num_classes).float(),
        ]
        return tuple(self._move_to_device(x) for x in components)

    def create_aligned_vector(self, main_Gdata):
        self._index_x_init = create_aligned_vector(self.map_dict, main_Gdata.map_dict, self.is_hetero)

    def train_test_split_process(self, val_size=0.33, neg_ratio=1.0, mask_sp=None, device=None, not_neg=False):
        edges = self.GraphDict['edge_index']
        edge_labels = self.GraphDict['edge_labels']
        edge_attr = self.GraphDict['edge_attr']
        if not device is None:
            self.device=device
        if mask_sp is not None:
            (edges_train, edges_test, labels_train, labels_test, edge_attr_train, edge_attr_test) = mask_func(
                edges, edge_labels, edge_attr, mask_sp=mask_sp, test_rate=val_size, is_hetero=self.is_hetero, undirected=self.undirected, node_map = self.map_dict)
        else:
            if edge_labels.shape[-1] >=2:
                edge_classes = edge_labels.argmax(axis=1)

                (edges_train, edges_test,
                 labels_train, labels_test, edge_attr_train, edge_attr_test) = train_test_split(
                    edges.T,
                    edge_labels,
                    edge_attr,
                    test_size=val_size,
                    stratify=edge_classes,
                )
            else:
                (edges_train, edges_test, labels_train, labels_test,edge_attr_train, edge_attr_test) = train_test_split(
                    edges.T, edge_labels,edge_attr, test_size=val_size)
        if 'True_edge_matrix' in self.GraphDict.keys():
            self.GraphDict['True_edge_matrix']['train_data'] = self.GraphDict['True_edge_matrix'][0].copy()
            edges_test_names = self.reverse_map_for_I(edges_test.T)
            for src_idx, dst_idx in zip(edges_test_names[0], edges_test_names[1]):
                self.GraphDict['True_edge_matrix']['train_data'].loc[src_idx, dst_idx] = 0
        pos_neg_edges = self._prepare_edges(edges_train, edges_test, neg_ratio)
        if edge_labels.shape[-1]>=2 or self.affinity or not_neg:
            pos_neg_edges['train'][1] = None
            pos_neg_edges['test'][1] = None
        if self.data is None:
            self.self_to_torch_geometric_data()
        return self._assemble_final_data(self.data, pos_neg_edges, labels_train, labels_test,edge_attr_train, edge_attr_test)

    def train_test_split_data(self, val_size=0.33,neg_ratio=10.0,mask_sp=None,device=None):
        self.train_test_splited_data = self.train_test_split_process(val_size, neg_ratio,mask_sp,device)

    def _prepare_edges(self, edges_train, edges_test=None,neg_ratio=1.0):
        def _generate_negatives(pos_edges):
            if self.is_hetero:
                return self._generate_hetero_negatives(pos_edges,neg_ratio)
            return self._generate_homo_negatives(pos_edges,neg_ratio)
        train_pos = torch.tensor(edges_train).t().long()
        if not edges_test is None:
            test_pos = torch.tensor(edges_test).t().long()
            return {
                'train': [train_pos, _generate_negatives(train_pos)],
                'test': [test_pos, _generate_negatives(test_pos)]
            }
        else:
            return {
                'train': [train_pos, _generate_negatives(train_pos)],
                'test':[None, None]
            }

    def _generate_hetero_negatives(self, pos_edges,neg_ratio):
        neg_edges = []
        existing = set(map(tuple, pos_edges.T.numpy()))
        while len(neg_edges) < len(pos_edges.T)*neg_ratio:
            src = np.random.randint(0, self.src_number)
            dst = np.random.randint(self.src_number, self.node_number)
            if (src, dst) not in existing and (dst, src) not in existing:
                neg_edges.append([src, dst])
        return torch.tensor(neg_edges).t().long()

    def _generate_homo_negatives(self, pos_edges,neg_ratio):
        neg_edges = []
        existing = set(map(tuple, pos_edges.T.numpy()))
        while len(neg_edges) < len(pos_edges.T)*neg_ratio:
            src, dst = np.random.randint(0, self.node_number, 2)
            if (src, dst) not in existing and (dst, src) not in existing:
                neg_edges.append([src, dst])
        return torch.tensor(neg_edges).t().long()

    def _random_edge_attr(self,pos_neg_edges, edge_attr):
        return np.random.random((pos_neg_edges.shape[0], edge_attr.shape[1]))

    def _assemble_final_data(self, data, pos_neg_edges, labels_train, labels_test, edge_attr_train=None, edge_attr_test=None):
        train_pos_labels = torch.from_numpy(labels_train).float()
        test_pos_labels = torch.from_numpy(labels_test).float()
        edge_attr_train = torch.from_numpy(edge_attr_train).float()
        edge_attr_test = torch.from_numpy(edge_attr_test).float()

        num_classes = labels_train.shape[1]

        train_data = data.clone()
        train_data.edge_index = pos_neg_edges['train'][0]
        train_data.edge_labels = train_pos_labels
        train_data.edge_feature = edge_attr_train

        test_data = data.clone()
        test_data.edge_index = pos_neg_edges['train'][0]
        test_data.edge_labels = train_pos_labels
        test_data.edge_feature = edge_attr_train

        components = [
            data,
            train_data,
            test_data,
            pos_neg_edges['train'][0],
            pos_neg_edges['train'][1],
            pos_neg_edges['test'][0],
            pos_neg_edges['test'][1],
            train_pos_labels,
            torch.zeros(len(pos_neg_edges['train'][1].T), num_classes).float() if not pos_neg_edges['train'][1] is None else None,
            test_pos_labels,
            torch.zeros(len(pos_neg_edges['test'][1].T), num_classes).float() if not pos_neg_edges['test'][1] is None else None
        ]

        return tuple(self._move_to_device(x) for x in components)

    def _move_to_device(self, obj):
        if isinstance(obj, (torch.Tensor, Data)):
            return obj.to(self.device)
        return obj

    def keys(self):
        return self.GraphDict.keys()

    def values(self):
        return self.GraphDict.values()

    @property
    def __call__(self):
        return self.GraphDict

    def save(self, file_path):
        """Save the BioGraphData instance to a file using joblib."""
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path):
        """Load a BioGraphData instance from a file using joblib."""
        obj = joblib.load(file_path)
        if not isinstance(obj, BioGraphData):
            raise TypeError(f"Loaded object is not an instance of BioGraphData. Got {type(obj).__name__} instead.")
        return obj

    def copy(self):
        return copy.deepcopy(self)

    def __getitem__(self, key):
        assert key in self.GraphDict.keys()
        return self.GraphDict[key]

    def __setitem__(self, key, value):
        self.GraphDict[key] = value

    def __repr__(self):
        return f"MyClass(data={self.GraphDict})"
