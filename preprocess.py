import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm

DATASET_DIRECTORY = '/path/to/dataset'  # Replace with actual path of CICIoT2023 dataset
NO_FILES = 1000000  # Limit number of files to process

SERVER_SPLIT = 0.2  # Proportion of data allocated to the server
TESTING_SPLIT = 0.2  # Proportion of data allocated for testing
CLIENTS_NUM = 5  # Number of clients for IID partitioning
BATCH_SIZE = 64  # Batch size for DataLoader

iid_clients = [f"client_{i}" for i in range(CLIENTS_NUM)]
noniid_clients = ["ddos", "dos", "mirai", "recon", "spoofing", "web", "bruteforce"]

# Constants
X_COLUMNS = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'urg_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
    'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
    'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
    'Radius', 'Covariance', 'Variance', 'Weight',
]
Y_COLUMN = 'label'

CATEGORIES = ['DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web', 'BruteForce']

LABELS = {
    'DDoS': [
        'DDoS-RSTFINFlood', 'DDoS-PSHACK_Flood', 'DDoS-SYN_Flood',
        'DDoS-UDP_Flood', 'DDoS-TCP_Flood', 'DDoS-ICMP_Flood',
        'DDoS-SynonymousIP_Flood', 'DDoS-ACK_Fragmentation',
        'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
        'DDoS-SlowLoris', 'DDoS-HTTP_Flood',
    ],
    'DoS': ['DoS-UDP_Flood', 'DoS-SYN_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood'],
    'Mirai': ['Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain'],
    'Recon': [
        'Recon-PingSweep', 'Recon-OSScan', 'Recon-PortScan',
        'VulnerabilityScan', 'Recon-HostDiscovery',
    ],
    'Spoofing': ['DNS_Spoofing', 'MITM-ArpSpoofing'],
    'Web': [
        'BrowserHijacking', 'Backdoor_Malware', 'XSS',
        'Uploading_Attack', 'SqlInjection', 'CommandInjection',
    ],
    'BruteForce': ['DictionaryBruteForce'],
    'Benign': ['BenignTraffic'],
}


# Read and combine datasets
def load_dataset(nofiles=NO_FILES):
    """Load datasets from the directory."""
    df_paths = sorted([
        os.path.join(DATASET_DIRECTORY, f)
        for f in os.listdir(DATASET_DIRECTORY) if f.endswith('.csv')
    ])[:nofiles]
    datasets = []
    for csv_file in tqdm.tqdm(df_paths):
        print(f"Reading {csv_file}...")
        datasets.append(pd.read_csv(csv_file, usecols=X_COLUMNS + [Y_COLUMN]))
    
    return pd.concat(datasets, ignore_index=True)
# Sample and shuffle datasets based on limits
def sample_and_shuffle(dataset, labels, benign_partition, limit):
    """
    Sample a dataset with specified labels and limit, add benign samples for balancing.
    """
    selected = pd.concat(
        [dataset[dataset[Y_COLUMN] == label][:limit] for label in labels] +
        [benign_partition], ignore_index=True)
    return selected.sample(frac=1).reset_index(drop=True)


# Encode labels and prepare data for training
def encode_and_prepare(dataframes, encoder):
    """
    Encode labels and prepare datasets for training.
    """
    data_x, data_y = [], []
    for df in dataframes.values():
        df = df.sample(n=min(len(df))) 
        x, y = StandardScaler().fit_transform(df.drop(columns=[Y_COLUMN]).values), encoder.transform(df[Y_COLUMN].values)
        data_x.append(x)
        data_y.append(y)
    return data_x, data_y


# Prepare data for each category
def prepare_datasets(dataset):
    """Prepare individual datasets for each category."""
    benign_samples = dataset[dataset[Y_COLUMN] == 'BenignTraffic']
    
    category_limits = {
        'DDoS': 200000,
        'DoS': 75000,
        'Mirai': 410000,
        'Recon': 50000,
        'Spoofing': 150000,
        'Web': 5000,
        'BruteForce': 11000,
    }
    total  = sum(category_limits.values())

    category_datasets = {}
    offset = 0
    for i, (category, limit) in enumerate(category_limits.items()):
        start = offset
        end = offset + len(benign_samples) * total // limit
        category_datasets[category] = sample_and_shuffle(
            dataset, LABELS[category],
            benign_samples[start:end],
            limit
        )
        offset = end
    encoder = LabelEncoder()
    encoder.fit(sum(LABELS.values(), []))

    data_x, data_y = encode_and_prepare(category_datasets, encoder)

    return data_x, data_y


def shuffle_data(x, y):
    """
    Shuffle data arrays in unison.

    Args:
        x: Features array.
        y: Labels array.

    Returns:
        Shuffled features and labels.
    """
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]

def partition_iid(data_x, data_y, server_split, testing_split, clients_num):
    """
    Partition data IID for federated learning.

    Args:
        data_x: Concatenated features for all clients.
        data_y: Concatenated labels for all clients.
        server_split: Proportion of data for the server.
        testing_split: Proportion of data for testing.
        clients_num: Number of clients.

    Returns:
        Tuple containing server, training, and testing DataLoaders for IID data.
    """
    # Shuffle data
    data_x, data_y = shuffle_data(data_x, data_y)

    # Server data
    server_x = torch.tensor(data_x[:int(len(data_x) * server_split)], dtype=torch.float32)
    server_y = torch.tensor(data_y[:int(len(data_y) * server_split)], dtype=torch.long).reshape(-1, 1)
    server_data = DataLoader(TensorDataset(server_x, server_y), batch_size=BATCH_SIZE, shuffle=True)

    # Split remaining data into training and testing
    data_x, data_y = data_x[int(len(data_x) * server_split):], data_y[int(len(data_y) * server_split):]
    train_x = data_x[:int(len(data_x) * (1 - testing_split))]
    train_y = data_y[:int(len(data_y) * (1 - testing_split))]
    test_x = data_x[int(len(data_x) * (1 - testing_split)):]
    test_y = data_y[int(len(data_y) * (1 - testing_split)):]

    # Partition data among clients
    train_partition_size = len(train_x) // clients_num
    test_partition_size = len(test_x) // clients_num
    clients_data = []

    for i in range(clients_num):
        x_train = train_x[i * train_partition_size:(i + 1) * train_partition_size]
        y_train = train_y[i * train_partition_size:(i + 1) * train_partition_size]
        x_test = test_x[i * test_partition_size:(i + 1) * test_partition_size]
        y_test = test_y[i * test_partition_size:(i + 1) * test_partition_size]

        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                       torch.tensor(y_train, dtype=torch.long).reshape(-1, 1))
        test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                      torch.tensor(y_test, dtype=torch.long).reshape(-1, 1))

        clients_data.append([
            DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        ])

    return server_data, [client[0] for client in clients_data], [client[1] for client in clients_data]

def partition_noniid(data_x, data_y, server_split, testing_split, clients):
    """
    Partition data non-IID for federated learning.

    Args:
        data_x: List of features for each client.
        data_y: List of labels for each client.
        server_split: Proportion of data for the server.
        testing_split: Proportion of data for testing.
        clients: List of client identifiers.

    Returns:
        Tuple containing server, training, and testing DataLoaders for non-IID data.
    """
    server_x, server_y = [], []
    clients_data = {}

    for i, client in enumerate(clients):
        # Server data
        server_x_client = torch.tensor(data_x[i][:int(len(data_x[i]) * server_split)], dtype=torch.float32)
        server_y_client = torch.tensor(data_y[i][:int(len(data_y[i]) * server_split)], dtype=torch.long).reshape(-1, 1)
        server_x.append(server_x_client)
        server_y.append(server_y_client)

        # Split remaining data into training and testing
        train_x = data_x[i][int(len(data_x[i]) * server_split):int(len(data_x[i]) * (1 - testing_split))]
        train_y = data_y[i][int(len(data_y[i]) * server_split):int(len(data_y[i]) * (1 - testing_split))]
        test_x = data_x[i][int(len(data_x[i]) * (1 - testing_split)):]
        test_y = data_y[i][int(len(data_y[i]) * (1 - testing_split)):]

        train_dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32),
                                       torch.tensor(train_y, dtype=torch.long).reshape(-1, 1))
        test_dataset = TensorDataset(torch.tensor(test_x, dtype=torch.float32),
                                      torch.tensor(test_y, dtype=torch.long).reshape(-1, 1))

        clients_data[client] = [
            DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        ]

    # Combine server data and shuffle
    server_x = torch.cat(server_x, dim=0)
    server_y = torch.cat(server_y, dim=0)
    idx = torch.randperm(len(server_x))
    server_data = DataLoader(TensorDataset(server_x[idx], server_y[idx]), batch_size=BATCH_SIZE, shuffle=True)

    return server_data, [data[0] for data in clients_data.values()], [data[1] for data in clients_data.values()]



# Main workflow
def get_data():
    print("Loading dataset...")
    dataset = load_dataset()

    print("Preparing datasets and Encoding labels for each category...")
    data_x, data_y = prepare_datasets(dataset)
    clients = ["ddos", "dos", "mirai", "recon", "spoofing", "web", "bruteforce"]
    # IID Partitioning
    print("Performing IID Partitioning...")
    server_data_iid, training_sets_iid, testing_sets_iid = partition_iid(
        np.concatenate(data_x), np.concatenate(data_y),
        server_split=SERVER_SPLIT, testing_split=TESTING_SPLIT, clients_num=CLIENTS_NUM
    )

    # Non-IID Partitioning
    print("Performing Non-IID Partitioning...")
    server_data_noniid, training_sets_noniid, testing_sets_noniid = partition_noniid(
        data_x, data_y,
        server_split=SERVER_SPLIT, testing_split=TESTING_SPLIT, clients=clients
    )

    print("Data partitioning complete.")
    return server_data_iid, training_sets_iid, testing_sets_iid, server_data_noniid, training_sets_noniid, testing_sets_noniid

