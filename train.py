from preprocess import get_data, iid_clients, noniid_clients
from models import ANN
from federated_utils import aggregate_models, update_client_models, ScaffoldOptimizer
from utils import loss_classifier, test_step, local_learning
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def train_federated(client_names, model, training_sets: list, n_iter: int, testing_sets: list, server_data: list, SCAFFOLD: bool, mu=0):
    """
    Federated learning process with optional SCAFFOLD implementation.

    Args:
        client_names (list): List of client names.
        model (torch.nn.Module): Global server model.
        training_sets (list): List of training DataLoaders for each client.
        n_iter (int): Number of global iterations (communication rounds).
        testing_sets (list): List of testing DataLoaders for each client.
        server_data (DataLoader): Server DataLoader for global model evaluation.
        SCAFFOLD (bool): Whether to apply SCAFFOLD updates.
        mu (float): Regularization parameter for FedProx.

    Returns:
        model (torch.nn.Module): Updated global server model.
        clients_models (list): List of updated client models.
        loss_hist (list): History of server model's loss during training.
        acc_hist (list): History of server model's accuracy during training.
    """
    # Define the loss function
    loss_f = loss_classifier if not SCAFFOLD else nn.MSELoss().to(device)

    # Initialize client-related variables
    clientsNum = len(client_names)
    n_samples = sum([len(db.dataset) for db in training_sets])  # Total training samples
    weights = [1 / clientsNum for _ in range(clientsNum)]  # Uniform weights for clients

    # Initialize control variates and delta values for SCAFFOLD (if enabled)
    clients_models = []
    for name, param in model.named_parameters():
        model.control[name] = torch.zeros_like(param.data)  # Initialize control variates
        model.delta_control[name] = torch.zeros_like(param.data)  # Initialize control deltas
        model.delta_y[name] = torch.zeros_like(param.data)  # Initialize delta_y

    for i in range(clientsNum):
        temp_model = deepcopy(model)
        temp_model.name = client_names[i]
        temp_model.control = deepcopy(model.control)
        temp_model.delta_control = deepcopy(model.delta_control)
        temp_model.delta_y = deepcopy(model.delta_y)
        clients_models.append(temp_model)

    # Log initial setup
    print(f"No. of clients: {clientsNum}, Weights: {weights}")
    print(f"Server testing samples: {len(server_data.dataset)}, No. of labels: {len(server_data.dataset.tensors[1].unique())}")
    for i in range(clientsNum):
        print(f"Client {i}\t Training samples: {len(training_sets[i].dataset)}, Testing samples: {len(testing_sets[i].dataset)}, No. of labels: {len(training_sets[i].dataset.tensors[1].unique())}")

    # Initialize performance history
    loss_hist, acc_hist = [], []

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) if not SCAFFOLD else ScaffoldOptimizer(model.parameters(), lr=lr, weight_decay=1e-4)

    # Global communication rounds
    for round_num in range(n_iter):
        print("\n\n====================================================================================================")
        print(f"Round: {round_num + 1}")

        # Train each client locally
        for client_idx in range(clientsNum):
            print(f"\n{client_idx}: Client {client_idx} Starts, Training size: {len(training_sets[client_idx].dataset)}, Testing size: {len(testing_sets[client_idx].dataset)}")
            clients_models[client_idx] = local_learning(
                clients_models[client_idx], model, mu, training_sets[client_idx], testing_sets[client_idx], epochs, loss_f, optimizer, SCAFFOLD
            )

        # Aggregate the models at the server
        model, clients_models = aggregate_models(deepcopy(model), clients_models, weights, SCAFFOLD)

        # Evaluate the global model on the server data
        server_loss, server_acc, server_mae, server_rmse, correct = test_step(model, server_data, loss_f)

        # Update performance history
        loss_hist.append(server_loss)
        acc_hist.append(server_acc)

        print(f'\n Server Testing: Loss: {server_loss:.4f}, Accuracy: {server_acc:.2f}%, Correct predictions: {correct}/{len(server_data.dataset)}, MAE: {server_mae:.4f}, RMSE: {server_rmse:.4f}')

        # Save the global server model for this round
        torch.save(model.state_dict(), f"./models/server_model_{round_num}.pth")

    return model, loss_hist, acc_hist


server_data_iid, training_sets_iid, testing_sets_iid, server_data_noniid, training_sets_noniid, testing_sets_noniid = get_data()
# Parameters
n_iter = 10
epochs = 20
lr = 0.01
decay = 0.8
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Results storage
results = {}

# ========================
# FedAvg Testing
# ========================
SCAFFOLD = False
mu = 0  # Not used in FedAvg

# FedAvg with IID data
print("\n==== FedAvg with IID Data ====")
model = ANN(num_classes=34, input_size=46).to(device)
fedavg_model_iid, fedavg_loss_hist_iid, fedavg_acc_hist_iid = train_federated(
    iid_clients, model, training_sets_iid, n_iter, testing_sets_iid, server_data_iid, SCAFFOLD, mu=mu
)

# Save results
results["fedavg_iid"] = {
    "loss_hist": fedavg_loss_hist_iid,
    "acc_hist": fedavg_acc_hist_iid
}
with open(f"{output_dir}/fedavg_iid.json", "w") as outfile:
    json.dump(results["fedavg_iid"], outfile)

# FedAvg with Non-IID data
print("\n==== FedAvg with Non-IID Data ====")
model = ANN(num_classes=34, input_size=46).to(device)
fedavg_model_noniid, fedavg_loss_hist_noniid, fedavg_acc_hist_noniid = train_federated(
    noniid_clients, model, training_sets_noniid, n_iter, testing_sets_noniid, server_data_noniid, SCAFFOLD, mu=mu
)

# Save results
results["fedavg_noniid"] = {
    "loss_hist": fedavg_loss_hist_noniid,
    "acc_hist": fedavg_acc_hist_noniid
}
with open(f"{output_dir}/fedavg_noniid.json", "w") as outfile:
    json.dump(results["fedavg_noniid"], outfile)

# ========================
# FedProx Testing
# ========================
SCAFFOLD = False

for mu_val in [0.01, 0.04, 0.1, 0.2, 0.3]:
    print(f"\n==== FedProx with IID Data (mu={mu_val}) ====")
    model = ANN(num_classes=34, input_size=46).to(device)
    fedprox_model_iid, fedprox_loss_hist_iid, fedprox_acc_hist_iid = train_federated(
        iid_clients, model, training_sets_iid, n_iter, testing_sets_iid, server_data_iid, SCAFFOLD, mu=mu_val
    )

    # Save results
    results[f"fedprox_iid_mu_{mu_val}"] = {
        "loss_hist": fedprox_loss_hist_iid,
        "acc_hist": fedprox_acc_hist_iid
    }
    with open(f"{output_dir}/fedprox_iid_mu_{mu_val}.json", "w") as outfile:
        json.dump(results[f"fedprox_iid_mu_{mu_val}"], outfile)

    print(f"\n==== FedProx with Non-IID Data (mu={mu_val}) ====")
    model = ANN(num_classes=34, input_size=46).to(device)
    fedprox_model_noniid, fedprox_loss_hist_noniid, fedprox_acc_hist_noniid = train_federated(
        noniid_clients, model, training_sets_noniid, n_iter, testing_sets_noniid, server_data_noniid, SCAFFOLD, mu=mu_val
    )

    # Save results
    results[f"fedprox_noniid_mu_{mu_val}"] = {
        "loss_hist": fedprox_loss_hist_noniid,
        "acc_hist": fedprox_acc_hist_noniid
    }
    with open(f"{output_dir}/fedprox_noniid_mu_{mu_val}.json", "w") as outfile:
        json.dump(results[f"fedprox_noniid_mu_{mu_val}"], outfile)

# ========================
# SCAFFOLD Testing
# ========================
SCAFFOLD = True
mu = 0  # Not applicable in SCAFFOLD

# SCAFFOLD with IID data
print("\n==== SCAFFOLD with IID Data ====")
model = ANN(num_classes=34, input_size=46).to(device)
scaffold_model_iid, scaffold_loss_hist_iid, scaffold_acc_hist_iid = train_federated(
    iid_clients, model, training_sets_iid, n_iter, testing_sets_iid, server_data_iid, SCAFFOLD, mu=mu
)

# Save results
results["scaffold_iid"] = {
    "loss_hist": scaffold_loss_hist_iid,
    "acc_hist": scaffold_acc_hist_iid
}
with open(f"{output_dir}/scaffold_iid.json", "w") as outfile:
    json.dump(results["scaffold_iid"], outfile)

# SCAFFOLD with Non-IID data
print("\n==== SCAFFOLD with Non-IID Data ====")
model = ANN(num_classes=34, input_size=46).to(device)
scaffold_model_noniid, scaffold_loss_hist_noniid, scaffold_acc_hist_noniid = train_federated(
    noniid_clients, model, training_sets_noniid, n_iter, testing_sets_noniid, server_data_noniid, SCAFFOLD, mu=mu
)

# Save results
results["scaffold_noniid"] = {
    "loss_hist": scaffold_loss_hist_noniid,
    "acc_hist": scaffold_acc_hist_noniid
}
with open(f"{output_dir}/scaffold_noniid.json", "w") as outfile:
    json.dump(results["scaffold_noniid"], outfile)

print("\n=== All experiments completed. Results saved. ===")