import torch
from torch.nn import Module
from torch.optim import Optimizer

def aggregate_models(model, clients_models: list, SCAFFOLD):
    """
    Aggregates client models into the global model using either simple averaging or SCAFFOLD method.

    Args:
        model (torch.nn.Module): The global model to be updated.
        clients_models (list): List of client models.
        SCAFFOLD (bool): Whether to apply SCAFFOLD aggregation.

    Returns:
        torch.nn.Module: The updated global model.
        list: The updated client models with the new global weights.
    """
    # Calculate the total dataset size across all clients
    total_data_size = sum(client.len for client in clients_models)

    if not SCAFFOLD:
        # Simple aggregation (e.g., FedAvg)
        aggregated_params = {name: torch.zeros_like(param.data) for name, param in model.named_parameters()}

        for client in clients_models:
            client_weight = client.len / total_data_size
            for name, param in client.named_parameters():
                aggregated_params[name] += param.data * client_weight

        for name, param in model.named_parameters():
            param.data = aggregated_params[name].clone()

    else:
        # SCAFFOLD aggregation
        delta_y = {name: torch.zeros_like(param.data) for name, param in model.named_parameters()}
        delta_c = {name: torch.zeros_like(param.data) for name, param in model.named_parameters()}

        for client in clients_models:
            client_weight = client.len / total_data_size
            for name, param in client.named_parameters():
                delta_y[name] += client.delta_y[name] * client_weight
                delta_c[name] += client.delta_control[name] * client_weight

        for name, param in model.named_parameters():
            param.data += delta_y[name].data
            model.control[name].data += delta_c[name].data

    # Update client models with the new global model
    return model, update_client_models(model, clients_models)


def update_client_models(model: Module, clients_models: list):
    """
    Update all client models with the weights from the global model.

    Args:
        model (torch.nn.Module): The global model.
        clients_models (list): List of client models.

    Returns:
        list: Updated client models with the global model weights.
    """
    for client_model in clients_models:
        for client_param, server_param in zip(client_model.parameters(), model.parameters()):
            client_param.data = server_param.data.clone()
    return clients_models


# -----------------------------------------------------------
# Scaffold Optimizer
# -----------------------------------------------------------
class ScaffoldOptimizer(Optimizer):
    """
    An optimizer for the Scaffold algorithm.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 penalty).
    """
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):
        """
        Perform a single optimization step.

        Args:
            server_controls (dict): Global model controls from the server.
            client_controls (dict): Local model controls from the client.
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss: Loss after the step (if closure is provided).
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p, c_server, c_client in zip(group['params'], server_controls.values(), client_controls.values()):
                if p.grad is None:
                    continue
                # Update parameter with gradient and control differences
                dp = p.grad.data + c_server.data - c_client.data
                p.data = p.data - dp * group['lr']

        return loss

