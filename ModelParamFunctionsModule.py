import torch

def reinitialize_weights(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        try:
            module.bias.data.zero_()
        except AttributeError:
            pass
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def print_params_details(model):
    for name, param in model.named_parameters():
        print(f"{name}: mean={param.data.mean():.4f}; std={param.data.std():.4f}")