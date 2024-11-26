from prettytable import PrettyTable

__all__ = ["count_parameters"]

def _scientific_notation(num):
    if num < 1e3:
        return str(num)
    elif num < 1e6:
        return f"{num/1e3:.2f}K"
    elif num < 1e9:
        return f"{num/1e6:.2f}M"
    else:
        return f"{num/1e9:.2f}B"

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, _scientific_notation(params)])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {_scientific_notation(total_params)}")
    return total_params
    