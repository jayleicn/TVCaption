import json


def save_json_pretty(data, file_path):
    """save formatted json, use this one for some json config files"""
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)


def save_parsed_args_to_json(parsed_args, file_path, pretty=True):
    args_dict = vars(parsed_args)
    if pretty:
        save_json_pretty(args_dict, file_path)
    else:
        save_json(args_dict, file_path)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def set_lr(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * decay_factor


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def count_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable


def sum_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    p_sum = sum(p.sum().item() for p in model.parameters())
    if verbose:
        print("Parameter sum {}".format(p_sum))
    return p_sum


def merge_dicts(list_dicts):
    merged_dict = list_dicts[0].copy()
    for i in range(1, len(list_dicts)):
        merged_dict.update(list_dicts[i])
    return merged_dict


def merge_json_files(paths, merged_path):
    merged_dict = merge_dicts([load_json(e) for e in paths])
    save_json(merged_dict, merged_path)


