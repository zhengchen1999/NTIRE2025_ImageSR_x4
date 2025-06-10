import copy


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def recursive_update(orig_dict, new_dict):
    """Recursively update the dictionary, replacing the values in orig_dict with new_dict."""
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in orig_dict and isinstance(orig_dict[key], dict):
            recursive_update(orig_dict[key], value)
        else:
            orig_dict[key] = value

def make(model_spec, args=None, load_sd=False):
    freeze_keys = None
    # print("model args:", model_spec['args'])
    if args is not None:
        if model_spec['args'] != args['args']:
            # freeze_keys = ['encoder']
            print("freeze_layers:", freeze_keys)

        model_args = copy.deepcopy(model_spec['args'])
        recursive_update(model_args, args['args'])

    else:
        model_args = model_spec['args']

    # print("Updated Model Spec:", model_spec['name'], model_args)

    model = models[model_spec['name']](**model_args)
    if load_sd:
        missing, unexpected = model.load_state_dict(model_spec['sd'], strict=False)
        # print("Missing keys:", missing)
        # print("Unexpected keys:", unexpected)

    if freeze_keys:
        for name, param in model.named_parameters():
            if any(name.startswith(freeze_key) for freeze_key in freeze_keys):
                param.requires_grad = False
                # print(f"Froze: {name}")
            else:
                print(f"Active: {name}")

    return model
