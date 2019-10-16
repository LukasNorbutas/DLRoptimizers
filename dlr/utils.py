from typing import *

def get_lr_multipliers(
    model: Any,
    lr: Union[Tuple[float], Tuple[float, float], Tuple[float, ...]],
    params: bool = False) -> dict:
    """
    This funcion takes an ImageLearner and a learning rate, and creates a dictionary, where each
    layer (or each layer's parameters if params == True) is mapped to an individual
    learning rate multiplier. This dictionary is then passed to DLR optimizer, which applies these
    layer/parameter learning rate multipliers during optimization (a.k.a. discriminative/differential
    learning rates).

    # Arguments:
        model: ImageLearner, where ImageLearner.model is the final model, and ImageLearner.base_model
            is the transfer learning type used in ImageLearner.model (e.g. keras.applications.Xception)
        lr: learning rates to be applied to layers/parameters. Supports a tuple of length 1
            for applying learning rates
    """

    # Dictionary that contains parts of layer names that can be safely used as splits
    # for defining layer groups
    architecture_slices = {
        "inception_v3": "mixed",
        "resnet50": "add",
        "efficientnet-b0": "expand_conv",
        "efficientnet-b0": "expand_conv",
        "xception": "add"
        }

    # FastAI kind of learning rate splits:
    # 1. If input lr is a tuple of length 1, apply no multiplier for top layers and
    # 0.3 multiplier for all base layer (ImageLearner.base_model)
    if len(lr) == 1:
        all_layers = [i.name for i in model.model.layers]
        top_layer = model.base_model.layers[-1].name
        idx_top = all_layers.index(top_layer)
        split_1 = {i: 0.3 for i in all_layers[1:idx_top]}
        split_2 = {i: 1 for i in all_layers[idx_top:]}
        split_1.update(split_2)
        lr_slices = split_1
        if params:
            return layer_to_param_dict(lr_slices, model)
        else:
            return lr_slices
    # 2. If lr's passed as double tuple, the top layers get LR * 1, the base layers
    # get divided into two groups: bottom group (first layers) gets a multiplier lr[0]/lr[1],
    # middle layers get the average of multipliers of top layers and bottom layers.
    elif len(lr) == 2:
        split_candidates = [x.name for x in model.model.layers
                            if architecture_slices[model.base_model.name] in x.name]
        split_layer = split_candidates[round(len(split_candidates)/2)]

        all_layers = [x.name for x in model.model.layers]
        top_layer = model.base_model.layers[-1].name

        idx_split = all_layers.index(split_layer)
        idx_top = all_layers.index(top_layer)

        split_1 = {i: lr[0]/lr[1] for i in all_layers[1:idx_split]}
        split_2 = {i: (1 + (lr[0] + lr[1])) / 2 for i in all_layers[idx_split:idx_top]}
        split_3 = {i: 1 for i in all_layers[idx_top:]}

        split_1.update(split_2)
        split_1.update(split_3)
        lr_slices = split_1
        if params:
            return layer_to_param_dict(lr_slices, model)
        else:
            return lr_slices


def layer_to_param_dict(
    lr_slices: dict,
    model: Any) -> dict:
    """
    This function takes a dictionary of model layers as keys and learning rate multipliers as values,
    the model which it was based on, and converts it to a dictionary of parameters and learning_rates,
    where each parameter's LR corresponds to its layer's LR in the input dictionary.
    """
    layers = {i.name: i for i in model.model.layers}
    parameters = {i.name: i.variables for i in layers.values() if len(i.variables) > 0}
    parameter_lr_rates = {}
    for layer, parameters in parameters.items():
        for param in parameters:
            parameter_lr_rates[param.name] = lr_slices[layer]
    return parameter_lr_rates
