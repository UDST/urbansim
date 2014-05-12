"""
Utilities for doing IO to YAML files.

"""
import yaml


def ordered_yaml(cfg):
    """
    Convert a dictionary to a YAML string with preferential ordering
    for some keys. Converted string is meant to be fairly human readable.

    Parameters
    ----------
    cfg : dict
        Dictionary to convert to a YAML string.

    Returns
    -------
    str
        Nicely formatted YAML string.

    """
    order = ['name', 'model_type', 'fit_filters', 'predict_filters',
             'choosers_fit_filters', 'choosers_predict_filters',
             'alts_fit_filters', 'alts_predict_filters',
             'interaction_predict_filters',
             'choice_column', 'sample_size', 'estimation_sample_size',
             'simple_relocation_rate',
             'patsy', 'dep_var', 'dep_var_transform', 'model_expression',
             'ytransform']

    s = ''
    for key in order:
        if key not in cfg:
            continue
        s += yaml.dump({key: cfg[key]}, default_flow_style=False, indent=4)

    for key in cfg:
        if key in order:
            continue
        s += yaml.dump({key: cfg[key]}, default_flow_style=False, indent=4)

    return s


def make_model_expression(cfg):
    """
    Turn the parameters into the string expected by patsy.

    Parameters
    ----------
    cfg : A dictionary of key-value pairs.  'patsy' defines
        patsy variables, 'dep_var' is the dependent variable,
        and 'dep_var_transform' is the transformation of the
        dependent variable.

    Returns
    -------
    Modifies the dictionary of params in place
    """
    if "patsy" not in cfg:
        return
    if "dep_var" not in cfg:
        return
    if "dep_var_transform" not in cfg:
        return
    patsy_exp = cfg['patsy']
    if type(patsy_exp) == list:
        patsy_exp = ' + '.join(cfg['patsy'])
    # deal with missing dep_var_transform
    patsy_exp = '%s(%s) ~ ' % (
        cfg['dep_var_transform'], cfg['dep_var']) + patsy_exp
    cfg['model_expression'] = patsy_exp
