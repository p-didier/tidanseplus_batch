import ast
import yaml
import numpy as np
from dataclasses import fields, is_dataclass

def dump_to_yaml_template(myDataclass, path=None):
    """Dumps a YAML template for a dataclass.
    
    Parameters
    ----------
    myDataclass : instance of a dataclass
        The dataclass to dump a template for.
    path : str
        The path to the YAML file to be created.
        If not provided, the file will be created in the current directory.
    """

    if path is None:
        path = f'{type(myDataclass).__name__}__template.yaml'

    def _convert_to_dict(cls):
        """
        Recursively converts a dataclass (and its nested dataclasses)
        to a dictionary.
        Difference with `dcw.asdict` is that this function does 
        not alter the capitalization of the field names, and keeps
        underscores in the field names.
        """
        outDict = {}
        for key in fields(cls):
            if is_dataclass(getattr(cls, key.name)):
                outDict[key.name] = _convert_to_dict(getattr(cls, key.name))
            elif type(getattr(cls, key.name)) is np.ndarray:
                # Convert numpy arrays to lists, then strings
                # before dumping to YAML.
                outDict[key.name] = str(getattr(cls, key.name).tolist())
            elif type(getattr(cls, key.name)) is list:
                # Convert lists to strings before dumping to YAML
                outDict[key.name] = str(getattr(cls, key.name))
            else:
                outDict[key.name] = getattr(cls, key.name)
        return outDict

    with open(path, 'w') as f:
        myDataclassAsDict = _convert_to_dict(myDataclass)
        yaml.dump(myDataclassAsDict, f, default_flow_style=False)
    
    print(f'YAML template for dataclass "{type(myDataclass).__name__}" dumped to "{path}".')


def load_from_yaml(path, myDataclass):
    """Loads data from a YAML file into a dataclass.
    
    Parameters
    ----------
    path : str
        The path to the YAML file to be loaded.
    myDataclass : instance of a dataclass
        The dataclass to load the data into.

    Returns
    -------
    myDataclass : instance of a dataclass
        The dataclass with the data loaded into it.
    """

    with open(path, 'r') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    
    return load_from_dict(d, myDataclass)


def load_from_dict(d, myDataclass):
    """
    Loads data from a dictionary into a dataclass.
    
    Parameters
    ----------
    d : dict
        The dictionary to be loaded.
    myDataclass : instance of a dataclass
        The dataclass to load the data into.

    Returns
    -------
    myDataclass : instance of a dataclass
        The dataclass with the data loaded into it.
    """

    def _interpret_lists(d):
        """Interprets lists in the YAML file as lists of floats, not strings"""
        for key in d:
            if type(d[key]) is str and len(d[key]) >= 2:
                if d[key][0] == '[' and d[key][-1] == ']':
                    d[key] = ast.literal_eval(d[key])  # Convert string to list
                    # Use of `literal_eval` hinted at by https://stackoverflow.com/a/1894296
            elif type(d[key]) is dict:
                d[key] = _interpret_lists(d[key])
        return d

    # Detect lists
    d = _interpret_lists(d)

    def _deal_with_arrays(d):
        """Transforms lists that should be numpy arrays into numpy arrays"""
        for key in d:
            if type(d[key]) is list:
                if myDataclass.__annotations__[key] is np.ndarray:
                    d[key] = np.array(d[key])
            elif type(d[key]) is dict:
                d[key] = _deal_with_arrays(d[key])
        return d

    # Deal with expected numpy arrays
    d = _deal_with_arrays(d)

    def _load_into_dataclass(d, myDataclass):
        """Loads data from a dict into a dataclass"""
        for key in d:
            if type(d[key]) is dict:
                setattr(
                    myDataclass,
                    key,
                    _load_into_dataclass(d[key], getattr(myDataclass, key))
                )
            else:
                setattr(myDataclass, key, d[key])
        return myDataclass

    # myDataclass = dcw.fromdict(myDataclass, d)
    myDataclass = _load_into_dataclass(d, myDataclass)
    
    # If there is a __post_init__ method, call it
    if hasattr(myDataclass, '__post_init__'):
        myDataclass.__post_init__()

    return myDataclass
    """Loads data from a dictionary into a dataclass.
    
    Parameters
    ----------
    d : dict
        The dictionary to load the data from.
    myDataclass : instance of a dataclass
        The dataclass to load the data into.
    
    Returns
    -------
    myDataclass : instance of a dataclass
        The dataclass with the data loaded into it.
    """
    
    def _interpret_lists(d):
        """Interprets lists in the dictionary as lists of floats, not strings"""
        for key in d:
            if type(d[key]) is str and len(d[key]) >= 2:
                if d[key][0] == '[' and d[key][-1] == ']':
                    d[key] = ast.literal_eval(d[key])  # Convert string to list
                    # Use of `literal_eval` hinted at by https://stackoverflow.com/a/1894296
            elif type(d[key]) is dict:
                d[key] = _interpret_lists(d[key])
        return d

    # Detect lists
    d = _interpret_lists(d)

    def _deal_with_arrays(d):
        """Transforms lists that should be numpy arrays into numpy arrays"""
        for key in d:
            if type(d[key]) is list:
                if myDataclass.__annotations__[key] is np.ndarray:
                    d[key] = np.array(d[key])
            elif type(d[key]) is dict:
                d[key] = _deal_with_arrays(d[key])
        return d

    # Deal with expected numpy arrays
    d = _deal_with_arrays(d)

    def _load_into_dataclass(d, myDataclass):
        """Loads data from a dict into a dataclass"""
        for key in d:
            if type(d[key]) is dict:
                setattr(
                    myDataclass,
                    key,
                    _load_into_dataclass(d[key], getattr(myDataclass, key))
                )
            else:
                setattr(myDataclass, key, d[key])
        return myDataclass

    # myDataclass = dcw.fromdict(myDataclass, d)
    myDataclass = _load_into_dataclass(d, myDataclass)
    
    # If there is a __post_init__ method, call it
    if hasattr(myDataclass, '__post_init__'):
        myDataclass.__post_init__()

    return myDataclass