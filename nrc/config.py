import logging
import pathlib
import pickle
from functools import wraps
from pathlib import Path
from typing import Optional


def registry_property(property_name):
    """
    Initializer for properties to be autosaved in subfolders of the object root.
    These property objects assume the presence of a self.property_registry in the instance.
    The registry contains a dictionary {'name': object_access_list}, where object_access_list
    is a list with three elements:
    1. A function that retrieves the object from memory
    2. A function that stores an input in memory
    3. A function that checks whether there is data in memory.

    :param property_name: name of the attribute
    :return property: a property object with autosaving function.
    """
    def property_getter(p_name):  # property fget wrapper.
        def _property_getter(obj):
            if obj.root_path is None:  # if object is unrooted, just use object attributes.
                return getattr(obj, '__' + p_name)
            else:
                try:
                    return getattr(obj, '__' + p_name)  # if object has the attribute, just use that
                except AttributeError:  # if it hasn't, try to load it from memory.
                    try:
                        setattr(obj, '__' + p_name, obj.property_registry[p_name][0]())
                        return getattr(obj, '__' + p_name)
                    except FileNotFoundError:
                        raise AttributeError
        return _property_getter

    def property_setter(p_name):  # property fset wrapper
        def _property_setter(obj, value):
            if obj.root_path is None:  # if object is unrooted, just use object attributes
                setattr(obj, '__' + p_name, value)
            else:
                if obj.property_registry[p_name][2]():  # if there is data, warn the user
                    logging.info("Modifying existing file for property " + p_name)
                obj.property_registry[p_name][1](value)  # store data
                setattr(obj, '__' + p_name, obj.property_registry[p_name][0]())
        return _property_setter

    def property_deleter(p_name):  # property fdel wrapper. Deletion is not allowed.
        def _property_deleter(obj):
            if obj.root_path is None:
                delattr(obj, '__' + p_name)
            else:
                raise PermissionError("Cannot delete existing file")
        return _property_deleter

    return property(
        fget=property_getter(property_name),
        fset=property_setter(property_name),
        fdel=property_deleter(property_name)
    )


def autosave_method(method: callable) -> callable:
    """
    Decorator to wrap object methods whose result is computationally expensive and should be reloaded
    instead of recomputed when possible. The decorator assumes that the first argument of the method is
    the object with root.
    :param method:
        Method to wrap.
    :return autosaving_method:
        Method with an autosave wrapper. Autosave location is determined by object root and function arguments.
    """
    @wraps(method)
    def autosave(*args, **kwargs):
        obj = args[0]
        assert RootedObject in obj.__class__.__bases__
        root_path = obj.root_path
        # check if it is possible to autosave.
        if not kwargs.get('autosave', True):
            logging.info("Autosave disabled.")
            kwargs.pop('autosave')
            return method(*args, **kwargs)
        if root_path is None:  # if object is unrooted
            logging.info("Object is unrooted. Cannot save.")
            return method(*args, **kwargs)
        if len(args) > 1:  # if not all arguments are specified by name.
            logging.info("Each argument must be explicitly stated to have autosave. " +
                  "Result of " + str(method.__name__) + " not saved")
            return method(*args, **kwargs)
        else:
            target_string = '_'.join([key + '_' + str(kwargs[key]) for key in sorted(kwargs)])
            target = root_path / (method.__name__ + '_' + target_string + '.pkl')  # autosave target
            if target.exists():  # if computation already occurred.
                logging.info("Target already computed. Retrieved from " + str(target))
                return pickle.load(target.open('rb'))
            else:  # if computation did not occurr.
                logging.info("Target doesn't exist. Saving to " + str(target))
                res = method(*args, **kwargs)
                pickle.dump(res, target.open('wb'))
                return res
    return autosave


def pickle_dump_to_path(obj, path: pathlib.Path) -> None:
    """
    Pickle directly to path.

    :param obj: Object to pickle dump.
    :param path: path to pickle to
    :return: None
    """
    with path.open('wb') as f:
        pickle.dump(obj, f)


def pickle_load_from_path(path: pathlib.Path):
    """
    Load from path.

    :param path: path to pickle from.
    :return: pickled object.
    """
    with path.open('rb') as f:
        return pickle.load(f)


class RootedObject:
    def __init__(self, root_path: Optional[Path]):
        self.root_path = None
        try:
            self.root(root_path)
        except ValueError:
            pass

    def root(self, root_path: Path):
        if root_path is None:
            raise ValueError
        self.root_path = root_path
        self.root_path.mkdir(exist_ok=True, parents=True)

    def unroot(self):
        self.root_path = None
