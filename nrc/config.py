import pickle

def registry_property(property_name):
    def property_getter(p_name):
        def _property_getter(obj):
            if obj.root is None:
                return getattr(obj, '__' + p_name)
            else:
                try:
                    return getattr(obj, '__' + p_name)
                except AttributeError:
                    try:
                        setattr(obj, '__' + p_name, obj.property_registry[p_name][0]())
                        return getattr(obj, '__' + p_name)
                    except FileNotFoundError:
                        raise AttributeError
        return _property_getter

    def property_setter(p_name):
        def _property_setter(obj, value):
            if obj.root is None:
                setattr(obj, '__' + p_name, value)
            else:
                if obj.property_registry[p_name][2]():
                    print("Modifying existing file for property " + p_name)
                obj.property_registry[p_name][1](value)
                setattr(obj, '__' + p_name, obj.property_registry[p_name][0]())
        return _property_setter

    def property_deleter(p_name):
        def _property_deleter(obj):
            if obj.root is None:
                delattr(obj, '__' + p_name)
            else:
                raise PermissionError("Cannot delete existing file")
        return _property_deleter

    return property(
        fget=property_getter(property_name),
        fset=property_setter(property_name),
        fdel=property_deleter(property_name)
    )
