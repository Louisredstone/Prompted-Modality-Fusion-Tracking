import logging
logger = logging.getLogger(__name__)

import os
import yaml

from enum import Enum, auto


class FieldStatus(Enum):
    Required = auto()
    Optional = auto()
    Deprecated = auto()
    Auto = auto()
    Reserved = auto()
    NotImplemented = auto()
yaml.add_representer(FieldStatus, lambda dumper, data: dumper.represent_none(None))
# skip all FieldStatus values when dumping to yaml

def stack_to_path(stack):
    return ''.join('[' + str(x) + ']' if isinstance(x, int) else '.' + str(x) for x in stack).strip('.')

class BetterEasyDict(dict):
    '''
    BetterEasyDict is a modified EasyDict.
    examples:
        d = BetterEasyDict({'Earth': {'Asia': {'China': {'Shanghai': 'City'}}}})
        # is equivalent to:
        d = BetterEasyDict(Earth=dict(Asia=dict(China=dict(Shanghai='City'))))
        d.Earth.Asia.China.Shanghai == 'City'
        d['Earth']['Asia']['China']['Shanghai'] == 'City'
        d.Earth.Asia.China['Shanghai'] == 'City'
        d['Earth', 'Asia', 'China', 'Shanghai'] == 'City'
    Unlike EasyDict, BetterEasyDict stores attrs and items seperately. When you set
        a value to a key, it will be stored as item, but you can still access it as
        attr. When you set an attr, it will be regarded as setting a value to the key,
        unless it's a special attr (with '__' prefix and '__' suffix).
    When initializing and setting values, it will automatically convert dict to 
        BetterEasyDict, even if it's deep in a list or tuple.
    It worth mention again that BE CAREFUL WITH SPECIAL KEYS/ATTRS (with '__' prefix
        and '__' suffix). 
    BetterEasyDict doesn't support Set well yet.
    '''
    def __init__(self, d=None, **kwargs):
        d = {} if d is None else dict(d)
        if kwargs: d.update(**kwargs)
        for k, v in d.items():
            self[k] = v

    def __is_name_special__(self, name):
        return name.startswith('__') and name.endswith('__')

    def __setattr__(self, name, value):
        if self.__is_name_special__(name):
            self.__dict__[name] = value
        else:
            self[name] = value
    
    def __getattr__(self, name, default=None):
        if self.__is_name_special__(name):
            if name not in self.__dict__:
                raise AttributeError(f"Attribute {name} not found.")
            if name == '__dict__': return self.__dict__
            if name == '__class__': return self.__class__
            return self.__dict__[name]
        else:
            return self[name] if name in self else default

    def __delattr__(self, name):
        if self.__is_name_special__(name):
            super(BetterEasyDict, self).__delattr__(name)
        else:
            del self[name]

    def __getitem__(self, name):
        if isinstance(name, (list, tuple)): # long path
            obj = self
            for k in name:
                obj = obj[k]
            return obj
        else:
            return super(BetterEasyDict, self).__getitem__(name)

    def __setitem__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = type(value)(self.__class__(x)
                     if isinstance(x, dict) else x for x in value)
        elif isinstance(value, dict) and not isinstance(value, BetterEasyDict):
            value = BetterEasyDict(value)
        super(BetterEasyDict, self).__setitem__(name, value)

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            self[k] = d[k]

    def pop(self, k, *args):
        return super(BetterEasyDict, self).pop(k, *args)

class Config(BetterEasyDict):
    '''
    Config is a subclass of BetterEasyDict. It provides some additional features:
    
    When initializing and setting values, it will automatically convert dict to 
        Config, even if it's deep in a list or tuple.
    CAUTION: If you set an existing key whose value is a Config with a dict, it
        be UPDATED recursively, not replaced. If you want to replace it, you 
        should set it to None first.
    It is lockable. A locked config means you may modify the value of a locked
        field, but you can't add or delete a field.
        This 'lock' doesn't mean multiprocessing lock.
    An empty config is not locked by default.
    Designing a specific config template is easy and recommended. Just 
        design a subclass Config and implement the init_fields() method.
        For a subclass of Config, once init_fields() is called, the config
            will be locked. This is for preventing accidental modification.
        The init_fields() method should set all fields with their default 
            values or their status (Required, Reserved, etc.).
        After that, pwd of all sub-configs will be initialized. 
            e.g., subconfig.path.to.field is a Config, then 
            subconfig.path.to.field.__pwd__ == ['path', 'to', 'field']
        Pwd also works for sub-configs in lists, where key will be the index.
    It can read and write config from/to yaml file.
        Before reading from file, all fields will be locked.
        When reading from file, those Required fields should be set in the file.
            Otherwise, an error will be raised.
        Vice versa, Reserved fields should not be set in the file. Otherwise, an
            error will be raised.
        When writing to file, fields with FieldStatus value will be skipped.
    You may implement recursive functions with Config.recursive() method easily.
    '''
    def __new__(cls, *args, **kwargs):
        instance = super(Config, cls).__new__(cls)
        instance.__dict__['__pwd__'] = []
        instance.__dict__['__locked__'] = False
        instance.__dict__['__reading_from_file__'] = False
        return instance
    
    def __init__(self, d=None, **kwargs):
        is_subclass = self.__class__ != Config # is subclass of Config, which means it is a specific config template
        if is_subclass:
            self.init_fields()
            self.lock()
        super(Config, self).__init__(d, **kwargs)
        if is_subclass: self.init_pwd()

    def init_fields(self):
        raise NotImplementedError("Subclass of Config must implement init_fields() method.")
        
    def init_pwd(self):
        '''initialize pwd for all sub-configs'''
        key_stack = []
        def func(value, key):
            if key is not None: key_stack.append(key)
            if isinstance(value, Config):
                value.__pwd__ = key_stack.copy()
        def post_func(res_item, res_children, key):
            if key is not None: key_stack.pop()
            return res_item, res_children
        self.recursive(func, post_func)
        
    def lock(self):
        def func(value, key):
            if isinstance(value, Config):
                value.__locked__ = True
        self.recursive(func)
    
    def unlock(self):
        def func(value, key):
            if isinstance(value, Config):
                value.__locked__ = False
        self.recursive(func)
    
    def recursive(self, func, post_func = None, key = None):
        '''
        func(value, key) -> res_item
        post_func(res_item, res_children, key) -> (res_item, res_children)
        '''
        def recursive_run(value, key, func, post_func=None):
            res_item = func(value, key)
            if isinstance(value, (list, tuple)):
                res_children = [recursive_run(value = v, key = i, func = func, post_func=post_func) for i, v in enumerate(value)]
            elif isinstance(value, dict):
                res_children = {k: recursive_run(value = v, key = k, func = func, post_func=post_func) for k, v in value.items()}
            else:
                res_children = None
            if post_func is not None:
                res_item, res_children = post_func(res_item, res_children, key)
            return res_item, res_children
        return recursive_run(value = self, key = key, func = func, post_func = post_func)
    
    def find_fields_with_status(self, field_status: FieldStatus):
        '''
        Find all fields with the given status.
        Args:
            field_status (FieldStatus): The status of the fields to find.
        Returns:
            list: A list of field paths.
        '''
        key_stack = []
        # key_stack: let func remember current path.
        #            e.g.: ['MODEL', 'BACKBONE', 'TYPE'] for self.MODEL.BACKBONE.TYPE
        #            e.g.: ['TRAIN', 'DATA', 'DATASETS', 0, 'NAME'] for self.TRAIN.DATA.DATASETS[0].NAME
        matched_fields = []
        def func(value, key):
            nonlocal matched_fields, key_stack
            if key is not None: key_stack.append(key)
            if value==field_status:
                matched_fields.append(key_stack.copy())
                return False # False for fail
            return True # True for pass
        def post_func(res_item, res_children, key):
            nonlocal key_stack
            if key is not None: key_stack.pop()
            item_passed = res_item
            if isinstance(res_children, dict):
                children_all_passed = all(res_child[0] for res_child in res_children.values())
            elif isinstance(res_children, (list, tuple)):
                children_all_passed = all(res_child[0] for res_child in res_children)
            elif res_children is None:
                children_all_passed = True
            else: 
                raise ValueError(f"Invalid res_children type: {type(res_children)}")
            if item_passed and children_all_passed: return True, None
            else: return False, None
        self.recursive(
            func = func,
            post_func = post_func,
        )
        return matched_fields
    
    def recursive_update(self, other: dict):
        if not isinstance(other, dict):
            raise TypeError(f"unsupported operand type(s) for +: 'Config' and '{type(other)}'")
        def smartcopy(lst: list):
            result = []
            for item in lst:
                if isinstance(item, (list, tuple)):
                    result.append(smartcopy(item))
                elif isinstance(item, dict):
                    result.append(Config(item))
                else:
                    result.append(item)
            return result
        key_stack = []
        def merge(ptr_self, ptr_other):
            if isinstance(ptr_self, dict) and isinstance(ptr_other, dict):
                for k in ptr_other:
                    if k not in ptr_self: # add new key
                        if isinstance(ptr_other[k], dict):
                            ptr_self[k] = Config(ptr_other[k])
                        elif isinstance(ptr_other[k], (list, tuple)):
                            ptr_self[k] = smartcopy(ptr_other[k])
                        else:
                            ptr_self[k] = ptr_other[k]
                    else: # k in ptr_self, update
                        if isinstance(ptr_self[k], dict) and isinstance(ptr_other[k], dict):
                            key_stack.append(k)
                            merge(ptr_self[k], ptr_other[k])
                            key_stack.pop()
                        elif isinstance(ptr_other[k], (list, tuple)):
                            ptr_self[k] = smartcopy(ptr_other[k])
                        else: # update existing key
                            ptr_self[k] = ptr_other[k]
            else:
                raise TypeError(f"Cannot merge {type(ptr_self)} and {type(ptr_other)}")
        merge(self, other)
        if self.__locked__: self.lock()
        else: self.unlock()
        self.init_pwd()
    
    def __setitem__(self, name, value):
        if self.__locked__ and name not in self:
            raise KeyError(f"Illegal key: {name}. Cannot add new key to a locked EasyDict. pwd: {stack_to_path(self.__pwd__)}")
        if self.__reading_from_file__ and name in self and self[name]==FieldStatus.Reserved:
            raise KeyError(f"Illegal key: {name}. Cannot set reserved key in a config file. pwd: {stack_to_path(self.__pwd__)}")
        if name in self and self[name]==FieldStatus.NotImplemented:
            raise NotImplementedError(f"Key {stack_to_path(self.__pwd__ + [name])} is not implemented yet.")
        if self.__locked__ and self[name] == FieldStatus.Deprecated:
            print(f"[WARN]: Key {stack_to_path(self.__pwd__ + [name])} is deprecated.")
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            if name in self and isinstance(self[name], Config):
                self[name].recursive_update(value)
                # Mention: if you need to set value directly, set self[name] to None first.
                return
        elif isinstance(value, (list, tuple)):
            value = type(value)(self.__class__(x)
                     if isinstance(x, dict) else x for x in value) # TODO: deep convert
        super(BetterEasyDict, self).__setitem__(name, value)
        # use super(EasyDict, self) here to call dict.__setitem__.

    def to_dict(self, enable_fieldstatus=False, enable_none=True) -> dict:
        # If 'enable_fieldstatus', return all fields including those with FieldStatus value. If not, skip them.
        # If 'enable_none', return None for all fields with None value. If not, skip them.
        def func(value, key):
            if not isinstance(value, (list, tuple, dict)):
                return value
            else: return None
        def post_func(res_item, res_children, key):
            if res_children is None: # current item is a leaf node
                return res_item, None
            elif isinstance(res_children, dict): # current item is a dict
                return {k: v[0] 
                        for k, v in res_children.items()
                        if (enable_fieldstatus or not isinstance(v[0], FieldStatus)) and (enable_none or v[0] is not None)}, None
            elif isinstance(res_children, (list, tuple)):
                return [v[0]
                        for v in res_children
                        if (enable_fieldstatus or not isinstance(v[0], FieldStatus)) and (enable_none or v[0] is not None)], None
            else:
                raise ValueError(f"Invalid res_children type: {type(res_children)}")
        result, _ = self.recursive(
            func = func,
            post_func = post_func,
        )
        return result

    def yaml(self, enable_fieldstatus=False, enable_none=True) -> str:
        return yaml.dump(
            self.to_dict(enable_fieldstatus=enable_fieldstatus, enable_none=enable_none), 
            default_flow_style=False)

    def __str__(self):
        return self.yaml()

    def save_as_yaml(self, filepath: str):
        with open(filepath, 'w') as file:
            file.write(self.yaml())

    @classmethod
    def from_file(cls, filepath: str, check_required_fields=True):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with open(filepath, 'r') as file:
            cfg_dict = yaml.safe_load(file)
        config: Config = cls(cfg_dict)
        config.__reading_from_file__ = True
        if check_required_fields:
            required_fields = config.find_fields_with_status(FieldStatus.Required)
            if required_fields:
                print("Required fields not found, should be set in config file: \n" + '\n'.join(stack_to_path(field) for field in required_fields))
                raise ValueError("Required fields not found")
        config.__reading_from_file__ = False
        return config
    
