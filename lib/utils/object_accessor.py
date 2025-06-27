import logging
logger = logging.getLogger(__name__)
from typing import Any, Union, Optional
from numpy import ndarray
from math import floor, ceil

class ObjectAccessor:
    '''A small utility class to access attributes of an object using indices.
    Recommended usage:
    class MyClass:
        def __getitem__(self, index):
            return ObjectAccessor(self, index)
        @property
        def data(self):
            return self._data # ndarray or other data structure
    
    obj = MyClass()
    accessor = obj[1:3, 4:6]
    print(accessor.data)
    # equivalent to:
    print(obj.data[1:3, 4:6])
    # equivalent to:
    print(obj[1:3, 4:6].data)
    '''
    def __init__(self, obj: object, indices: Any =slice(None)):
        self.__object__ = obj
        self.__indices__ = indices # slice(None) for 'all'
    
    def __len__(self):
        if self.__indices__ is None:
            return len(self.__object__)
        elif isinstance(self.__indices__, slice):
            _, _, _, _, length = self.__parse_slice__(self.__indices__)
            return length
        else:
            return len(self.__indices__)
    
    def __getattr__(self, name):
        if name in self.__dict__ and name in self.__object__.__dict__:
            logger.warning(f"Attribute {name} is both in the object and the accessor, returning the accessor's attribute.")
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.__object__, name)[self.__indices__]

    def __getitem__(self, index) -> 'ObjectAccessor':
        if index is None: return self
        new_indices = self.__merge_indices__(self.__indices__, index)
        return ObjectAccessor(self.__object__, new_indices)
    
    def __setitem__(self, index, value):
        raise NotImplementedError("Setting values is not supported yet.")
    
    def __merge_indices__(self, index1: Union[int, slice, list, tuple, ndarray], index2: Union[int, slice, list, tuple, ndarray, None]) -> Union[int, slice, list, tuple, ndarray]:
        # merge two indices into one, like:
        # "100:200", "3" -> "103"
        # "100:200", "3:5" -> "103:105"
        # "100:200", "3:10:2" -> "103:110:2"
        # "[101, 103, 107, 109]", "2" -> "107"
        # Rule: object[index1][index2] == object[merge_indices(index1, index2)]
        if index2 is None: return index1
        if isinstance(index1, ndarray) or isinstance(index2, ndarray):
            raise NotImplementedError("Cannot merge indices of type `numpy.ndarray` yet.")
        if isinstance(index1, int):
            if isinstance(index2, (int, slice, list)):
                return (index1, index2)
            elif isinstance(index2, tuple):
                return (index1, *index2)
            else: raise TypeError(f"Cannot merge indices of type `{type(index1)}` with type `{type(index2)}`.")
        elif isinstance(index1, slice):
            length_obj: int = len(self.__object__)
            start1, stop1, step1, tail1, length1 = self.__parse_slice__(index1, length_obj)
            # start1, stop1, step1 = index1.start, index1.stop, (index1.step or 1)
            # start1: int = start1 or 1 if step1 > 0 else -1
            # if start1 < 0: start1 += length_obj
            # tail1: int|None = ((length_obj - (length_obj-start1)%step1) if (stop1 is None 
            #                                                                 or stop1 > length_obj)\
            #                         else None if stop1<=start1\
            #                         else stop1 - 1) if step1 > 0\
            #                     else (start1%step1 if (stop1 is None or stop1<0)\
            #                         else None if stop1>=start1\
            #                         else stop1 + 1) # step1 < 0
            if tail1 is None: raise ValueError("Index1 is void indices, cannot merge with any other indices.")
            # length1: int = (tail1 - start1) // step1 + 1
            
            if isinstance(index2, int):
                if not -length1 <= index2 < length1:
                    raise IndexError(f"Index {index2} is out of bounds with size {length1}.")
                return start1 + index2 * step1 if index2 >= 0 else tail1 + (index2 + 1) * step1
            elif isinstance(index2, slice):
                start2, stop2, step2, tail2, length2 = self.__parse_slice__(index2, length1)
                # start2, stop2, step2 = index2.start, index2.stop, (index2.step or 1)
                # start2: int = start2 or 1 if step2 > 0 else -1
                # if start2 < 0: start2 += length1
                # tail2: int|None = ((length1 - (length1-start2)%step2) if (stop2 is None 
                #                                                           or stop2 > length1)\
                #                         else None if stop2<=start2\
                #                         else stop2 - 1) if step2 > 0\
                #                     else (start2%step2 if (stop2 is None or stop2<0)\
                #                         else None if stop2>=start2\
                #                         else stop2 + 1) # step2 < 0
                if tail2 is None: return slice(0, 0, 1)
                new_start = start1 + start2 * step1
                new_tail = start1 + tail2 * step1
                new_step = step1 * step2
                new_stop = new_tail + new_step
                return slice(new_start, new_stop, new_step)
            elif isinstance(index2, list):
                assert all(isinstance(i, int) for i in index2), "List indices must be integers."
                assert all(-length1 <= i < length1 for i in index2), "List indices out of bounds."
                return [start1 + i * step1 if i >= 0 else tail1 + (i + 1) * step1 for i in index2]
            elif isinstance(index2, tuple):
                return (self.__merge_indices__(index1, index2[0]), *index2[1:])
            else:
                raise TypeError(f"Cannot merge indices of type `{type(index1)}` with type `{type(index2)}`.")
        elif isinstance(index1, list):
            pass
        elif isinstance(index1, tuple):
            if not all(isinstance(i, int) for i in index1):
                raise NotImplementedError("Cannot merge indices of type `tuple` with non-integer elements yet.")
            if isinstance(index2, (int, slice, list)):
                return (*index1, index2)
            elif isinstance(index2, tuple):
                return (*index1, *index2)
            else: raise TypeError(f"Cannot merge indices of type `{type(index1)}` with type `{type(index2)}`.")
        else:
            raise TypeError(f"Cannot merge indices of type `{type(index1)}` with type `{type(index2)}`.")
        
    def __parse_slice__(self, index: slice, length: Optional[int] = None) -> tuple[int, int, int]:
        if length is None: length = len(self.__object__)
        start, stop, step = index.start, index.stop, (index.step or 1)
        start: int = start or 1 if step > 0 else -1
        if start < 0: start += length
        tail: int|None = ((length - 1 - (length - 1 - start)%step) if (stop is None 
                                                                        or stop > length)\
                                else None if stop<=start\
                                else stop - 1) if step > 0\
                            else (start%step if (stop is None or stop<0)\
                                else None if stop>=start\
                                else stop + 1) # step1 < 0
        if tail is None: 
            stop = start
            index_length = None
        else: 
            stop = tail + step
            index_length: int = (tail - start) // step + 1
        return start, stop, step, tail, index_length
    
    def __iter__(self):
        indices = self.__indices__
        if indices is None: return iter(self.__object__)
        elif isinstance(indices, slice):
            start, stop, step, tail, length = self.__parse_slice__(indices)
            if length is None: return []
            for i in range(start, stop, step):
                yield self.__object__[i]
        elif isinstance(indices, list):
            for i in indices:
                yield self.__object__[i]
        else:
            raise NotImplementedError("Iteration over non-slice indices is not supported yet.")