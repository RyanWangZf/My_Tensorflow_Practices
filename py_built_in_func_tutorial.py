# -*- coding: utf-8 -*-
# Python 3.5 on Linux
# Tutorials on python Built-in Functions
# Reference : https://blog.csdn.net/wangxin6722513/article/details/79498577


import pdb

class TestDict(object):

    def __getitem__(self,key):
        return self.__dict__.get(key)

    def __setitem__(self,key,value):
        self.__dict__[key] = value

    def __delitem__(self,key):
        self.__dict__.pop(key)

'''
if __name__ == "__main__":
    
    td = TestDict()
    td["a"] = 1
    td["b"] = 2
    print(td["a"])
    pdb.set_trace()
'''


class TestNew(object):
    def __new__(cls,*args,**kwargs):
        print("__new__ is called.")
        return super(TestNew,cls).__new__(cls,*args,**kwargs)

    def __init__(self):
        print("__init__ is called.")
        self.a = 1

'''
if __name__ == "__main__":
    tn = TestNew()
    print(tn.a)
    pdb.set_trace()
'''
class Singleton(object):
    def __new__(cls):
        # every time returns the same instance
        if not hasattr(cls,"instance"):
            cls.instance = super(Singleton,cls).__new__(cls)
        return cls.instance

'''
if __name__ == "__main__":
    obj1 = Singleton()
    obj2 = Singleton()
    obj1.attr1 = "value1"
    print(obj1.attr1,obj2.attr1)
    print(obj1 is obj2)
'''

class TestIterNext(object):
    def __init__(self,data=1):
        self.data = data

    def __next__(self):
        if self.data > 5:
            raise StopIteration
        else:
            self.data += 1
            return self.data
    def __iter__(self):
        print("iter")
        return self

'''
if __name__ == "__main__":
    ti =  TestIterNext()
    for i in ti:
        print(i)
'''

class TestCall(object):
    def __call__(self):
        print("class is called.")
'''
if __name__ == "__main__":
    tc = TestCall()
    tc()
'''


'''
# Test the __all__
if __name__ == "__main__":
    # need create another py file named module.py
    # codes in module.py:
    # __all__ = ["bar","baz"]
    # waz = 5; bar = 10;
    # def baz():
    #   return "baz"
    from module import *
    print(bar)
    print(baz)
    try:
        print(waz)
    except:
        print("cannot call 'waz' from module")
    pdb.set_trace()
'''

class TestAttr(object):
    def __init__(self):
        self.name = "abc"

    def __getattr__(self,item):
        print("item:",str(item))
        print("getattr")
        return 10

    def __setattr__(self,*args,**kwargs):
        print("set attr")
        object.__setattr__(self,*args,**kwargs)

    def __delattr__(self,*args,**kwargs):
        print("delete attr")
        object.__delattr__(self,*args,**kwargs)

'''
if __name__ == "__main__":
    ta = TestAttr()
    print(ta.__dict__)
    print(ta.name)
    del ta.name
    print(ta.__dict__)
    pdb.set_trace()
'''

class TestCompare(object):
    def __lt__(self,other):
        return "aaa"
    def __eq__(self,other):
        return "bbb"
'''
if __name__ == "__main__":
    t = TestCompare()
    print(t<1)
    print(t==1)
'''

class TestSlots(object):
    # __slots__
    # 1. faster call on attributes of class
    # 2. save RAM consume
    __slots__ = ["name","age"]
    def __init__(self,name,age):
        self.name = name
        self.age = age
'''    
if __name__ == "__main__":
    ts = TestSlots("a",1)
    print(ts.name,ts.age)
    ts.name = 1
    print(ts.name)
'''

if __name__ == "__main__":
    # explore on class and metaclass of the Python
    # create a class by function type
    foo = type("Foo",(),{"bar":True,"val":1}) # var: name, bases, dict
    # create a class with inheritance of class foo, add a new method echo_bar 
    foochild = type("FooChild",(foo,),{"echo_bar":lambda self: print(self.bar)})
    print(foo.val,foo.bar)
    subfoo =  foochild()
    subfoo.echo_bar()
    print(hasattr(foo,"echo_bar"))
    print(hasattr(subfoo,"echo_bar"))

    func = lambda x:x
    cls = type("foo",(),{})
    c = cls()
    print(func.__class__)
    print(c.__class__)
    print(c.__class__.__class__)
