
import os, sys
import json
import yaml
from datetime import datetime, timedelta
import logging
import re

# Python 3.6 以前普通字典是无序的，需要使用 collections.OrderedDict
if sys.version < "3.6":
    print("Python version 3.6 or above is required.")

class OO(dict):
    __id_count = 0
    def __init__(self, id:str=None, default:dict=None, **key_value_pairs):
        if id is None:
            OO.__id_count += 1
            id = "__OO_%04d"%(OO.__id_count)
        self.__id = id
        self.update(default, **key_value_pairs)
        
    def update(self, override:dict=None, **key_value_pairs):
        if isinstance(override, OO):
            self.__update(override)
        elif override is not None:
            self.__update(self.__parse(self.__id, override, self, self))
        if len(key_value_pairs)>0:
            self.__update(self.__parse(self.__id, key_value_pairs, self, self))
        return self
    
    def __update(self, ndic):
        if isinstance(ndic, OO):
            ndic = ndic.__dict__
        keepid = self.__id
        for k in ndic:
            if re.match(r"_\w+__\w+", k) is None:
                if isinstance(ndic[k], OO):
                    if k in self.__dict__ and isinstance(self.__dict__[k], OO):
                        self.__dict__[k].update(ndic[k])
                    else:
                        self.__dict__[k] = ndic[k]
                    self[k] = self.__dict__[k]
                    ndic[k].__setid(keepid+"."+k)
                else:
                    self.__dict__[k] = ndic[k]
                    self[k] = self.__dict__[k]
        return self

    def __setid(self, id):
        if self.__id != id:
            cid = id+"."
            for k in self.__dict__:
                if re.match(r"_\w+__\w+", k) is None:
                    if isinstance(self.__dict__[k], OO) and self.__dict__[k].__id[:len(cid)] != cid:
                        self.__dict__[k].__setid(id+"."+k)
            self.__id = id
    
    def __parse(self, name, value, ref_dict:dict, inherit_obj:None):
        if value is None or isinstance(value, OO):
            return value
        if isinstance(ref_dict, OO):
            ref_dict = ref_dict.__dict__
        if isinstance(value, dict):
            nd = {}
            if isinstance(inherit_obj, dict): nd.update(inherit_obj)
            for k in value:
                ref = {}
                ref.update(os=os)
                ref.update(value)
                ref.update(ref_dict)
                ref.update(nd)
                v = self.__parse(name+"."+k, value[k], ref, nd[k] if k in nd else None)
                if isinstance(v, dict) and not isinstance(v, OO):
                    v = self.__parse(name+"."+k, v, ref, nd[k] if k in nd else None)
                if k in nd and isinstance(v, OO):
                    if isinstance(nd[k], OO):
                        nd[k].update(v)
                    elif isinstance(nd[k], dict):
                        nd[k].update(v.__dict__)
                    else:
                        nd[k] = v
                else:
                    nd[k] = v
            return OO(name).__update(nd)
        if isinstance(value, list):
            oo = []
            i = 0
            for v in value:
                oo.append(self.__parse("%s.%d"%(name,i), v, ref_dict, None))
                i += 1
            return oo
        if isinstance(value, tuple):
            oo = []
            i = 0
            for v in value:
                oo.append(self.__parse("%s.%d"%(name,i), v, ref_dict, None))
                i += 1
            return tuple(oo)
        if isinstance(value, set):
            oo = []
            i = 0
            for v in value:
                oo.append(self.__parse("%s.%d"%(name,i), v, ref_dict, None))
                i += 1
            return set(oo)
        if isinstance(value, str):
            if re.match(r'^\$\{[\.\w]*\}$', value) is not None:
                try:
                    return eval(value[2:-1], ref_dict)
                except Exception as e:
                    if not self.__id.startswith("__OO_"): print("OO_Parse[%s]: "%name, e)
                    pass
            elif re.search(r'\$\{.*\}', value) is not None:
                try:
                    return re.sub(r'\$\{(?P<v>[^\}]+)\}', lambda m:  str(eval(m.group('v'), ref_dict)), value, flags=re.M)
                except Exception as e:
                    if not self.__id.startswith("__OO_"): print("OO_Parse[%s]: "%name, e)
                    pass
            try:
                return int(value)
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                pass
            try:
                return complex(value)
            except ValueError:
                pass
        return value

class Config(OO):
    def __init__(self, cfgfile:str=None, default:[dict,OO]=None, override:[dict,OO]=None, **key_value_pairs):
        super(Config, self).__init__(re.sub(r"(.+)\..*$", lambda m: m.group(1), os.path.basename(os.path.realpath(cfgfile))))  # name可能会被配置信息改变
        self.update(message="")
        self.__config_file__ = cfgfile
        self.__default = OO("__default", default, **key_value_pairs)
        self.__override = OO("__override", override)
        self.reload()
        
    def reload(self):
        if self.__default is not None:
            self.update(self.__default)
        if self.__config_file__ is not None and os.access(self.__config_file__, os.F_OK):
            with open(self.__config_file__, encoding='UTF-8') as yamlfile:
                info = yaml.full_load(yamlfile)
                self.update(info)
        else:
            self.update(message=OO(CFNF="Config File '"+self.__config_file__+"' Not Found"))
        self.update(self.__override)
    
    def override(self, override:[dict,OO]=None, **key_value_pairs):
        self.__override.update(override, **key_value_pairs)
        self.update(self.__override)

    def __repr__(self):
        d = {}
        d.update(self)
        for k in list(d.keys()):
            if k.startswith("__config_"):
                del d[k]
        return str(d)

class Tracer():
    TRACE="EWID"
    FILE:str=None
    logger:logging.Logger=None
    __setting__=OO(FILE=None, FileHandler=None)
    @staticmethod
    def trace(*info):
        if Tracer.logger is None:
            Tracer.logger = logging.getLogger()
            Tracer.logger.setLevel(logging.INFO)
            console = logging.StreamHandler()
            Tracer.logger.addHandler(console)
        if Tracer.FILE != Tracer.__setting__.FILE:
            if Tracer.__setting__.FileHandler is not None:
                Tracer.logger.removeHandler(Tracer.__setting__.FileHandler)
                Tracer.__setting__.FileHandler = None
            if Tracer.FILE is not None and len(Tracer.FILE) > 0:
                handler = logging.FileHandler(Tracer.FILE, mode='a+')
                Tracer.logger.addHandler(handler)
                Tracer.__setting__.FileHandler = handler
            Tracer.__setting__.FILE = Tracer.FILE
        info = list(info) if info is not None and isinstance(info, tuple) else [info]
        if len(info)<2 or not isinstance(info[-1], str) or len(info[-1]) != 1:
            flag = "I"
        else:
            flag = info[-1]
            info = info[:-1]
        s = ""
        for i in range(len(info)):
            s += " " + str(info[i])
        if flag in Tracer.TRACE:
            dt = datetime.now()
            s = "%s.%03d [%s]%s"%(dt.strftime("%Y-%m-%d %H:%M:%S"), dt.microsecond/1000, flag, s)
            Tracer.logger.info(s)
            
