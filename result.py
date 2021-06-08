    
class Result(object):
    __value = None
    __warnings = []
    __is_avaliable = False

    def __init__(self, is_avaliable: bool, value=None, warnings=[]):
        self.__value = value
        self.__is_avaliable = is_avaliable
        if isinstance(warnings, list):
            self.__warnings = self.__warnings + warnings # several warnings
        else:
            self.__warnings.append(warnings) # one warning

    def val(self):
        if self.__is_avaliable == False:
            raise RuntimeError("getting value from failure object")
        return self.__value

    def val_or_none(self):
        if self.__is_avaliable == False:
            return None
        return self.__value

    def warnings(self) -> list:
        return self.__warnings

    def is_avaliable(self) -> bool:
        return self.__is_avaliable

