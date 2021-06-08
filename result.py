
LANGS = [ "ENG", "RUS" ]

WARNINGS = {
    "TOO_BIG_IMAGE_WAS_COMPRESSED": {
        LANGS[0]: "it is too big size of image, image was compressed",
        LANGS[1]: "слишком большой размер изображения, изображение было сжато"
    },
    "NEGATIVE_VALUE_MUST_BE_POSITIVE": {
        LANGS[0]: "this value must be positive",
        LANGS[1]: "это значение должно быть положительным"
    },
    "TOO_BIG_VALUE": {
        LANGS[0]: "it is too big value",
        LANGS[1]: "это слишком большое значение"
    }
}
    

    
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

