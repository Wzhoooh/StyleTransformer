import properties as prop

LANGS = [ "ENG", "RUS" ]


BUTTONS = {
    "HELP": {
        LANGS[0]: "help",
        LANGS[1]: "пояснение"
    },
    "CANCEL": {
        LANGS[0]: "cancel",
        LANGS[1]: "отмена"
    },
    "LANGUAGE": {
        LANGS[0]: "language",
        LANGS[1]: "язык"
    },
    "PROCESS": {
        LANGS[0]: "process",
        LANGS[1]: "начать"
    }
}

COMMANDS = {
    "START": {
        LANGS[0]: "Hello! This bot can do simple style transfer and GAN style transfer",
        LANGS[1]: "Добрый день! Этот бот умеет делать простой перенос стиля и перенос стиля с помощью генеративно-состязательной сети"
    },
    "IMAGE_HELP": {
        LANGS[0]: "Use \U0001f4ce to load source image",
        LANGS[1]: "Используйте \U0001f4ce для загрузки исходного изображения"
    },
    "STYLE_HELP": {
        LANGS[0]: ("Use \U0001f4ce to load style image for simple style transfer",
"Use buttons 1 - 4 to choose number of style image for GAN style transfer",
"Use command /get_all_predefined_styles to view all images for GAN style transfer"),
        LANGS[1]: ("Используйте \U0001f4ce для загрузки изображения со стилем для простого переноса стиля",
"Используйте кнопки 1 - 4 для выбора номера изображения для переноса стиля генеративно-состязательной сетью",
"Используйте команду /get_all_predefined_styles, чтобы посмотреть все изображения для переноса стиля с помощью\
генеративно-состязательной сети")
    },
    "AFFECT_HELP": {
        LANGS[0]: "Use buttons 1 - 10 to select the degree of affect of the style image on final \
image (works only for simple style transfer)",
        LANGS[1]: "Используйте кнопки 1 - 10 для выбора степени влияния изображения со стилем на итоговое \
изображение (работает только для простого переноса стиля)"
    },   
    "HELP": {
        LANGS[0]: ("Use /image command to load source image",
"Use /style command to load style image (for simple style transfer), or to choose style image (for GAN style transfer)",
"Use /affect command to select the degree of affect of the style image on final \
image (works only for simple style transfer)", "Use /make_magic command to start style transfer"),
        LANGS[1]: ("Используйте команду /image, чтобы загрузить исходное изображение",
"Используйте команду /style, чтобы загрузить изображение со стилем (для простого переноса стиля), \
или для выбора изображения со стилем (для переноса стиля с помощью генеративно-состязательной сети)",
"Используйте команду /affect для выбора степени влияния изображения со стилем на итоговое изображение \
(работает только для простого переноса стиля)", "Используйте команду /make_magic, чтобы начать перенос стиля")
    },
    "CANCEL": {
        LANGS[0]: "Cancelled",
        LANGS[1]: "Отменено"
    },
    "LANGUAGE": {
        LANGS[0]: "Choose language",
        LANGS[1]: "Выберите язык"
    },
    "AFFECT": {
        LANGS[0]: f"Send me value of affect of style image (value: 1 - {prop.AFFECT_MAX})",
        LANGS[1]: f"Пошлите мне значение влияния изображения со стилем (значение: 1 - {prop.AFFECT_MAX})"
    },
}

MESSAGES = {
    "SEND_ME_CONTENT_IMAGE": {
        LANGS[0]: "Send me source image",
        LANGS[1]: "Пошлите мне исходное изображение"
    },
    "CONTENT_IMAGE_RECEIVED": {
        LANGS[0]: "Content image received",
        LANGS[1]: "Исходное изображение получено"
    },
    "CHOOSE_STYLE_IMAGE": {
        LANGS[0]: "Send me style image for simple style transfer or choose style image for GAN style transfer",
        LANGS[1]: "Пошлите мне изображение со стилем для простого переноса стиля, или выберите изображение со \
стилем для переноса стиля с помощью генеративно-состязательной сети"
    },
    "STYLE_IMAGE_RECEIVED": {
        LANGS[0]: "Style image received",
        LANGS[1]: "Изображение со стилем получено"
    },
    "STYLE_IMAGE_CHOSEN": {
        LANGS[0]: "Style image chosen",
        LANGS[1]: "Изображение со стилем выбрано"
    },
    "AFFECT_CHANGED": {
        LANGS[0]: "Style image affect changed",
        LANGS[1]: "Влияние изображения со стилем изменено"
    },
    "WAIT_FOR_FEW_MINUTES": {
        LANGS[0]: "Please wait for a few minutes",
        LANGS[1]: "Пожалуйста, подождите несколько минут"
    },
    "WARNING": {
        LANGS[0]: "Warning!: ",
        LANGS[1]: "Внимание!: "
    },
    "ERROR": {
        LANGS[0]: "Error!: ",
        LANGS[1]: "Ошибка!: "
    },
    "LANGUAGE_CHANGED": {
        LANGS[0]: "Language changed",
        LANGS[1]: "Язык изменен"
    }
}


WARNINGS = {
    "TOO_BIG_IMAGE_WAS_COMPRESSED": {
        LANGS[0]: "it is too big size of image, image was compressed",
        LANGS[1]: "слишком большой размер изображения, изображение было сжато"
    },
    "VALUE_MUST_BE_INT": {
        LANGS[0]: "this value must be integer",
        LANGS[1]: "это значение должно быть целым"
    },
    "VALUE_MUST_BE_POSITIVE": {
        LANGS[0]: "this value must be positive",
        LANGS[1]: "это значение должно быть положительным"
    },
    "TOO_BIG_VALUE": {
        LANGS[0]: "it is too big value",
        LANGS[1]: "это слишком большое значение"
    },
    "TOO_SMALL_VALUE": {
        LANGS[0]: "it is too small value",
        LANGS[1]: "это слишком маленькое значение"
    },
    "VALUE_MUST_BE_NON_ZERO": {
        LANGS[0]: "this value must be non-zero",
        LANGS[1]: "это значение должно быть ненулевым"
    },
    "UNKNOWN_COMMAND": {
        LANGS[0]: "I don't know this command. Use /help",
        LANGS[1]: "Я не знаю такой команды. Используйте /help"
    },
    "UNKNOWN_LANGUAGE": {
        LANGS[0]: "I don't know this language",
        LANGS[1]: "я не знаю этот язык"
    },
    "CONTENT_IMAGE_NOT_RECEIVED": {
        LANGS[0]: "I didn't receive source image",
        LANGS[1]: "я не получил исходное изображение"
    },
    "STYLE_IMAGE_NOT_RECEIVED": {
        LANGS[0]: "I didn't receive style image",
        LANGS[1]: "я не получил изображение со стилем"
    },
    "STYLE_TRANSFERRING_ALREADY_RUNNING": {
        LANGS[0]: "style transferring is already running",
        LANGS[1]: "преобразование стиля уже идет"
    }
}

def warn(warning, language) -> str:
    return MESSAGES["WARNING"][language] + WARNINGS[warning][language]

def error(error, language) -> str:
    return MESSAGES["ERROR"][language] + WARNINGS[error][language]

    return MESSAGES["WARNING"][language] + WARNINGS[warning][language]
