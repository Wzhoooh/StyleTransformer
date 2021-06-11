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
        LANGS[0]: "Hello! This bot can do style transfer. To load source image and image with \
style, use /image and /style commands, respectively. To start style transfer, use /make_magic command.",
        LANGS[1]: "Добрый день! Этот бот умеет делать перенос стиля. Чтобы загрузить исходное \
изображение и изображение со стилем, используйте команды /image и /style соответственно. Для того, \
чтобы запустить перенос стиля, используйте команду /make_magic."
    },
    "HELP": {
        LANGS[0]: "To load source image and image with style, use /image and /style commands, \
respectively. To start style transfer, use /make_magic command.",
        LANGS[1]: "Чтобы загрузить исходное изображение и изображение со стилем, используйте \
команды /image и /style соответственно. Для того, чтобы запустить перенос стиля, используйте \
команду /make_magic."
    },
    "CANCEL": {
        LANGS[0]: "Cancelled",
        LANGS[1]: "Отменено"
    },
    "LANGUAGE": {
        LANGS[0]: "Choose language",
        LANGS[1]: "Выберите язык"
    }
}

MESSAGES = {
    "SEND_ME_CONTENT_IMAGE": {
        LANGS[0]: "Send me source image",
        LANGS[1]: "Пошлите мне исходное изображение"
    },
    "SEND_ME_STYLE_IMAGE": {
        LANGS[0]: "Send me style image",
        LANGS[1]: "Пошлите мне изображение со стилем"
    },
    "IMAGE_RECEIVED": {
        LANGS[0]: "Image received",
        LANGS[1]: "Изображение получено"
    },
    "WAIT_FOR_SEVERAL_MINUTES": {
        LANGS[0]: "Wait for several minutes",
        LANGS[1]: "Подождите несколько минут"
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
    "NEGATIVE_VALUE_MUST_BE_POSITIVE": {
        LANGS[0]: "this value must be positive",
        LANGS[1]: "это значение должно быть положительным"
    },
    "TOO_BIG_VALUE": {
        LANGS[0]: "it is too big value",
        LANGS[1]: "это слишком большое значение"
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
    }
}

def warn(warning, language) -> str:
    return MESSAGES["WARNING"][language] + WARNINGS[warning][language]

def error(error, language) -> str:
    return MESSAGES["ERROR"][language] + WARNINGS[error][language]

    return MESSAGES["WARNING"][language] + WARNINGS[warning][language]
