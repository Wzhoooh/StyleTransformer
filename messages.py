CUR_LANG = "RUS"

LANGS = [ "ENG", "RUS" ]


COMMANDS = {
    "start": {
        LANGS[0]: "Hello, this bot can do style transferring. To learn how, use /help command",
        LANGS[1]: "Добрый день, этот бот умеет делать перенос стиля. Чтобы узнать, как воспольльуйтесь командой /help"
    },
    "help": {
        LANGS[0]: "This is /help documentation",
        LANGS[1]: "Это /help документация"
    }
}

MESSAGES = {
    "SEND_ME_CONTENT_IMAGE": {
        LANGS[0]: "Send me content image",
        LANGS[1]: "Пошлите мне сообщение с контентом"
    },
    "SEND_ME_STYLE_IMAGE": {
        LANGS[0]: "Send me style image",
        LANGS[1]: "Пошлите мне сообщение со стилем"
    },
    "WAIT_FOR_SEVERAL_MINUTES": {
        LANGS[0]: "Wait for several minutes",
        LANGS[1]: "Подождите несколько минут"
    },
    "UNKNOWN_COMMAND": {
        LANGS[0]: "I don't know this command. Use /help",
        LANGS[1]: "Я не знаю такой команды. Используйте /help"
    },
    "WARNING": {
        LANGS[0]: "Warning!: ",
        LANGS[1]: "Внимание!: "
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
    }
}

def warn(warning, language) -> str:
    return MESSAGES["WARNING"][language] + WARNINGS[warning][language]
