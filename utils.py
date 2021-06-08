from aiogram import Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup


class States(StatesGroup):
    # state after calling of /start or /cancel
    # or after finish of processing immages of language selecting
    init_state = State()
    # state after calling of /language
    waiting_for_language = State()
    # state after calling of /process
    waiting_for_content_image = State()
    waiting_for_style_image = State()


