import os
from urllib.parse import urljoin
import asyncio

from aiogram import Bot
from aiogram.utils import executor
from aiogram.dispatcher import Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.contrib.middlewares.logging import LoggingMiddleware

import bot as bot_handlers
import properties as prop
import config


# webhook settings
WEBHOOK_PATH = f"/webhook/{config.TOKEN}"
WEBHOOK_URL = urljoin(config.WEBHOOK_HOST, WEBHOOK_PATH)
 
# webserver settings
WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = int(os.environ.get('PORT', config.WEBAPP_PORT))

bot = Bot(token=config.TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())
bot_handlers.register_handlers(dp)


async def on_startup(dp):
    print('Starting...')
    await bot.set_webhook(WEBHOOK_URL)

async def on_shutdown(dp):
    print('Shutting down...')
    print('Bye!')


def main():
    if prop.CONNECTION_TYPE == "POLLING":
        print("Connection is polling")
        executor.start_polling(dp)
    elif prop.CONNECTION_TYPE == "WEBHOOKS":
        print("Connection is polling")
        print("Host:", WEBAPP_HOST)
        print("Port:", WEBAPP_PORT)
        print("Url:", WEBHOOK_URL)
        executor.start_webhook(
            dispatcher=dp,
            webhook_path=WEBHOOK_PATH,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            skip_updates=True,
            host=WEBAPP_HOST,
            port=WEBAPP_PORT,
        )
    else:
        print("Uncorrect CONNECTION_TYPE")


if __name__ == '__main__':
    print("main started")
    main()

