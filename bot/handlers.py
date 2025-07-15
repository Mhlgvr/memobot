from aiogram import Router, Bot
from aiogram.filters import Command
from aiogram.fsm.state import State, StatesGroup
from dotenv import load_dotenv
import os

from .keyboards import *

load_dotenv()
TOKEN = os.getenv('TOKEN')

router = Router()


class Classify(StatesGroup):
    sending_photo = State()
    last_message = State()

@router.message(Command('start'))
async def start(message):
    await message.answer('Выбери инструмент', reply_markup=main_menu)
    await message.delete()


@router.callback_query(lambda m: m.data == 'back')
async def back_to_menu(callback, state):
    await callback.answer()
    await state.clear()
    await start(callback.message)


@router.callback_query(lambda m: m.data.startswith('meme'))
async def select_service(callback):
    await callback.answer()
    model = callback.data.split('_')[-1]
    if model != 'classifier':
        await callback.message.edit_text('пока в разработке', reply_markup=main_menu)
    else:
        await callback.message.edit_text('Выбери модель', reply_markup=choose_classifier)


@router.callback_query(lambda m: m.data.startswith('model'))
async def select_model(callback, state):
    await callback.answer()
    model = callback.data.split('_')[-1]
    if model != 'resnet':
        await callback.message.edit_text('пока в разработке', reply_markup=choose_classifier)
    else:
        await state.set_state(Classify.sending_photo)
        await state.update_data(last_message_id=callback.message.message_id)
        await callback.message.edit_text('Кидай мем', reply_markup=cancel)

@router.message(Classify.sending_photo)
async def classify_photo(message, state):
    data = await state.get_data()
    await state.clear()
    bot = Bot(token=TOKEN)            # сомнительно но допустим
    await bot.delete_message(chat_id=message.chat.id, message_id=data['last_message_id'])
    await message.answer('Пошел нахуй', reply_markup=back)
    await message.delete()

