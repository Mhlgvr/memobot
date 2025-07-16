from aiogram import Router, Bot
from aiogram.filters import Command
from aiogram.fsm.state import State, StatesGroup
from io import BytesIO

from dotenv import load_dotenv
import os

from .keyboards import *
from models.efficientnet import EfficientNetMemeClassifier
from models.resnet import   ResNetMemeClassifier

load_dotenv()
TOKEN = os.getenv('TOKEN')

router = Router()
available_models = ['resnet', 'efficientnet']


class Classify(StatesGroup):
    sending_photo = State()
    last_message = State()

@router.message(Command('start'))
async def start(message):
    await message.answer('Выбери инструмент', reply_markup=main_menu)


@router.callback_query(lambda m: m.data == 'back')
async def back_to_menu(callback, state):
    await callback.answer()
    await state.clear()
    await callback.message.edit_text('Выбери инструмент', reply_markup=main_menu)


@router.callback_query(lambda m: m.data.startswith('meme'))
async def select_service(callback):
    model = callback.data.split('_')[-1]
    if model != 'classifier':
        await callback.answer('пока в разработке')
    else:
        await callback.answer()
        await callback.message.edit_text('Выбери модель', reply_markup=choose_classifier)


@router.callback_query(lambda m: m.data.startswith('model'))
async def select_model(callback, state):
    model = callback.data.split('_')[-1]
    if model not in available_models:
        await callback.answer('Эта модель еще не готова')
    else:
        await callback.answer()
        await state.update_data(model = model)
        await state.set_state(Classify.sending_photo)
        await state.update_data(last_message_id=callback.message.message_id)
        await callback.message.edit_text('Кидай мем', reply_markup=cancel)

@router.message(Classify.sending_photo)
async def classify_photo(message, state):
    data = await state.get_data()
    await state.clear()

    if data['model'] == 'resnet':
        model = ResNetMemeClassifier()
    elif data['model'] == 'efficientnet':
        model = EfficientNetMemeClassifier()
    else:
        print('чето не то')
    
    bot = message.bot 

    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)

    byte_stream = BytesIO()
    await bot.download_file(file_info.file_path, destination=byte_stream)
    byte_stream.seek(0)

    pred, probs = model.predict(byte_stream)

    # bot = Bot(token=TOKEN)            # сомнительно но допустим
    await bot.delete_message(chat_id=message.chat.id, message_id=data['last_message_id'])
    await message.answer(f"Хуйня (уверенность: {probs[0]})", reply_markup=back)
    await message.delete()

