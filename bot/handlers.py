from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.state import State, StatesGroup
from io import BytesIO

from dotenv import load_dotenv
import os

from .keyboards import (
    main_menu, choose_model, cancel, back
)
from models.efficientnet import EfficientNetMemeClassifier, EfficientNetMemeRegressor
from models.resnet import ResNetMemeClassifier, ResNetMemeRegressor

load_dotenv()
TOKEN = os.getenv('TOKEN')

router = Router()
available_models = ['resnet', 'efficientnet']


class Classifier(StatesGroup):
    sending_photo = State()
    last_message = State()


class Regressor(StatesGroup):
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
async def select_service(callback, state):
    model = callback.data.split()[-1]
    if model == 'regressor':
        await callback.answer()
        await state.update_data(task='regression')
        await callback.message.edit_text('Выбери модель', reply_markup=choose_model)
    elif model == 'classifier':
        await callback.answer()
        await state.update_data(task='classification')
        await callback.message.edit_text('Выбери модель', reply_markup=choose_model)
    else:
        await callback.answer('пока в разработке')


@router.callback_query(lambda m: m.data == 'cancel')
async def back_to_select_model(callback, state):
    await callback.answer()
    await state.clear()
    await select_service(callback, state)


@router.callback_query(lambda m: m.data.startswith('model'))
async def select_model(callback, state):
    data = await state.get_data()
    model = callback.data.split()[-1]
    task = data['task']
    if model not in available_models:
        await callback.answer('Эта модель еще не готова')
    else:
        await callback.answer()
        await state.update_data(model=model)
        if task == 'classification':
            await state.set_state(Classifier.sending_photo)
        else:
            await state.set_state(Regressor.sending_photo)
        await state.update_data(last_message_id=callback.message.message_id)
        await callback.message.edit_text('Кидай мем', reply_markup=cancel)


@router.message(Classifier.sending_photo)
async def classify_photo(message, state):
    data = await state.get_data()
    await state.clear()

    if data['model'] == 'resnet':
        model = ResNetMemeClassifier()
    elif data['model'] == 'efficientnet':
        model = EfficientNetMemeClassifier()
    else:
        print('чето не то')

    bot = message.bot  # сомнительно но допустим

    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)

    byte_stream = BytesIO()
    await bot.download_file(file_info.file_path, destination=byte_stream)
    byte_stream.seek(0)

    pred, probs = model.predict(byte_stream)

    await bot.delete_message(
        chat_id=message.chat.id, message_id=data['last_message_id']
    )
    await message.answer(
        f"Хуйня (уверенность: {probs[0]})", reply_markup=back
    )
    await message.delete()


@router.message(Regressor.sending_photo)
async def predict_photo(message, state):
    data = await state.get_data()
    await state.clear()

    if data['model'] == 'resnet':
        model = ResNetMemeRegressor()
    elif data['model'] == 'efficientnet':
        model = EfficientNetMemeRegressor()
    else:
        print('чето не то')

    bot = message.bot  # сомнительно но допустим

    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)

    byte_stream = BytesIO()
    await bot.download_file(file_info.file_path, destination=byte_stream)
    byte_stream.seek(0)

    likes, reposts, views = model.predict(byte_stream)

    await bot.delete_message(
        chat_id=message.chat.id, message_id=data['last_message_id']
    )
    await message.answer(
        f"Пиздец. {likes} лайков, {reposts} репостов, {views} views",
        reply_markup=back
    )
    await message.delete()
