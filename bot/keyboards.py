from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton



main_menu = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='Чек мема', callback_data='meme_classifier')],
        [InlineKeyboardButton(text='Генерация мема', callback_data='meme_generator')],
        [InlineKeyboardButton(text='Оценка мема', callback_data='meme_rating')]
])

choose_classifier = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='ResNet', callback_data='model_resnet')],
        [InlineKeyboardButton(text='EfficientNet', callback_data='model_efficientnet')],
        [InlineKeyboardButton(text='YOLO', callback_data='model_yolo')],
        [InlineKeyboardButton(text='Swin', callback_data='model_swin')],
])

cancel = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='Отмена', callback_data='cancel')]
])

back = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='Спасибо', callback_data='back')]
])