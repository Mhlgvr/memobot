from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

main_menu = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='Чек мема', callback_data='meme classifier')],
    [InlineKeyboardButton(text='Генерация мема', callback_data='meme generator')],
    [InlineKeyboardButton(text='Оценка мема', callback_data='meme regressor')]
])

choose_model = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='ResNet', callback_data='model resnet')],
    [InlineKeyboardButton(
        text='EfficientNet', callback_data='model efficientnet'
    )],
    [InlineKeyboardButton(text='ViT', callback_data='model vit')],
    [InlineKeyboardButton(text='Назад', callback_data='back')]
])

# choose_regressor = InlineKeyboardMarkup(inline_keyboard=[
#         [InlineKeyboardButton(text='ResNet', callback_data='regressor_resnet')],
#         [InlineKeyboardButton(text='EfficientNet', callback_data='regressor_efficientnet')],
# ])

cancel = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='Отмена', callback_data='cancel')]
])

back = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='Спасибо', callback_data='back')]
])