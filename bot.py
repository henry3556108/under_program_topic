from telegram.ext import Updater, CommandHandler, MessageHandler
from telegram.ext.filters import Filters
import requests
from PIL import Image
import config
import generate as G
def hello(bot, update):
    # print(bot)
    print(update.message.chat.username)
    reply = "hello world"
    update.message.reply_text(reply)


def voice_handler(bot, update):
    file = bot.getFile(update.message.document.file_id)
    file_name = f"{update.message.chat.username}_{file.file_id}_image.png"
    file.download(f'bot_image/from_usr/{file_name}')
    print(str(file.file_id))

    im = Image.open(f'bot_image/from_usr/{file_name}')
    # # 將傳過來的圖片儲存至 bot_image/from_usr/{username}_{file_id}_image.png
    image_name = f"bot_image/to_usr/{update.message.chat.username}_{file.file_id}_image_new.png"
    time = G.generate(file_name)
    update.message.reply_text(f"generator spend {time:2.2f} seconds")
    requests.post(config.url + 'sendDocument', data={'chat_id': update.message.chat.id}, files={'document': open(image_name, 'rb')})
    
def main():
    updater = Updater(config.token)
    updater.dispatcher.add_handler(CommandHandler('test', hello))
    # 可以接收使用者傳過來的圖片
    updater.dispatcher.add_handler(
        MessageHandler(Filters.document, voice_handler))
    updater.start_polling()
    updater.idle()  # 不要結束程式


if __name__ == "__main__":
    main()
