from telegram.ext import Updater, CommandHandler, MessageHandler
from telegram.ext.filters import Filters
import requests
from PIL import Image
import config

def hello(bot, update):
    # print(bot)
    print(update.message.chat.username)
    reply = "hello world"
    update.message.reply_text(reply)


def voice_handler(bot, update):
    file = bot.getFile(update.message.document.file_id)
    im = Image.open(file.download('image.png'))
    # 將傳過來的圖片儲存至 bot_image/from_usr/{username}_{file_id}_image.png
    image_name = f"bot_image/from_usr/{update.message.chat.username}_{file.file_id}_image.png"
    im.save(image_name)
    # 用 post 的傳輸圖片
    requests.post(config.url + 'sendPhoto', data={'chat_id': update.message.chat.id}, files={'photo': open('bot_image/from_usr/Yeamao_BQADBQADdwADuTv4VOQIN14c7u1FFgQ_image.png', 'rb')})
    

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
