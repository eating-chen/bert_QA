# -- coding:UTF-8 --
from __future__ import unicode_literals
import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import demo

app = Flask(__name__)
epoch_model_str = ''
model_obj = demo.load_model(epoch_model_str)
user_record = {}

line_bot_api = LineBotApi('XXXX')
handler = WebhookHandler('XXXX')

# 接收 LINE 的資訊
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']

    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def echo(event):
    if event.source.user_id != "Udeadbeefdeadbeefdeadbeefdeadbeef":
        final_answer = ''
        model_answer = demo.chatBotFlow(event.message.text,
                                        model_obj['tokenizer'],
                                        model_obj['model'],
                                        model_obj['device'],
                                        model_obj['s_bert_model'])
        # print("ans ======> ", model_answer)
        for idx in range(len(model_answer)):
            if model_answer[idx]['answer'] != '輸入其他問題試試?':
                final_answer += model_answer[idx]['answer'] + '\n'
        
        if final_answer == '':
            final_answer = '我有點不太清楚你的意思><，可以用其他方式再問問看ㄛ~~'
            
        line_bot_api.push_message(event.source.user_id, 
                        TextSendMessage(text=final_answer))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))