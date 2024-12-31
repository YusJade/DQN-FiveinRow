from flask import Flask, request, jsonify
import openai
import os

# 初始化 Flask 应用
app = Flask(__name__)

# 配置 OpenAI API 密钥
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key, base_url="https://api.deepseek.com")


@app.route('/chat', methods=['POST'])
def chat():
    try:
        # 获取前端传递的数据
        data = request.json
        user_input = data.get('message', '')

        if not user_input:
            return jsonify({"error": "Message is required"}), 400

        # 调用 OpenAI API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            stream=False
        )

        # 提取回复
        assistant_reply = response['choices'][0]['message']['content']
        return jsonify({"reply": assistant_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
