# -*- coding: utf-8 -*-
from dotenv import load_dotenv

load_dotenv()
import json
import logging
import os
import time
from threading import Lock, Thread
from typing import Dict, Optional

import requests
import telebot
from flask import Flask
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ====================== НАСТРОЙКИ ======================

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")

MAX_FREE_REQUESTS = 3

USERS_FILE = "users_limits.json"

MODEL_NAME = "gemini-2.5-flash"

REQUEST_TIMEOUT = (5, 60)

# =======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("CoverLetterBot")

# =======================================================

app = Flask(__name__)

lock = Lock()


@app.route("/")
def home():
    return "Bot is running!"


def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    app.run(
        host="0.0.0.0",
        port=port
    )

# =======================================================

SYSTEM_PROMPT = """
Ты профессиональный HR-директор с опытом более 15 лет.

Твоя задача — написать максимально качественное сопроводительное письмо.

Правила:

• НЕ придумывай опыт которого нет.

• Используй исключительно информацию из вакансии.

• Сделай письмо естественным.

• Не используй шаблонные фразы.

• Пиши уверенно.

• Максимум 250 слов.

• Начни с приветствия.

• Закончи призывом пригласить на собеседование.

• Используй красивое форматирование.

Описание вакансии:

"""

# =======================================================


class GeminiClient:

    def __init__(self, api_key: str):

        self.api_key = api_key

        self.base_url = (
            "https://generativelanguage.googleapis.com/v1"
        )

        self.session = requests.Session()

        retry = Retry(
            total=3,
            connect=3,
            backoff_factor=1,
            status_forcelist=[
                429,
                500,
                502,
                503,
                504,
            ],
        )

        adapter = HTTPAdapter(max_retries=retry)

        self.session.mount("https://", adapter)

        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        }

    def generate_content(
        self,
        prompt: str
    ) -> Optional[str]:

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 2048
            }
        }

        try:

            response = self.session.post(
                f"{self.base_url}/models/{MODEL_NAME}:generateContent",
                params={
                    "key": self.api_key
                },
                json=payload,
                headers=self.headers,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            data = response.json()

            candidates = data.get("candidates")

            if not candidates:
                logger.warning("Gemini вернул пустой ответ.")
                return None

            content = (
                candidates[0]
                .get("content", {})
                .get("parts", [])
            )

            if not content:
                return None

            return content[0].get("text")

        except requests.exceptions.Timeout:

            logger.error("Таймаут Gemini")

        except requests.exceptions.HTTPError as e:

            logger.error(f"HTTP ошибка: {e}")

        except requests.exceptions.RequestException as e:

            logger.error(f"Ошибка сети: {e}")

        except Exception as e:

            logger.exception(e)

        return None


# =======================================================

bot = telebot.TeleBot(BOT_TOKEN)

ai_client = GeminiClient(GEMINI_API_KEY)

# =======================================================


def load_limits() -> Dict[str, int]:

    with lock:

        if not os.path.exists(USERS_FILE):
            return {}

        try:

            with open(
                USERS_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                return {
                    str(k): int(v)
                    for k, v in json.load(f).items()
                }

        except Exception:

            return {}


def save_limits(data: Dict[str, int]):

    with lock:

        with open(
            USERS_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=4
            )
def get_user_limits(user_id: str) -> int:

    limits = load_limits()

    if user_id not in limits:

        limits[user_id] = MAX_FREE_REQUESTS

        save_limits(limits)

    return limits[user_id]


def decrease_user_limits(user_id: str):

    limits = load_limits()

    current = limits.get(
        user_id,
        MAX_FREE_REQUESTS
    )

    limits[user_id] = max(
        0,
        current - 1
    )

    save_limits(limits)


def split_message(
    text: str,
    max_length: int = 4000
):

    if len(text) <= max_length:
        return [text]

    parts = []

    while len(text) > max_length:

        split_pos = text.rfind(
            "\n",
            0,
            max_length
        )

        if split_pos == -1:
            split_pos = max_length

        parts.append(
            text[:split_pos]
        )

        text = text[split_pos:]

    if text:
        parts.append(text)

    return parts


def send_long_message(
    chat_id: int,
    text: str
):

    parts = split_message(text)

    for part in parts:

        bot.send_message(
            chat_id,
            part,
            parse_mode="HTML"
        )


# =======================================================

@bot.message_handler(
    commands=["start"]
)
def send_welcome(message):

print(
    f"ID: {message.from_user.id} | "
    f"Username: @{message.from_user.username} | "
    f"Имя: {message.from_user.first_name}"
)

    user_id = str(
        message.from_user.id
    )

    remaining = get_user_limits(
        user_id
    )

    text = (
        "👋 <b>Привет!</b>\n\n"
        "Я помогу составить "
        "качественное сопроводительное письмо.\n\n"
        f"📊 Осталось бесплатных попыток: "
        f"<b>{remaining}</b>\n\n"
        "Отправь описание вакансии."
    )

    bot.reply_to(
        message,
        text,
        parse_mode="HTML"
    )

    logger.info(
        f"START | {user_id}"
    )


# =======================================================

@bot.message_handler(
    func=lambda m: True,
    content_types=["text"]
)
def handle_vacancy(message):

    if message.text.startswith("/"):
        return

    text = (
        message.text or ""
    ).strip()

    if not text:

        bot.reply_to(
            message,
            "❌ Текст вакансии пуст."
        )

        return

    user_id = str(
        message.from_user.id
    )

    remaining = get_user_limits(
        user_id
    )

    if remaining <= 0:

        bot.reply_to(
            message,
            (
                "🚫 Лимит исчерпан.\n\n"
                f"Для продления доступа: "
                f"{ADMIN_USERNAME}"
            )
        )

        return

    logger.info(
        f"REQUEST | {user_id}"
    )

    wait_message = bot.reply_to(
        message,
        "⏳ Генерирую письмо..."
    )

    try:

        prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + text
        )

        result = ai_client.generate_content(
            prompt
        )

        try:

            bot.delete_message(
                wait_message.chat.id,
                wait_message.message_id
            )

        except Exception:
            pass

        if not result:

            bot.reply_to(
                message,
                (
                    "❌ Не удалось получить "
                    "ответ от Gemini.\n"
                    "Попробуйте позже."
                )
            )

            return

        decrease_user_limits(
            user_id
        )

        new_remaining = (
            get_user_limits(
                user_id
            )
        )

        # Отправляем письмо
        send_long_message(
            message.chat.id,
            result
        )

        # Отправляем количество попыток отдельным сообщением
        bot.send_message(
            message.chat.id,
            f"📊 Осталось попыток: <b>{new_remaining}</b>",
            parse_mode="HTML"
        )

        logger.info(
            f"SUCCESS | {user_id}"
        )

    except Exception as e:

        logger.exception(e)

        try:

            bot.reply_to(
                message,
                (
                    "❌ Внутренняя ошибка.\n"
                    "Попробуйте позже."
                )
            )

        except Exception:
            pass
# =======================================================
ADMIN_ID = 6882795498

@bot.message_handler(commands=['grant'])
def grant_attempts(message):

    # Проверка прав администратора
    if message.from_user.id != ADMIN_ID:
        bot.reply_to(message, "❌ У вас нет прав.")
        return

    try:
        parts = message.text.split()

        # Проверяем формат команды
        if len(parts) != 3:
            bot.reply_to(message, "❌ Использование: /grant ID_ПОЛЬЗОВАТЕЛЯ КОЛИЧЕСТВО")
            return

        target_uid = str(parts[1])

        try:
            count = int(parts[2])
        except ValueError:
            bot.reply_to(message, "❌ Количество должно быть числом.")
            return

        if count <= 0:
            bot.reply_to(message, "❌ Количество должно быть больше нуля.")
            return

        lim = load_limits()
        lim[target_uid] = lim.get(target_uid, 0) + count
        save_limits(lim)

        bot.reply_to(
            message,
            f"✅ Успешно! Пользователю {target_uid} добавлено {count} попыток."
        )

        # Пытаемся уведомить пользователя
        try:
            bot.send_message(
                target_uid,
                f"🎉 Баланс успешно пополнен! Вам добавлено {count} попыток. Удачи в поиске работы!"
            )
        except Exception as e:
            print(f"Не удалось отправить сообщение пользователю {target_uid}: {e}")

except Exception as e:
    bot.reply_to(message, f"❌ Ошибка: {e}")


def start_bot():
    while True:

        try:

            logger.info(
                "Запуск Telegram-бота..."
            )

            bot.infinity_polling(
                timeout=30,
                long_polling_timeout=30,
                skip_pending=True
            )

        except KeyboardInterrupt:

            logger.info(
                "Бот остановлен."
            )

            break

        except Exception as e:

            logger.exception(e)

            logger.info(
                "Перезапуск через 5 секунд..."
            )

            time.sleep(5)


# =======================================================

if __name__ == "__main__":

    logger.info(
        "Запуск веб-сервера..."
    )

    web_thread = Thread(
        target=run_web_server,
        daemon=True
    )

    web_thread.start()

    logger.info(
        "Веб-сервер успешно запущен."
    )

    start_bot()
