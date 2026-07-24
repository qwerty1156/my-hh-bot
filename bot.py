# -*- coding: utf-8 -*-
from dotenv import load_dotenv

load_dotenv()
import json
import logging
import os
import time
from threading import Lock
from typing import Dict, Optional

import requests
import telebot
from flask import Flask, request
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

# Webhook configuration
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}" if BOT_TOKEN else "/webhook/unknown"

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


def set_webhook():
    """Удалить старый webhook и установить новый"""
    try:
        # Сначала удаляем любой существующий webhook
        logger.info("Удаление старого webhook...")
        bot.delete_webhook()
        time.sleep(1)
        
        if not WEBHOOK_URL:
            logger.warning(
                "WEBHOOK_URL не установлен. "
                "Webhook не будет работать."
            )
            return False
        
        # Затем устанавливаем новый
        logger.info(f"Установка webhook: {WEBHOOK_URL}{WEBHOOK_PATH}")
        bot.set_webhook(
            url=f"{WEBHOOK_URL}{WEBHOOK_PATH}"
        )
        logger.info("✅ Webhook успешно установлен")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при установке webhook: {e}")
        return False


@app.route(WEBHOOK_PATH, methods=["POST"])
def webhook():
    """Обработчик webhook от Telegram"""
    if request.headers.get("content-type") == "application/json":
        json_data = request.get_json()
        try:
            update = telebot.types.Update.de_json(json_data)
            bot.process_new_updates([update])
            logger.info(f"✅ Обновление обработано: {update.update_id}")
        except Exception as e:
            logger.error(f"❌ Ошибка обработки webhook: {e}")
    return "OK", 200


def run_web_server():
    """Запуск Flask веб-сервера"""
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 Запуск Flask на порту {port}...")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
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

        if not self.api_key:
            logger.error("GEMINI_API_KEY не задан. Пропускаем генерацию.")
            return None

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

        response = None

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

            # Попробуем несколько вариантов извлечения текста, чтобы быть устойчивыми к формату ответа
            text = None

            # Вариант: candidates -> content -> parts -> text
            candidates = data.get("candidates") or []
            if candidates:
                first = candidates[0]
                content = first.get("content") or {}
                parts = content.get("parts") or []
                if parts and isinstance(parts, list) and isinstance(parts[0], dict):
                    text = parts[0].get("text")

            # Вариант: outputs/outputs[0]/content
            if not text:
                outputs = data.get("outputs") or data.get("output") or []
                if outputs:
                    first_out = outputs[0]
                    if isinstance(first_out, dict):
                        # content может быть списком с объектами, содержащими text
                        cont = first_out.get("content")
                        if isinstance(cont, list) and cont:
                            maybe = cont[0]
                            if isinstance(maybe, dict):
                                text = maybe.get("text") or maybe.get("text_generation")
                        elif isinstance(cont, str):
                            text = cont
                        else:
                            # Прямое поле text
                            text = first_out.get("text")

            if not text:
                logger.warning("Не удалось извлечь текст из ответа Gemini: %s", data)
                return None

            return text

        except requests.exceptions.Timeout:

            logger.error("Таймаут Gemini")
            return None

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP ошибка: {e}")
            try:
                if response is not None:
                    logger.error(f"Ответ Gemini: {response.text}")
            except Exception:
                pass
            return None

        except requests.exceptions.RequestException as e:

            logger.error(f"Ошибка сети: {e}")
            return None

        except Exception as e:

            logger.exception(e)
            return None


# =======================================================

if not BOT_TOKEN:
    logger.warning("BOT_TOKEN не установлен. Telegram-бот не сможет запуститься.")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY не установлен. Генерация контента не будет работать.")

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

        # Удаляем уже добавленную часть и пропускаем переносы строк
        text = text[split_pos:].lstrip("\n")

    if text:
        parts.append(text)

    return parts


def clean_markdown(text: str) -> str:
    """Удаляет Markdown-символы из текста"""
    # Удаляем ** (жирный)
    text = text.replace("**", "")
    # Удаляем * (курсив)
    text = text.replace("*", "")
    # Удаляем __ (жирный)
    text = text.replace("__", "")
    # Удаляем _ (курсив)
    text = text.replace("_", "")
    # Удаляем ## и другие заголовки
    text = text.replace("##", "").replace("#", "")
    return text


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

        if not result or len(result.strip()) < 50:

            bot.reply_to(
                message,
                (
                    "❌ Не удалось получить "
                    "полный ответ от Gemini.\n"
                    "Попробуйте позже."
                )
            )

            return

        result = clean_markdown(result)

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
                int(target_uid),
                f"🎉 Баланс успешно пополнен! Вам добавлено {count} попыток. Удачи в поиске работы!"
            )
        except Exception as e:
            print(f"Не удалось отправить сообщение пользователю {target_uid}: {e}")

    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка: {e}")


# =======================================================

if __name__ == "__main__":
    
    logger.info("=" * 50)
    logger.info("🤖 Запуск Telegram-бота в webhook-режиме...")
    logger.info("=" * 50)
    
    # Проверяем обязательные переменные окружения
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN не установлен!")
        exit(1)
    
    if not WEBHOOK_URL:
        logger.error("❌ WEBHOOK_URL не установлен!")
        logger.error("Пример для Render: https://my-app.onrender.com")
        exit(1)
    
    if not GEMINI_API_KEY:
        logger.warning("⚠️  GEMINI_API_KEY не установлен. Генерация контента не будет работать.")
    
    # Устанавливаем webhook перед запуском сервера
    if set_webhook():
        logger.info("✅ Инициализация успешна. Запуск веб-сервера...")
    else:
        logger.error("❌ Не удалось установить webhook. Проверьте WEBHOOK_URL и BOT_TOKEN.")
        exit(1)
    
    # Запускаем Flask
    run_web_server()
