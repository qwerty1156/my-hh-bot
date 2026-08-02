# -*- coding: utf-8 -*-
from dotenv import load_dotenv

load_dotenv()
import json
import logging
import os
import time
from threading import Lock
from typing import Dict, Optional
from datetime import datetime, timedelta
import uuid

import requests
import telebot
from flask import Flask, request
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from telebot import types as teletypes

# ====================== НАСТРОЙКИ / CONFIG ======================

# Основные параметры (залейте сюда значения или используйте os.getenv)
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID") or "6882795498")  # целое число
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
KASPI_PHONE = os.getenv("KASPI_PHONE") or "+7XXXXXXXXXX"
KASPI_NAME = os.getenv("KASPI_NAME") or "ИП Пример"
PROVIDER_TOKEN = os.getenv("PROVIDER_TOKEN") or "PROVIDER_TOKEN_FROM_BOTFATHER"  # для Telegram Payments (Stars)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Прочие настройки
MAX_FREE_REQUESTS = 3
USERS_FILE = "users_limits.json"
PAYMENTS_FILE = "payments.json"          # отслеживание заявок/инвойсов
SUBSCRIPTIONS_FILE = "subscriptions.json"  # подписки/безлимит
KASPI_PENDING_FILE = "kaspi_pending.json"  # опционально, можно хранить в PAYMENTS_FILE

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
                "maxOutputTokens": 4096
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

def load_json_file(path: str) -> dict:
    """Универсальная загрузка json-файла, возвращает {} при ошибке."""
    with lock:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


def save_json_file(path: str, data: dict):
    """Универсальное сохранение json-файла."""
    with lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def get_user_limits(user_id: str) -> int:
    """
    Возвращает количество оставшихся откликов пользователя.
    Если у пользователя активный безлимит — возвращаем большое число (или специальное значение).
    """
    # Сначала проверим подписки (безлимит)
    subs = load_json_file(SUBSCRIPTIONS_FILE)
    user_sub = subs.get(user_id)
    if user_sub:
        # ожидаем структуру: {"type": "unlimited", "expires_at": "ISO8601"}
        try:
            expires = datetime.fromisoformat(user_sub.get("expires_at"))
            if expires > datetime.utcnow():
                # Активный безлимит: возвращаем большой лимит (не будем уменьшать)
                return 999999
        except Exception:
            pass

    # Обычные лимиты
    limits = load_json_file(USERS_FILE)
    if user_id not in limits:
        limits[user_id] = MAX_FREE_REQUESTS
        save_json_file(USERS_FILE, limits)
    try:
        return int(limits.get(user_id, MAX_FREE_REQUESTS))
    except Exception:
        return MAX_FREE_REQUESTS


def decrease_user_limits(user_id: str):
    """
    Уменьшает лимит пользователя на 1, если у него нет активного безлимита.
    """
    # Проверим подписку
    subs = load_json_file(SUBSCRIPTIONS_FILE)
    user_sub = subs.get(user_id)
    if user_sub:
        try:
            expires = datetime.fromisoformat(user_sub.get("expires_at"))
            if expires > datetime.utcnow():
                # Безлимит активен — не уменьшаем
                return
        except Exception:
            pass

    limits = load_json_file(USERS_FILE)
    current = int(limits.get(user_id, MAX_FREE_REQUESTS))
    limits[user_id] = max(0, current - 1)
    save_json_file(USERS_FILE, limits)


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


# ====================== PAYMENT / KASPI IMPLEMENTATION ======================

# Тарифы: ключ -> параметры
TARIFFS = {
    "start": {
        "title": "⚡️ СТАРТ",
        "replies": 15,
        "price_tenge": 990,
        "stars": 100,
        "duration_days": None,  # None = не подписка
    },
    "benefit": {
        "title": "🚀 ВЫГОДНЫЙ",
        "replies": 50,
        "price_tenge": 1990,
        "stars": 200,
        "duration_days": None,
    },
    "unlimited": {
        "title": "👑 БЕЗЛИМИТ",
        "replies": None,
        "price_tenge": 3990,
        "stars": 400,
        "duration_days": 30,  # 30 дней безлимита
    },
}


# Вспомогательные функции для работы с хранилищами
def load_payments() -> dict:
    return load_json_file(PAYMENTS_FILE)


def save_payments(data: dict):
    save_json_file(PAYMENTS_FILE, data)


def create_payment_record(user_id: str, tariff_key: str, method: str) -> str:
    """
    Создаёт запись платежа и возвращает request_id.
    method: 'stars' или 'kaspi'
    """
    payments = load_payments()
    req_id = str(uuid.uuid4())
    payments[req_id] = {
        "user_id": user_id,
        "tariff": tariff_key,
        "method": method,
        "created_at": datetime.utcnow().isoformat(),
        "status": "pending",  # pending, confirmed, rejected
        "processed_at": None,
        "notes": None,
    }
    save_payments(payments)
    return req_id


def mark_payment_processed(req_id: str, status: str, admin_id: Optional[int] = None, notes: Optional[str] = None) -> bool:
    payments = load_payments()
    rec = payments.get(req_id)
    if not rec:
        return False
    if rec.get("status") != "pending":
        # уже обработано
        return False
    rec["status"] = status
    rec["processed_at"] = datetime.utcnow().isoformat()
    rec["processed_by"] = admin_id
    rec["notes"] = notes
    save_payments(payments)
    return True


def apply_tariff_to_user(user_id: str, tariff_key: str):
    """
    Начисление тарифа пользователю.
    Для обычных пакетов — прибавляем количество откликов.
    Для безлимита — создаём запись подписки с датой окончания.
    """
    t = TARIFFS.get(tariff_key)
    if not t:
        raise ValueError("Unknown tariff")
    if t.get("duration_days"):
        # создаём/обновляем подписку
        subs = load_json_file(SUBSCRIPTIONS_FILE)
        expires = datetime.utcnow() + timedelta(days=t["duration_days"])
        subs[user_id] = {
            "type": "unlimited",
            "expires_at": expires.isoformat()
        }
        save_json_file(SUBSCRIPTIONS_FILE, subs)
    else:
        # начисляем отклики
        limits = load_json_file(USERS_FILE)
        current = int(limits.get(user_id, MAX_FREE_REQUESTS))
        add = int(t.get("replies") or 0)
        limits[user_id] = current + add
        save_json_file(USERS_FILE, limits)


# Временная память в работе процесса: user_id -> req_id
KASPI_WAITING = {}


def make_tariffs_keyboard():
    keyboard = teletypes.InlineKeyboardMarkup()
    for key, t in TARIFFS.items():
        text_line = f"{t['title']} — {t['price_tenge']} ₸"
        # Две кнопки на тариф: Kaspi и Stars
        kb = teletypes.InlineKeyboardMarkup(row_width=2)
        btn_kaspi = teletypes.InlineKeyboardButton(
            f"💳 Kaspi ({t['price_tenge']}₸)",
            callback_data=f"kaspi:{key}"
        )
        btn_stars = teletypes.InlineKeyboardButton(
            f"⭐️ Stars ({t['stars']} XTR)",
            callback_data=f"stars:{key}"
        )
        kb.add(btn_kaspi, btn_stars)
        keyboard.add(kb)
    return keyboard


@bot.message_handler(commands=['buy', 'prices'])
def cmd_buy(message):
    """
    Показывает список тарифов и кнопки для оплаты.
    """
    text_lines = ["💰 <b>Выберите тариф для оплаты:</b>\n"]
    for key, t in TARIFFS.items():
        if t.get("duration_days"):
            desc = f"<b>{t['title']}</b>\n💬 {t['duration_days']} дней безлимита\n💰 {t['price_tenge']} ₸ | ⭐️ {t['stars']} XTR\n"
        else:
            desc = f"<b>{t['title']}</b>\n💬 {t.get('replies')} откликов\n💰 {t['price_tenge']} ₸ | ⭐️ {t['stars']} XTR\n"
        text_lines.append(desc)
    bot.send_message(message.chat.id, "\n".join(text_lines), reply_markup=make_tariffs_keyboard(), parse_mode="HTML")


@bot.callback_query_handler(func=lambda call: call.data and (call.data.startswith("kaspi:") or call.data.startswith("stars:")))
def callback_payment_choice(call: teletypes.CallbackQuery):
    data = call.data  # e.g., "kaspi:start" или "stars:benefit"
    method, tariff_key = data.split(":", 1)
    user_id = str(call.from_user.id)

    if tariff_key not in TARIFFS:
        bot.answer_callback_query(call.id, "Неизвестный тариф.", show_alert=True)
        return

    if method == "kaspi":
        # Создаём заявку и отправляем инструкцию пользователю
        req_id = create_payment_record(user_id, tariff_key, "kaspi")
        t = TARIFFS[tariff_key]
        msg = (
            f"Переведите <b>{t['price_tenge']} ₸</b> на номер <b>{KASPI_PHONE}</b>\n\n"
            f"Получатель: <b>{KASPI_NAME}</b>\n\n"
            "После перевода нажмите кнопку ниже и отправьте фото чека."
        )
        kb = teletypes.InlineKeyboardMarkup()
        kb.add(teletypes.InlineKeyboardButton("🧾 Я оплатил (Отправить чек)", callback_data=f"kaspi_paid:{req_id}"))
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, msg, reply_markup=kb, parse_mode="HTML")
        return

    if method == "stars":
        # Создадим invoice через Telegram Payments
        t = TARIFFS[tariff_key]
        title = t["title"]
        description = f"{t['title']} — {t.get('replies') or 'Безлимит'}"
        # payload: используем уникальную строку, чтобы потом распознать
        payload = f"stars|{user_id}|{tariff_key}|{str(uuid.uuid4())}"
        # Цены: currency XTR, amount в "минимальных единицах" (целое).
        # Интерпретируем "stars" как сумма в XTR (в минимальных единицах умножаем на 100).
        amount = int(t["stars"]) * 100
        prices = [teletypes.LabeledPrice(label=f"{title}", amount=amount)]

        try:
            # Создаем запись платежа, чтобы отследить позже
            payments = load_payments()
            req_id = create_payment_record(user_id, tariff_key, "stars")
            # Сохраняем связь payload -> req_id
            payments = load_payments()
            payments_key = f"payload:{payload}"
            payments[payments_key] = {"req_id": req_id}
            save_payments(payments)

            bot.send_invoice(
                chat_id=call.message.chat.id,
                title=title,
                description=description,
                invoice_payload=payload,
                provider_token=PROVIDER_TOKEN,
                currency="XTR",
                prices=prices,
                start_parameter=f"start-{req_id}"
            )
            bot.answer_callback_query(call.id)
        except Exception as e:
            bot.answer_callback_query(call.id, "Ошибка создания инвойса. Попробуйте позже.", show_alert=True)
            logger.exception(e)
        return


@bot.pre_checkout_query_handler(func=lambda query: True)
def precheckout(pre_checkout_query):
    """
    Подтверждаем pre-checkout запрос — всегда разрешаем.
    """
    try:
        bot.answer_pre_checkout_query(pre_checkout_query.id, ok=True)
    except Exception as e:
        logger.exception(e)


@bot.message_handler(content_types=['successful_payment'])
def handle_successful_payment(message: teletypes.Message):
    """
    Обработка успешной оплаты через Telegram Payments (Stars, currency XTR).
    """
    try:
        sp = message.successful_payment  # SuccessfulPayment
        payload = sp.invoice_payload  # payload, который мы передали
        # Ищем нашу запись по payload
        payments = load_payments()
        payments_key = f"payload:{payload}"
        payload_rec = payments.get(payments_key)
        if not payload_rec:
            # На всякий случай: пытаемся найти по другому способу
            # payload format: "stars|user_id|tariff_key|uuid"
            parts = payload.split("|")
            if len(parts) >= 3:
                _, user_id, tariff_key = parts[:3]
            else:
                bot.send_message(message.chat.id, "Оплата принята, но не удалось найти тариф. Свяжитесь с поддержкой.")
                logger.warning(f"Не найден payload_rec для payload={payload}")
                return
            # создаём запись платежа вручную
            req_id = create_payment_record(str(message.from_user.id), tariff_key, "stars")
        else:
            req_id = payload_rec.get("req_id")
            rec = load_payments().get(req_id)
            tariff_key = rec.get("tariff") if rec else None

        # Защита от повторной обработки
        payments_all = load_payments()
        rec = payments_all.get(req_id)
        if rec and rec.get("status") == "pending":
            # пометим как подтверждённую
            mark_payment_processed(req_id, "confirmed", admin_id=None, notes="paid_via_stars")
            apply_tariff_to_user(str(message.from_user.id), tariff_key)
            # Уведомим пользователя
            t = TARIFFS.get(tariff_key)
            amount_replies = t.get("replies") if not t.get("duration_days") else f"безлимит на {t['duration_days']} дней"
            bot.send_message(message.chat.id, f"🎉 Оплата прошла успешно! Вам начислено: {amount_replies}")
        else:
            # уже обработано
            bot.send_message(message.chat.id, "ℹ️ Эта оплата уже обработана ранее.")
    except Exception as e:
        logger.exception(e)
        try:
            bot.send_message(message.chat.id, "❌ Ошибка обработки платежа. Свяжитесь с поддержкой.")
        except Exception:
            pass


@bot.callback_query_handler(func=lambda call: call.data and call.data.startswith("kaspi_paid:"))
def callback_kaspi_paid(call: teletypes.CallbackQuery):
    req_id = call.data.split(":", 1)[1]
    payments = load_payments()
    rec = payments.get(req_id)
    if not rec:
        bot.answer_callback_query(call.id, "Заявка не найдена.", show_alert=True)
        return
    if rec.get("status") != "pending":
        bot.answer_callback_query(call.id, "Эта заявка уже обработана.", show_alert=True)
        return

    # Помечаем, что от этого пользователя ожидается фото чека для конкретной заявки
    user_id = str(call.from_user.id)
    KASPI_WAITING[user_id] = req_id
    bot.answer_callback_query(call.id)
    bot.send_message(call.message.chat.id, "Отправьте одно изображение (скриншот чека). После отправки изображение будет переслано администратору для проверки.")


@bot.message_handler(content_types=['photo'])
def handle_kaspi_receipt_photo(message: teletypes.Message):
    user_id = str(message.from_user.id)
    if user_id not in KASPI_WAITING:
        # это не чек-ответ
        return

    req_id = KASPI_WAITING.pop(user_id)
    payments = load_payments()
    rec = payments.get(req_id)
    if not rec:
        bot.send_message(message.chat.id, "Заявка не найдена. Попробуйте ещё раз.")
        return
    if rec.get("status") != "pending":
        bot.send_message(message.chat.id, "Эта заявка уже обработана.")
        return

    # Скачиваем фото (берём самый большой вариант)
    try:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded = bot.download_file(file_info.file_path)
        local_path = f"receipts/{req_id}.jpg"
        os.makedirs("receipts", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(downloaded)
    except Exception as e:
        logger.exception(e)
        bot.send_message(message.chat.id, "❌ Не удалось сохранить изображение. Попробуйте ещё раз.")
        return

    # Пересылаем админу: картинка + текст с информацией + inline-кнопки Подтвердить/Отклонить
    tariff_key = rec.get("tariff")
    t = TARIFFS.get(tariff_key)
    caption = (
        f"Пользователь:\n\n"
        f"Username: @{message.from_user.username if message.from_user.username else '---'}\n"
        f"ID: {user_id}\n\n"
        f"Хочет приобрести:\n\n"
        f"{t['title']}\n\n"
        f"Стоимость:\n{t['price_tenge']} ₸\n\n"
        "Чек выше."
    )

    kb = teletypes.InlineKeyboardMarkup()
    kb.add(
        teletypes.InlineKeyboardButton("✅ Подтвердить и начислить", callback_data=f"admin_confirm:{req_id}"),
        teletypes.InlineKeyboardButton("❌ Отклонить", callback_data=f"admin_reject:{req_id}")
    )

    try:
        with open(local_path, "rb") as photo:
            bot.send_photo(int(ADMIN_ID), photo, caption=caption, reply_markup=kb)
    except Exception as e:
        logger.exception(e)
        bot.send_message(message.chat.id, "❌ Не удалось отправить чек админу. Попробуйте позже.")
        return

    # Подтвердим пользователю, что чек отправлен
    bot.send_message(message.chat.id, "✅ Чек отправлен на проверку администратору. Ожидайте подтверждения.")


@bot.callback_query_handler(func=lambda call: call.data and (call.data.startswith("admin_confirm:") or call.data.startswith("admin_reject:")))
def callback_admin_action(call: teletypes.CallbackQuery):
    if call.from_user.id != int(ADMIN_ID):
        bot.answer_callback_query(call.id, "У вас нет прав для этого действия.", show_alert=True)
        return

    data = call.data
    action, req_id = data.split(":", 1)
    payments = load_payments()
    rec = payments.get(req_id)
    if not rec:
        bot.answer_callback_query(call.id, "Заявка не найдена.", show_alert=True)
        return

    if rec.get("status") != "pending":
        bot.answer_callback_query(call.id, "Заявка уже обработана ранее.", show_alert=True)
        return

    user_id = rec.get("user_id")
    tariff_key = rec.get("tariff")

    if action == "admin_confirm":
        ok = mark_payment_processed(req_id, "confirmed", admin_id=call.from_user.id, notes="confirmed_by_admin")
        if not ok:
            bot.answer_callback_query(call.id, "Не удалось пометить заявку как обработанную.", show_alert=True)
            return
        try:
            apply_tariff_to_user(user_id, tariff_key)
            # уведомляем пользователя
            t = TARIFFS.get(tariff_key)
            amount_replies = t.get("replies") if not t.get("duration_days") else f"безлимит на {t['duration_days']} дней"
            bot.send_message(int(user_id), f"🎉 Ваша оплата подтверждена! Вам начислено: {amount_replies}")
            bot.answer_callback_query(call.id, "✅ Пользователь уведомлён и начисление выполнено.")
        except Exception as e:
            logger.exception(e)
            bot.answer_callback_query(call.id, "Ошибка начисления. См. логи.", show_alert=True)
            return

    elif action == "admin_reject":
        ok = mark_payment_processed(req_id, "rejected", admin_id=call.from_user.id, notes="rejected_by_admin")
        if not ok:
            bot.answer_callback_query(call.id, "Не удалось пометить заявку как обработанную.", show_alert=True)
            return
        # уведомляем пользователя
        try:
            bot.send_message(int(user_id), "❌ Оплата не подтверждена. Если произошла ошибка, свяжитесь с поддержкой.")
            bot.answer_callback_query(call.id, "❌ Заявка отклонена и пользователь уведомлён.")
        except Exception as e:
            logger.exception(e)
            bot.answer_callback_query(call.id, "Заявка отклонена, но не удалось уведомить пользователя.", show_alert=True)


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
        "Отправь описание вакансии.\n\n"
        "💰 Нужно больше? Используй /buy для покупки тарифа."
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
                "⚡ <b>Вы использовали все бесплатные письма!</b>\n\n"
                "Но не спешите расстраиваться! 🚀\n\n"
                "✨ <b>Откройте PRO-версию</b> и получите:\n"
                "• Улучшенные письма\n"
                "• Улучшенный AI\n"
                "• Приоритетная обработка\n\n"
                "💰 <b>Доступные тарифы:</b>\n"
                "• ⚡️ СТАРТ — 15 откликов за 990₸\n"
                "• 🚀 ВЫГОДНЫЙ — 50 откликов за 1990₸\n"
                "• 👑 БЕЗЛИМИТ — 30 дней за 3990₸\n\n"
                f"💳 Нажми /buy или напиши: {ADMIN_USERNAME}"
            ),
            parse_mode="HTML"
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

        cleaned_result = clean_markdown(result).strip()

        # Проверка: письмо должно быть содержательным (минимум 100 символов)
        if len(cleaned_result) < 100:
            logger.warning(f"Письмо слишком короткое ({len(cleaned_result)} символов): {cleaned_result[:50]}")
            bot.reply_to(
                message,
                (
                    "❌ Не удалось получить "
                    "полный ответ от Gemini.\n"
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
            cleaned_result
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
    
    if not PROVIDER_TOKEN or PROVIDER_TOKEN == "PROVIDER_TOKEN_FROM_BOTFATHER":
        logger.warning("⚠️  PROVIDER_TOKEN не установлен. Оплата через Telegram Stars не будет работать.")
    
    # Устанавливаем webhook перед запуском сервера
    if set_webhook():
        logger.info("✅ Инициализация успешна. Запуск веб-сервера...")
    else:
        logger.error("❌ Не удалось установить webhook. Проверьте WEBHOOK_URL и BOT_TOKEN.")
        exit(1)
    
    # Запускаем Flask
    run_web_server()
