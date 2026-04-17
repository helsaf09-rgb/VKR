from __future__ import annotations

CATEGORY_LABELS_RU: dict[str, str] = {
    "groceries": "Супермаркеты",
    "restaurants": "Рестораны",
    "transport": "Транспорт",
    "travel": "Путешествия",
    "utilities": "Коммунальные услуги",
    "healthcare": "Здоровье",
    "education": "Образование",
    "entertainment": "Развлечения",
    "electronics": "Электроника",
    "fashion": "Одежда и стиль",
    "home": "Дом",
    "investments": "Инвестиции",
    "insurance": "Страхование",
    "cash_withdrawal": "Снятие наличных",
    "money_transfer": "Переводы",
}

SEGMENT_LABELS_RU: dict[str, str] = {
    "daily_life": "Повседневные траты",
    "traveler": "Путешественник",
    "family": "Семья",
    "digital_pro": "Цифровой клиент",
    "investor": "Инвестор",
    "student": "Студент",
}

PRODUCT_TYPE_LABELS_RU: dict[str, str] = {
    "card": "Карта",
    "investment": "Инвестпродукт",
    "credit": "Кредит",
    "insurance": "Страхование",
    "deposit": "Вклад",
    "partner": "Партнерский сервис",
    "subscription": "Подписка",
    "service": "Сервис",
    "bundle": "Пакет",
}

MODEL_LABELS_RU: dict[str, str] = {
    "time_decay": "Time-Decay",
    "profile_baseline": "Профильный baseline",
    "hybrid_semantic": "Гибридная semantic-модель",
    "implicit_mf": "Implicit MF",
    "neural_cf": "Neural CF",
    "sasrec": "SASRec",
    "lightgcn": "LightGCN",
    "item_knn": "Item-kNN",
    "popularity": "Popularity",
}

CHANNEL_LABELS_RU: dict[str, str] = {
    "card": "Карта",
    "online": "Онлайн",
    "mobile": "Мобильный банк",
    "atm": "Банкомат",
    "transfer": "Перевод",
}

OFFER_LABELS_RU: dict[str, dict[str, str]] = {
    "O001": {
        "offer_name": "Карта с кэшбэком на ежедневные траты",
        "description": "Кэшбэк за повседневные покупки с бонусами на еду и городской транспорт.",
    },
    "O002": {
        "offer_name": "Премиальная карта путешественника",
        "description": "Мили и преимущества в поездках для клиентов с выраженными travel- и transport-тратами.",
    },
    "O003": {
        "offer_name": "Стартовый инвестиционный счет",
        "description": "Базовый инвестиционный продукт с обучающим треком и простым входом в инвестиции.",
    },
    "O004": {
        "offer_name": "Гибкий потребительский кредит",
        "description": "Нецелевой кредит с гибким графиком для крупных бытовых и lifestyle-покупок.",
    },
    "O005": {
        "offer_name": "Программа рефинансирования ипотеки",
        "description": "Предложение для клиентов со стабильными расходами на дом, ЖКУ и страхование.",
    },
    "O006": {
        "offer_name": "Семейный страховой пакет",
        "description": "Комплексная защита здоровья и семьи с цифровым урегулированием страховых случаев.",
    },
    "O007": {
        "offer_name": "Защита авто и мобильности",
        "description": "Страховое решение для клиентов с высокой транспортной активностью и поездками.",
    },
    "O008": {
        "offer_name": "Семейный накопительный вклад",
        "description": "Вклад для клиентов со стабильным ежемесячным бытовым бюджетом и целями накопления.",
    },
    "O009": {
        "offer_name": "Образовательная подписка",
        "description": "Скидочный доступ к образовательным сервисам для цифрового обучения и апскиллинга.",
    },
    "O010": {
        "offer_name": "Премиальная lifestyle-подписка",
        "description": "Подписка с партнерскими бонусами в ресторанах, развлечениях и lifestyle-сервисах.",
    },
    "O011": {
        "offer_name": "Кэшбэк за автоплатежи ЖКУ",
        "description": "Карта с бонусами за коммунальные платежи и регулярные списания.",
    },
    "O012": {
        "offer_name": "Пакет цифровой безопасности",
        "description": "Мониторинг мошенничества и защита операций для активно цифровых клиентов.",
    },
    "O013": {
        "offer_name": "Студенческая стартовая карта",
        "description": "Карта для студентов со скидками на образование, транспорт и цифровые сервисы.",
    },
    "O014": {
        "offer_name": "Пакет управления наличными",
        "description": "Оптимизация комиссий для клиентов с частым снятием наличных и переводами.",
    },
    "O015": {
        "offer_name": "Сбалансированный финансовый пакет",
        "description": "Пакет для финансово дисциплинированных клиентов, сочетающий стабильность и защиту.",
    },
}


def translate_category(category: str) -> str:
    return CATEGORY_LABELS_RU.get(category, category.replace("_", " "))


def translate_segment(segment: str) -> str:
    return SEGMENT_LABELS_RU.get(segment, segment.replace("_", " ").title())


def translate_product_type(product_type: str) -> str:
    return PRODUCT_TYPE_LABELS_RU.get(product_type, product_type.replace("_", " ").title())


def translate_model(model: str) -> str:
    return MODEL_LABELS_RU.get(model, model.replace("_", " ").title())


def translate_channel(channel: str) -> str:
    return CHANNEL_LABELS_RU.get(channel, channel.replace("_", " ").title())


def translate_offer(offer_id: str, offer_name: str, description: str) -> tuple[str, str]:
    localized = OFFER_LABELS_RU.get(offer_id, {})
    return localized.get("offer_name", offer_name), localized.get("description", description)
