"""
Модуль генерации текстов черновиков объявлений.
"""

import re
from typing import List, Dict, Any

from .detector import normalize_text
from .microcatalog import MC_DICT


def generate_draft_text(description: str, mc_id: int, source_mc_id: int) -> str:
    """
    Генерирует текст черновика объявления для указанной микрокатегории.
    Формирует персонализированный текст только с фразами, релевантными данной МК.
    
    Args:
        description: Исходный текст объявления
        mc_id: ID микрокатегории для черновика
        source_mc_id: ID исходной микрокатегории объявления
        
    Returns:
        Сгенерированный текст черновика
    """
    import re
    
    mc = MC_DICT[mc_id]
    normalized = normalize_text(description)
    original_lower = description.lower()
    
    # Извлекаем релевантные фразы и ключевые слова ТОЛЬКО для данной микрокатегории
    found_phrases = set()
    
    # Находим все ключевые фразы микрокатегории в тексте
    for phrase in mc.keyPhrases:
        norm_phrase = normalize_text(phrase)
        if len(norm_phrase) >= 3 and norm_phrase in normalized:
            # Проверяем, что фраза найдена именно в контексте этой услуги
            # Ищем оригинальную фразу в тексте (с сохранением регистра для точности)
            if phrase.lower() in original_lower or norm_phrase in normalized:
                found_phrases.add(norm_phrase)
    
    # Также ищем специфичные паттерны ТОЛЬКО для этой микрокатегории
    specific_patterns = {
        101: [r"ремонт\s+под\s+ключ", r"комплексный\s+ремонт", r"полный\s+ремонт", r"комплекс\s+работ"],
        102: [r"сантехник", r"разводка\s+труб", r"монтаж\s+сантехники", r"демонтаж\s+сантехники", r"монтаж\s+бойлера", r"бойлер"],
        103: [r"электрик", r"проводка", r"электрощит", r"электромонтаж", r"штробление"],
        104: [r"натяжной\s+потолок", r"потолок\s+в\s+коридоре", r"тканевые\s+потолки", r"пвх\s+потолок"],
        105: [r"плитк", r"керамогранит", r"облицовка\s+плиткой", r"укладка\s+плитки"],
        106: [r"обои", r"оклейка\s+обоями", r"поклейка\s+обоев", r"клеим\s+обои"],
        107: [r"шпаклевк", r"покраск", r"малярн", r"короед"],
        108: [r"штукатурк", r"цементная\s+штукатурка", r"механизированная\s+штукатурка"],
        109: [r"напольн", r"ламинат", r"паркет", r"стяжка", r"линолеум"],
        110: [r"гкл", r"гипсокартон", r"декоративные\s+конструкции\s+из\s+гкл", r"перегородки\s+из\s+гипсокартона", r"монтаж\s+перегородок"],
        111: [r"демонтаж", r"снос", r"упаковка\s+мусора", r"вывоз\s+мусора"],
    }
    
    if mc_id in specific_patterns:
        for pattern in specific_patterns[mc_id]:
            matches = re.findall(pattern, normalized, re.IGNORECASE)
            for match in matches:
                match_clean = normalize_text(match)
                if len(match_clean) >= 3:
                    found_phrases.add(match_clean)
    
    # Фильтруем найденные фразы - оставляем только те, что относятся к данной МК
    # и не являются слишком общими
    general_words = {"отдельно", "делаем", "выполняем", "можно", "работаем", "опыт", "лет", "звоните", "выезд"}
    filtered_phrases = {p for p in found_phrases if p not in general_words and len(p.split()) >= 1}
    
    # Если нашли ключевые фразы - используем их
    if filtered_phrases:
        # Формируем текст из найденных фраз
        phrases_list = sorted(list(filtered_phrases))[:5]  # Берём до 5 фраз, сортируем для детерминизма
        draft_text = ", ".join(phrases_list)
        if len(phrases_list) > 1:
            draft_text = draft_text + "."
        else:
            draft_text = draft_text + "."
    else:
        # Если не нашли конкретных фраз, используем общие ключевые фразы микрокатегории
        draft_text = ", ".join(mc.keyPhrases[:3]) + "."
    
    # Добавляем информацию о том, что услуга предоставляется отдельно
    # Формируем префикс с названием услуги
    service_name = mc.mcTitle.lower()
    draft_text = f"Отдельно выполняем услуги по направлению \"{service_name}\": {draft_text.lower()}"
    
    return draft_text


def create_draft(description: str, mc_id: int, source_mc_id: int) -> Dict[str, Any]:
    """
    Создает черновик объявления для указанной микрокатегории.
    
    Args:
        description: Исходный текст объявления
        mc_id: ID микрокатегории для черновика
        source_mc_id: ID исходной микрокатегории объявления
        
    Returns:
        Словарь с данными черновика
    """
    mc = MC_DICT.get(mc_id)
    if not mc:
        raise ValueError(f"Микрокатегория с ID {mc_id} не найдена")
    
    draft_text = generate_draft_text(description, mc_id, source_mc_id)
    
    return {
        "mcId": mc_id,
        "mcTitle": mc.mcTitle,
        "text": draft_text
    }


def create_drafts(description: str, split_mc_ids: set[int], source_mc_id: int) -> List[Dict[str, Any]]:
    """
    Создает список черновиков для указанных микрокатегорий.
    
    Args:
        description: Исходный текст объявления
        split_mc_ids: Множество ID микрокатегорий для создания черновиков
        source_mc_id: ID исходной микрокатегории объявления
        
    Returns:
        Список словарей с данными черновиков
    """
    drafts = []
    for mc_id in sorted(split_mc_ids):
        draft = create_draft(description, mc_id, source_mc_id)
        drafts.append(draft)
    
    return drafts
