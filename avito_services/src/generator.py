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
    
    Args:
        description: Исходный текст объявления
        mc_id: ID микрокатегории для черновика
        source_mc_id: ID исходной микрокатегории объявления
        
    Returns:
        Сгенерированный текст черновика
    """
    mc = MC_DICT[mc_id]
    normalized = normalize_text(description)
    
    # Извлекаем релевантные фразы и ключевые слова для данной микрокатегории
    relevant_parts = []
    found_phrases = set()
    
    # Находим все ключевые фразы микрокатегории в тексте
    for phrase in mc.keyPhrases:
        norm_phrase = normalize_text(phrase)
        if norm_phrase in normalized and norm_phrase not in found_phrases:
            found_phrases.add(norm_phrase)
    
    # Если нашли ключевые фразы - используем их
    if found_phrases:
        # Формируем текст из найденных фраз
        phrases_list = list(found_phrases)[:5]  # Берём до 5 фраз
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
