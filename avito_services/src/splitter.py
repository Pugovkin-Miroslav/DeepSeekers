"""
Модуль определения необходимости сплита (создания дополнительных черновиков).
"""

import re
from typing import Tuple, Set

from .detector import normalize_text
from .microcatalog import MC_DICT


def is_service_offered_separately(description: str, mc_id: int, source_mc_id: int) -> bool:
    """
    Определяет, предлагается ли услуга данной микрокатегории отдельно.
    
    Args:
        description: Текст объявления
        mc_id: ID микрокатегории
        source_mc_id: ID исходной микрокатегории объявления
        
    Returns:
        True, если услуга предлагается отдельно
    """
    normalized = normalize_text(description)
    
    # Ключевые индикаторы отдельного предложения услуги
    separate_indicators = [
        "отдельно", "как самостоятельную работу", "как самостоятельную услугу",
        "можно заказать отдельно", "берем отдельно", "делаем отдельно",
        "выполняем отдельно", "предлагаем отдельно", "заказывайте отдельно",
    ]
    
    has_separate = any(indicator in normalized for indicator in separate_indicators)
    
    # Проверяем контекст вокруг ключевых фраз микрокатегории
    mc = MC_DICT.get(mc_id)
    if not mc:
        return False
    
    # Ищем паттерны с "отдельно" рядом с ключевыми фразами
    for phrase in mc.keyPhrases[:10]:  # Проверяем первые 10 фраз
        norm_phrase = normalize_text(phrase)
        
        # Паттерн: "отдельно" + фраза или фраза + "отдельно"
        pattern1 = rf"отдельно\s+(?:\w+\s+)?{re.escape(norm_phrase)}"
        pattern2 = rf"{re.escape(norm_phrase)}(?:\s+\w+)?\s+отдельно"
        
        if re.search(pattern1, normalized) or re.search(pattern2, normalized):
            return True
        
        # Паттерн: "можем/делаем/выполняем" + фраза
        pattern3 = rf"(?:можем|делаем|выполняем)\s+{re.escape(norm_phrase)}"
        if re.search(pattern3, normalized):
            return True
    
    # Если это исходная микрокатегория и нет явных признаков отдельной услуги - не сплитовать
    if mc_id == source_mc_id:
        return False
    
    # Для non-turnkey категорий (не 101) - если услуга найдена и не является частью комплекса
    if mc_id != 101 and source_mc_id != 101:
        # Проверяем, не является ли упоминание частью перечисления в комплексе
        if "как часть ремонта" in normalized or "в составе" in normalized:
            return False
        # Если есть явное перечисление через "/" или запятую - может быть отдельной
        if "/" in description or "также" in normalized:
            return True
    
    return has_separate


def should_split_announcement(
    description: str,
    detected_mc_ids: Set[int],
    source_mc_id: int
) -> Tuple[bool, Set[int]]:
    """
    Определяет, нужно ли создавать дополнительные черновики объявлений.
    
    Args:
        description: Текст объявления
        detected_mc_ids: Множество ID обнаруженных микрокатегорий
        source_mc_id: ID исходной микрокатегории объявления
        
    Returns:
        Кортеж (shouldSplit, split_mc_ids)
    """
    normalized = normalize_text(description)
    
    # Исходная микрокатегория не включается в сплит
    candidate_ids = detected_mc_ids - {source_mc_id}
    
    if not candidate_ids:
        return False, set()
    
    # Проверяем явные индикаторы работы только в комплексе
    complex_only_phrases = [
        "по отдельным видам работ не выезжаю",
        "ищу заказы именно на комплекс",
        "работаем только в комплексе",
        "без дробления на этапы",
        "под ключ без дробления",
        "не выезжаю отдельно",
    ]
    
    is_complex_only = any(phrase in normalized for phrase in complex_only_phrases)
    
    # Если явно указано, что работают только в комплексе - НЕ сплитим
    if is_complex_only:
        return False, set()
    
    # Определяем, какие микрокатегории предлагаются отдельно
    split_ids = set()
    for mc_id in candidate_ids:
        # Для non-turnkey категорий - если услуга найдена, считаем её отдельной
        # если нет явных признаков, что она только в составе комплекса
        if mc_id == 101:  # Ремонт под ключ - особая категория
            if is_service_offered_separately(description, mc_id, source_mc_id):
                split_ids.add(mc_id)
        else:
            # Проверяем, не является ли упоминание частью перечисления в комплексе
            if "как часть ремонта" in normalized or "в составе работ" in normalized or "все этапы выполняем как часть ремонта" in normalized:
                # Но если есть явное указание на отдельную услугу - всё равно сплитим
                if is_service_offered_separately(description, mc_id, source_mc_id):
                    split_ids.add(mc_id)
            else:
                # По умолчанию считаем, что услуга предлагается отдельно
                split_ids.add(mc_id)
    
    # Если есть хотя бы одна микрокатегория для сплита
    if split_ids:
        return True, split_ids
    
    return False, set()


def determine_should_split(detected_mc_ids: Set[int], source_mc_id: int, has_additional_services: bool = True) -> bool:
    """
    Упрощенная функция определения необходимости сплита.
    Если найдены дополнительные микрокатегории кроме исходной - нужен сплит.
    
    Args:
        detected_mc_ids: Множество ID обнаруженных микрокатегорий
        source_mc_id: ID исходной микрокатегории
        has_additional_services: Флаг наличия дополнительных услуг
        
    Returns:
        True, если нужно создавать дополнительные черновики
    """
    # Если есть дополнительные микрокатегории кроме исходной
    additional_ids = detected_mc_ids - {source_mc_id}
    if additional_ids:
        return True
    return False
