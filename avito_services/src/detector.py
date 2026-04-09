"""
Модуль детекции микрокатегорий в тексте объявления на основе ML-классификатора.
Использует TF-IDF векторизацию и логистическую регрессию для мультилейбл классификации.
"""

import re
import pickle
import os
from typing import Set, List
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

from .microcatalog import MICROCATALOG, MC_DICT, get_all_mc_ids


# Пути для сохранения модели
MODEL_DIR = Path(__file__).parent / "ml_models"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
CLASSIFIER_PATH = MODEL_DIR / "ml_classifier.pkl"
MLB_PATH = MODEL_DIR / "multi_label_binarizer.pkl"


class MLCategoryDetector:
    """ML-детектор микрокатегорий на основе текста объявления."""
    
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.mlb = None
        self.all_mc_ids = sorted(get_all_mc_ids())
        self.is_loaded = False
        
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста для обработки."""
        text = text.lower()
        text = re.sub(r'[^а-яёa-z0-9\s/]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def train(self, texts: List[str], labels: List[Set[int]], save_model: bool = True):
        # Нормализуем тексты
        normalized_texts = [self._normalize_text(t) for t in texts]
        
        # Создаем векторизатор
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b'
        )
        
        X = self.vectorizer.fit_transform(normalized_texts)
        
        # Преобразуем лейблы в бинарный формат
        self.mlb = MultiLabelBinarizer(classes=self.all_mc_ids)
        y = self.mlb.fit_transform(labels)
        
        # Создаем и обучаем классификатор
        base_classifier = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        )
        
        self.classifier = OneVsRestClassifier(base_classifier, n_jobs=-1)
        self.classifier.fit(X, y)
        
        self.is_loaded = True
        
        if save_model:
            self.save_model()
        
        return self
    
    def predict(self, description: str, threshold: float = 0.3) -> Set[int]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        normalized = self._normalize_text(description)
        X = self.vectorizer.transform([normalized])
        
        # Для OneVsRestClassifier predict_proba возвращает массив (n_samples, n_classes)
        probabilities = self.classifier.predict_proba(X)[0]
        
        detected_ids = set()
        for i, prob in enumerate(probabilities):
            # prob - это скаляр для каждого класса в OneVsRestClassifier
            if prob >= threshold:
                detected_ids.add(self.all_mc_ids[i])
        
        return detected_ids
    
    def save_model(self):
        MODEL_DIR.mkdir(exist_ok=True)
        
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(CLASSIFIER_PATH, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        with open(MLB_PATH, 'wb') as f:
            pickle.dump(self.mlb, f)
    
    def load_model(self, model_dir: str = None) -> bool:
        if model_dir:
            base_dir = Path(model_dir)
        else:
            base_dir = MODEL_DIR
        
        vec_path = base_dir / "tfidf_vectorizer.pkl"
        clf_path = base_dir / "ml_classifier.pkl"
        mlb_path = base_dir / "multi_label_binarizer.pkl"
        
        if not all(p.exists() for p in [vec_path, clf_path, mlb_path]):
            return False
        
        with open(vec_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(clf_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        with open(mlb_path, 'rb') as f:
            self.mlb = pickle.load(f)
        
        self.is_loaded = True
        return True


_ml_detector = None


def get_ml_detector() -> MLCategoryDetector:
    global _ml_detector
    
    if _ml_detector is None:
        _ml_detector = MLCategoryDetector()
        if not _ml_detector.load_model():
            print("Warning: Could not load saved model.")
    
    return _ml_detector


def detect_microcategories(description: str, use_ml: bool = True, threshold: float = 0.3) -> Set[int]:
    if use_ml:
        detector = get_ml_detector()
        if detector.is_loaded:
            return detector.predict(description, threshold=threshold)
    
    return _heuristic_detect(description)


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _heuristic_detect(description: str) -> Set[int]:
    normalized = normalize_text(description)
    detected_ids = set()
    
    priority_phrases = {
        101: ["ремонт под ключ", "комплексный ремонт", "полный ремонт", "комплекс работ", "генподряд"],
        102: ["сантехника", "сантехнические работы", "монтаж сантехники", "разводка труб", "опрессовка труб"],
        103: ["электрика", "электромонтажные работы", "замена проводки", "электрощит"],
        104: ["натяжные потолки", "натяжной потолок", "тканевые потолки", "пвх потолок"],
        105: ["укладка плитки", "плиточные работы", "облицовка плиткой", "укладка керамогранита"],
        106: ["поклейка обоев", "оклейка обоями", "клеим обои", "наклейка обоев"],
        107: ["малярные работы", "шпаклевка", "шпатлевка", "покраска стен", "покраска потолка"],
        108: ["штукатурка", "штукатурные работы", "механизированная штукатурка", "цементная штукатурка"],
        109: ["напольные покрытия", "укладка ламината", "укладка паркета", "стяжка пола", "укладка линолеума"],
        110: ["гипсокартон", "гкл", "монтаж гипсокартона", "перегородки из гипсокартона"],
        111: ["демонтаж", "демонтажные работы", "снос стен", "упаковка строительного мусора"],
    }
    
    for mc_id, phrases in priority_phrases.items():
        for phrase in phrases:
            if phrase in normalized:
                detected_ids.add(mc_id)
                break
    
    for mc in MICROCATALOG:
        if mc.mcId in detected_ids:
            continue
            
        skip_general = {
            102: {"трубы", "труб", "ванна", "ванную", "унитаз", "смеситель"},
            103: {"розетка", "розеток", "выключатель", "люстра", "свет", "кабель"},
            104: {"потолок", "потолки", "потолков"},
            105: {"плитка", "плитку", "затирка", "облицовка"},
            106: {"обои", "обоями", "обоев"},
            107: {"краска", "красим", "грунтовка", "шлифовка"},
            108: {"выравнивание", "маяки", "заделка"},
            109: {"пол", "пола", "ламинат", "паркет", "плинтус"},
            110: {"профиль", "короб", "ниша", "арка", "перегородка"},
            111: {"снятие", "разборка", "мусор", "вывоз", "очистка"},
        }
        
        for phrase in mc.keyPhrases:
            norm_phrase = normalize_text(phrase)
            if len(norm_phrase) < 5:
                continue
            if mc.mcId in skip_general and norm_phrase in skip_general[mc.mcId]:
                continue
            if mc.mcId in [102, 103, 104, 105, 106, 107, 109, 110, 111] and len(norm_phrase.split()) < 2:
                specific_terms = {
                    102: ["бойлер", "водонагреватель", "полотенцесушитель", "канализация", "водоснабжение"],
                    103: ["штробление", "электромонтаж", "проводка"],
                    104: ["натяжной", "натяжных"],
                    105: ["керамогранит", "мозаика", "кафель"],
                    106: ["стеклообои", "флизелиновые", "виниловые", "фотообои"],
                    107: ["малярка", "короед", "венецианская"],
                    109: ["ковролин", "линолеум", "пробка", "кварцвинил", "наливной"],
                    110: ["гкл", "гвл", "гипрок"],
                    111: ["демонтаж", "перфоратор"],
                }
                if mc.mcId in specific_terms and norm_phrase not in specific_terms[mc.mcId]:
                    continue
                    
            if norm_phrase in normalized:
                detected_ids.add(mc.mcId)
                break
    
    if "/" in description:
        parts = description.split("/")
        for part in parts:
            part_normalized = normalize_text(part)
            for mc in MICROCATALOG:
                if mc.mcId in detected_ids:
                    continue
                for phrase in mc.keyPhrases[:5]:
                    if normalize_text(phrase) in part_normalized:
                        detected_ids.add(mc.mcId)
                        break
    
    return detected_ids


def train_ml_model(data_path: str = None, save_model: bool = True) -> MLCategoryDetector:
    import pandas as pd
    import ast
    
    if data_path is None:
        possible_paths = [
            Path(__file__).parent.parent / "data" / "rnc_dataset.csv",
            Path(__file__).parent / "data" / "rnc_dataset.csv",
            Path("data") / "rnc_dataset.csv",
        ]
        for path in possible_paths:
            if path.exists():
                data_path = str(path)
                break
        
        if data_path is None:
            raise FileNotFoundError("Dataset file not found.")
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    texts = df['description'].tolist()
    labels = []
    for idx, row in df.iterrows():
        target_ids = row['targetDetectedMcIds']
        if isinstance(target_ids, str):
            target_ids = ast.literal_eval(target_ids)
        labels.append(set(target_ids))
    
    print(f"Training model on {len(texts)} examples...")
    detector = MLCategoryDetector()
    detector.train(texts, labels, save_model=save_model)
    
    # Явно сохраняем модель после обучения
    if save_model:
        detector.save_model()
        print(f"Model saved to {MODEL_DIR}")
    
    print("Model trained successfully!")
    return detector


def normalize_text(text: str) -> str:
    """Нормализация текста: приведение к нижнему регистру, удаление лишних пробелов."""
    return _normalize_text(text)
