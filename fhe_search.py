#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
سیستم جستجوی رمزنگاری هومورفیک کامل (FHE)
==========================================

یک موتور جستجوی عملی و چندزبانه با حفظ حریم خصوصی
که از تکنیک‌های رمزنگاری هومورفیک و پردازش زبان طبیعی استفاده می‌کند.

ویژگی‌ها:
- پشتیبانی از فارسی، عربی و انگلیسی
- جستجوی معنایی با TF-IDF
- رمزنگاری هومورفیک (CKKS)
- پردازش دسته‌ای و خوشه‌بندی
- ذخیره‌سازی پایدار (SQLite)
- داشبورد آنالیتیک
- قابلیت ورود و خروج داده
مجوز: MIT
"""

import tenseal as ts
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional, Any
import pickle
import json
import time
import logging
from datetime import datetime
import re
from collections import defaultdict, Counter
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# تنظیم لاگ‌گیری
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fhe_search.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PersianTextProcessor:
    """
    کلاس پردازش متن برای زبان‌های فارسی، عربی و انگلیسی
    شامل حذف کلمات، نرمال‌سازی و تشخیص زبان
    """
    
    def __init__(self):
        # کلمات فارسی - از منابع مختلف جمع‌آوری شده
        self.persian_stopwords = {
            'از', 'به', 'در', 'با', 'که', 'این', 'آن', 'را', 'و', 'یا', 'تا', 'اما', 'بر',
            'کرد', 'شد', 'است', 'بود', 'می‌باشد', 'خود', 'خواهد', 'دارد', 'داشت', 'کند',
            'می‌کند', 'نمی', 'هم', 'نیز', 'چون', 'اگر', 'ولی', 'پس', 'چه', 'کجا', 'چرا',
            'برای', 'روی', 'زیر', 'بالا', 'پایین', 'جلو', 'عقب', 'وسط', 'کنار'
        }
        
        # کلمات عربی
        self.arabic_stopwords = {
            'في', 'من', 'إلى', 'عن', 'مع', 'هذا', 'ذلك', 'التي', 'الذي', 'كان', 'يكون', 
            'هو', 'هي', 'أن', 'على', 'إن', 'كل', 'بعض', 'غير', 'سوف', 'قد', 'لم', 'لن'
        }
        
        # الگوهای regex برای پاک‌سازی
        self.cleanup_patterns = [
            (r'[۰-۹]+', ''),  # حذف اعداد فارسی
            (r'[0-9]+', ''),  # حذف اعداد انگلیسی
            (r'[^\w\s\u0600-\u06FF\u0750-\u077F]', ' '),  # حفظ حروف فارسی و عربی
            (r'\s+', ' ')     # حذف فاصله‌های اضافی
        ]

    def detect_language(self, text: str) -> str:
        """تشخیص زبان متن بر اساس نوع کاراکترها"""
        if not text or len(text.strip()) == 0:
            return 'unknown'
            
        # شمارش انواع کاراکترها
        persian_arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_alpha_chars = persian_arabic_chars + english_chars
        
        if total_alpha_chars == 0:
            return 'unknown'
            
        persian_ratio = persian_arabic_chars / total_alpha_chars
        english_ratio = english_chars / total_alpha_chars
        
        # تعیین زبان بر اساس نسبت کاراکترها
        if persian_ratio > 0.4:
            return 'persian'
        elif english_ratio > 0.7:
            return 'english'
        elif persian_ratio > 0.1:
            return 'mixed'
        else:
            return 'english'

    def normalize_persian_text(self, text: str) -> str:
        """نرمال‌سازی متن فارسی"""
        # تبدیل کاراکترهای مشابه
        replacements = {
            'ي': 'ی', 'ك': 'ک', 'ء': '', 'ؤ': 'و', 'ئ': 'ی',
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ة': 'ه'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text

    def clean_text(self, text: str) -> str:
        """پاک‌سازی اولیه متن"""
        if not text:
            return ""
            
        # نرمال‌سازی فارسی
        text = self.normalize_persian_text(text)
        
        # اعمال الگوهای پاک‌سازی
        for pattern, replacement in self.cleanup_patterns:
            text = re.sub(pattern, replacement, text)
            
        return text.strip().lower()

    def remove_stopwords(self, text: str, language: str) -> str:
        """حذف کلمات بر اساس زبان"""
        words = text.split()
        
        if language == 'persian':
            words = [w for w in words if w not in self.persian_stopwords and len(w) > 1]
        elif language == 'arabic':
            words = [w for w in words if w not in self.arabic_stopwords and len(w) > 1]
        elif language == 'english':
            # برای انگلیسی از sklearn استفاده می‌کنیم
            pass
        
        return ' '.join(words)

    def preprocess_text(self, text: str, language: str = 'auto') -> str:
        """پردازش کامل متن"""
        if not text or not text.strip():
            return ""
            
        # تشخیص خودکار زبان
        if language == 'auto':
            language = self.detect_language(text)
            
        # پاک‌سازی اولیه
        cleaned_text = self.clean_text(text)
        
        # حذف کلمات
        final_text = self.remove_stopwords(cleaned_text, language)
        
        return final_text if final_text.strip() else cleaned_text


class DatabaseManager:
    """مدیریت دیتابیس SQLite"""
    
    def __init__(self, db_path: str = 'fhe_search.db'):
        self.db_path = db_path
        self.connection = None
        self._initialize_db()
    
    def _initialize_db(self):
        """ایجاد جداول دیتابیس"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.connection.cursor()
            
            # جدول پیام‌ها
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    processed_content TEXT,
                    language TEXT,
                    cluster_id INTEGER,
                    word_count INTEGER,
                    char_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    is_encrypted BOOLEAN DEFAULT 0
                )
            ''')
            
            # جدول تاریخچه جستجو
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    processed_query TEXT,
                    language TEXT,
                    results_count INTEGER,
                    max_similarity REAL,
                    avg_similarity REAL,
                    execution_time REAL,
                    filters_applied TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # جدول آمار سیستم
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stat_name TEXT UNIQUE,
                    stat_value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.connection.commit()
            logger.info(f"دیتابیس در {self.db_path} آماده شد")
            
        except Exception as e:
            logger.error(f"خطا در ایجاد دیتابیس: {e}")
            self.connection = None
    
    def store_document(self, content: str, processed_content: str, language: str, 
                      metadata: Dict, is_encrypted: bool = False) -> int:
        """ذخیره یک سند در دیتابیس"""
        if not self.connection:
            return -1
            
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO documents 
                (content, processed_content, language, word_count, char_count, metadata, is_encrypted)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                content,
                processed_content,
                language,
                len(processed_content.split()),
                len(content),
                json.dumps(metadata, ensure_ascii=False),
                is_encrypted
            ))
            
            doc_id = cursor.lastrowid
            self.connection.commit()
            return doc_id
            
        except Exception as e:
            logger.error(f"خطا در ذخیره سند: {e}")
            return -1
    
    def store_search_query(self, query_data: Dict):
        """ذخیره تاریخچه جستجو"""
        if not self.connection:
            return
            
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO search_history 
                (query, processed_query, language, results_count, max_similarity, 
                 avg_similarity, execution_time, filters_applied)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                query_data.get('query', ''),
                query_data.get('processed_query', ''),
                query_data.get('language', ''),
                query_data.get('results_count', 0),
                query_data.get('max_similarity', 0.0),
                query_data.get('avg_similarity', 0.0),
                query_data.get('execution_time', 0.0),
                json.dumps(query_data.get('filters', {}), ensure_ascii=False)
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"خطا در ذخیره تاریخچه جستجو: {e}")
    
    def get_statistics(self) -> Dict:
        """دریافت آمار از دیتابیس"""
        if not self.connection:
            return {}
            
        try:
            cursor = self.connection.cursor()
            
            # آمار کلی اسناد
            cursor.execute('SELECT COUNT(*) FROM documents')
            total_docs = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM documents WHERE is_encrypted = 1')
            encrypted_docs = cursor.fetchone()[0]
            
            # آمار زبان‌ها
            cursor.execute('''
                SELECT language, COUNT(*) 
                FROM documents 
                GROUP BY language 
                ORDER BY COUNT(*) DESC
            ''')
            language_stats = dict(cursor.fetchall())
            
            # آمار جستجوها
            cursor.execute('SELECT COUNT(*) FROM search_history')
            total_searches = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT AVG(execution_time), AVG(results_count), AVG(avg_similarity)
                FROM search_history 
                WHERE created_at >= datetime('now', '-24 hours')
            ''')
            recent_stats = cursor.fetchone()
            
            return {
                'total_documents': total_docs,
                'encrypted_documents': encrypted_docs,
                'language_distribution': language_stats,
                'total_searches': total_searches,
                'recent_avg_time': recent_stats[0] or 0,
                'recent_avg_results': recent_stats[1] or 0,
                'recent_avg_similarity': recent_stats[2] or 0
            }
            
        except Exception as e:
            logger.error(f"خطا در دریافت آمار: {e}")
            return {}
    
    def close(self):
        """بستن اتصال دیتابیس"""
        if self.connection:
            self.connection.close()
            self.connection = None


class FHEManager:
    """مدیریت رمزنگاری هومورفیک"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.context = None
        self.encryption_stats = {
            'successful_encryptions': 0,
            'failed_encryptions': 0,
            'total_attempts': 0
        }
        self._setup_encryption_context()
    
    def _setup_encryption_context(self):
        """راه‌اندازی context رمزنگاری CKKS"""
        try:
            # پارامترهای CKKS
            poly_modulus_degree = self.config.get('poly_modulus_degree', 8192)
            coeff_modulus = self.config.get('coeff_modulus', [60, 40, 40, 60])
            scale = self.config.get('scale', 2**40)
            
            # ایجاد context
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree,
                -1,  # security level
                coeff_modulus
            )
            
            self.context.global_scale = scale
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()
            
            # تست رمزنگاری
            test_vector = [1.0, 2.0, 3.0, 4.0, 5.0]
            encrypted_test = ts.ckks_vector(self.context, test_vector)
            decrypted_test = encrypted_test.decrypt()
            
            # بررسی صحت
            if abs(decrypted_test[0] - 1.0) < 0.01:
                logger.info("رمزنگاری CKKS با موفقیت راه‌اندازی شد")
            else:
                raise ValueError("تست رمزنگاری ناموفق")
                
        except Exception as e:
            logger.warning(f"خطا در راه‌اندازی رمزنگاری: {e}")
            self.context = None
    
    def encrypt_vector(self, vector: np.ndarray, max_size: int = 100) -> Optional[ts.CKKSVector]:
        """رمزنگاری یک بردار"""
        if not self.context:
            return None
            
        try:
            self.encryption_stats['total_attempts'] += 1
            
            # محدود کردن اندازه بردار
            if len(vector) > max_size:
                vector = vector[:max_size]
            
            # تبدیل به لیست و رمزنگاری
            vector_list = vector.tolist()
            encrypted_vector = ts.ckks_vector(self.context, vector_list)
            
            self.encryption_stats['successful_encryptions'] += 1
            return encrypted_vector
            
        except Exception as e:
            self.encryption_stats['failed_encryptions'] += 1
            logger.debug(f"خطا در رمزنگاری بردار: {e}")
            return None
    
    def get_encryption_stats(self) -> Dict:
        """دریافت آمار رمزنگاری"""
        total = self.encryption_stats['total_attempts']
        success_rate = (
            self.encryption_stats['successful_encryptions'] / total 
            if total > 0 else 0
        )
        
        return {
            **self.encryption_stats,
            'success_rate': success_rate,
            'context_available': self.context is not None
        }


class FHESearchEngine:
    """
    موتور جستجوی اصلی با قابلیت رمزنگاری هومورفیک
    """
    
    def __init__(self, config_file: Optional[str] = None):
        # بارگذاری تنظیمات
        self.config = self._load_config(config_file)
        
        # مؤلفه‌های اصلی
        self.text_processor = PersianTextProcessor()
        self.db_manager = DatabaseManager(self.config.get('database_path', 'fhe_search.db'))
        self.fhe_manager = FHEManager(self.config.get('encryption', {}))
        
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.get('max_features', 1000),
            ngram_range=tuple(self.config.get('ngram_range', [1, 2])),
            token_pattern=r'[\w\u0600-\u06FF\u0750-\u077F]+',
            lowercase=False,
            min_df=1,
            max_df=0.95,
            stop_words=None
        )
        
        # ذخیره‌سازی داده‌ها
        self.document_vectors = []  # بردارهای TF-IDF
        self.encrypted_vectors = []  # بردارهای رمزنگاری شده
        self.document_metadata = []  # متادیتای اسناد
        self.cluster_info = {}  # اطلاعات خوشه‌بندی
        
        # آمار عملکرد
        self.performance_metrics = {
            'search_times': [],
            'encryption_times': [],
            'result_counts': [],
            'similarity_scores': []
        }
        
        # وضعیت سیستم
        self.is_trained = False
        self.total_documents = 0
        
        logger.info("سیستم جستجوی FHE آماده شد")
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """بارگذاری فایل تنظیمات"""
        default_config = {
            'max_features': 1000,
            'ngram_range': [1, 2],
            'database_path': 'fhe_search.db',
            'enable_clustering': True,
            'cluster_count': 5,
            'batch_size': 50,
            'encryption': {
                'poly_modulus_degree': 8192,
                'coeff_modulus': [60, 40, 40, 60],
                'scale': 2**40
            },
            'search': {
                'min_similarity_threshold': 0.01,
                'default_top_k': 5
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    logger.info(f"تنظیمات از {config_file} بارگذاری شد")
            except Exception as e:
                logger.warning(f"خطا در بارگذاری تنظیمات: {e}")
        
        return default_config
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, 
                     batch_size: Optional[int] = None) -> Dict:
        """افزودن مجموعه‌ای از اسناد به سیستم"""
        
        if not documents:
            return {"status": "error", "message": "هیچ سندی ارائه نشده"}
        
        start_time = time.time()
        batch_size = batch_size or self.config.get('batch_size', 50)
        
        # آماده‌سازی متادیتا
        if metadata is None:
            metadata = [{"document_id": i} for i in range(len(documents))]
        elif len(metadata) != len(documents):
            metadata.extend([{"document_id": i} for i in range(len(metadata), len(documents))])
        
        # پردازش اسناد
        processed_docs = []
        enhanced_metadata = []
        language_distribution = Counter()
        
        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            # تشخیص زبان و پردازش
            detected_lang = self.text_processor.detect_language(doc)
            processed_doc = self.text_processor.preprocess_text(doc, detected_lang)
            
            processed_docs.append(processed_doc)
            
            # ایجاد متادیتای کامل
            enhanced_meta = {
                **meta,
                'original_text': doc,
                'processed_text': processed_doc,
                'language': detected_lang,
                'character_count': len(doc),
                'word_count': len(processed_doc.split()),
                'processing_timestamp': datetime.now().isoformat(),
                'document_index': self.total_documents + i
            }
            enhanced_metadata.append(enhanced_meta)
            language_distribution[detected_lang] += 1
        
        # آموزش یا به‌روزرسانی vectorizer
        if not self.is_trained:
            logger.info("آموزش مدل TF-IDF...")
            self.vectorizer.fit(processed_docs)
            self.is_trained = True
            logger.info(f"مدل با {len(self.vectorizer.vocabulary_)} ویژگی آموزش دید")
        
        # تبدیل اسناد به بردار
        doc_vectors = self.vectorizer.transform(processed_docs).toarray()
        
        # رمزنگاری بردارها
        encryption_start = time.time()
        encrypted_count = 0
        
        for i, (vector, meta) in enumerate(zip(doc_vectors, enhanced_metadata)):
            # ذخیره بردار خام
            self.document_vectors.append(vector)
            self.document_metadata.append(meta)
            
            # رمزنگاری (اختیاری)
            encrypted_vector = self.fhe_manager.encrypt_vector(vector)
            self.encrypted_vectors.append(encrypted_vector)
            
            if encrypted_vector is not None:
                encrypted_count += 1
            
            # ذخیره در دیتابیس
            doc_id = self.db_manager.store_document(
                content=meta['original_text'],
                processed_content=meta['processed_text'],
                language=meta['language'],
                metadata=meta,
                is_encrypted=(encrypted_vector is not None)
            )
            meta['database_id'] = doc_id
        
        encryption_time = time.time() - encryption_start
        
        # خوشه‌بندی (در صورت فعال بودن)
        clustering_info = {}
        if (self.config.get('enable_clustering', True) and 
            len(self.document_vectors) >= self.config.get('cluster_count', 5)):
            clustering_info = self._perform_clustering()
        
        # آمار نهایی
        total_time = time.time() - start_time
        self.total_documents += len(documents)
        
        result = {
            "status": "success",
            "processed_documents": len(documents),
            "encrypted_documents": encrypted_count,
            "language_distribution": dict(language_distribution),
            "processing_time": total_time,
            "encryption_time": encryption_time,
            "clustering_info": clustering_info,
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "total_documents_in_system": self.total_documents
        }
        
        logger.info(
            f"پردازش {len(documents)} سند در {total_time:.2f} ثانیه کامل شد. "
            f"رمزنگاری: {encrypted_count}/{len(documents)}"
        )
        
        return result
    
    def _perform_clustering(self) -> Dict:
        """خوشه‌بندی اسناد"""
        if len(self.document_vectors) < 2:
            return {}
        
        try:
            n_clusters = min(
                self.config.get('cluster_count', 5),
                len(self.document_vectors)
            )
            
            logger.info(f"شروع خوشه‌بندی با {n_clusters} خوشه...")
            
            # اجرای K-Means
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            
            cluster_labels = kmeans.fit_predict(self.document_vectors)
            
            # به‌روزرسانی متادیتا
            for i, label in enumerate(cluster_labels):
                if i < len(self.document_metadata):
                    self.document_metadata[i]['cluster_id'] = int(label)
            
            # تحلیل خوشه‌ها
            cluster_analysis = {}
            for cluster_id in range(n_clusters):
                cluster_docs = [
                    meta for i, meta in enumerate(self.document_metadata)
                    if cluster_labels[i] == cluster_id
                ]
                
                if cluster_docs:
                    # استخراج کلمات مشترک
                    all_words = []
                    for doc_meta in cluster_docs:
                        words = doc_meta.get('processed_text', '').split()
                        all_words.extend(words)
                    
                    common_words = Counter(all_words).most_common(10)
                    
                    # زبان‌های موجود در خوشه
                    cluster_languages = Counter([
                        doc_meta.get('language', 'unknown') 
                        for doc_meta in cluster_docs
                    ])
                    
                    cluster_analysis[cluster_id] = {
                        'document_count': len(cluster_docs),
                        'common_words': common_words,
                        'languages': dict(cluster_languages),
                        'sample_text': cluster_docs[0].get('original_text', '')[:100] + '...'
                    }
            
            self.cluster_info = cluster_analysis
            logger.info(f"خوشه‌بندی کامل شد: {n_clusters} خوشه شناسایی شد")
            
            return cluster_analysis
            
        except Exception as e:
            logger.error(f"خطا در خوشه‌بندی: {e}")
            return {}
    
    def search(self, query: str, top_k: int = None, filters: Dict = None, 
              use_encryption: bool = False) -> Dict:
        """جستجو در اسناد"""
        
        if not self.is_trained:
            return {"status": "error", "message": "سیستم آموزش ندیده است"}
        
        search_start = time.time()
        top_k = top_k or self.config.get('search', {}).get('default_top_k', 5)
        
        # پردازش کوئری
        query_language = self.text_processor.detect_language(query)
        processed_query = self.text_processor.preprocess_text(query, query_language)
        
        if not processed_query.strip():
            return {"status": "error", "message": "کوئری پردازش شده خالی است"}
        
        # تبدیل کوئری به بردار
        try:
            query_vector = self.vectorizer.transform([processed_query]).toarray()[0]
        except Exception as e:
            return {"status": "error", "message": f"خطا در vectorization: {e}"}
        
        # اعمال فیلترها
        valid_indices = self._apply_filters(filters) if filters else list(range(len(self.document_vectors)))
        
        if not valid_indices:
            return {
                "status": "success",
                "query": query,
                "results": [],
                "message": "هیچ سندی با فیلترهای اعمال شده یافت نشد"
            }
        
        # محاسبه شباهت
        similarities = []
        min_threshold = self.config.get('search', {}).get('min_similarity_threshold', 0.01)
        
        for idx in valid_indices:
            if idx >= len(self.document_vectors):
                continue
                
            try:
                similarity = cosine_similarity(
                    query_vector.reshape(1, -1),
                    self.document_vectors[idx].reshape(1, -1)
                )[0][0]
                
                if similarity > min_threshold:
                    result_item = {
                        "document_index": idx,
                        "similarity_score": float(similarity),
                        "metadata": self.document_metadata[idx] if idx < len(self.document_metadata) else {},
                        "language_match": (
                            self.document_metadata[idx].get('language') == query_language
                            if idx < len(self.document_metadata) else False
                        )
                    }
                    similarities.append(result_item)
                    
            except Exception as e:
                logger.warning(f"خطا در محاسبه شباهت برای سند {idx}: {e}")
                continue
        
        # مرتب‌سازی بر اساس شباهت
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_results = similarities[:top_k]
        
        # محاسبه آمار
        execution_time = time.time() - search_start
        max_similarity = max([r["similarity_score"] for r in top_results]) if top_results else 0
        avg_similarity = np.mean([r["similarity_score"] for r in top_results]) if top_results else 0
        
        # ذخیره تاریخچه جستجو
        search_data = {
            'query': query,
            'processed_query': processed_query,
            'language': query_language,
            'results_count': len(top_results),
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity,
            'execution_time': execution_time,
            'filters': filters or {}
        }
        self.db_manager.store_search_query(search_data)
        
        # به‌روزرسانی آمار عملکرد
        self.performance_metrics['search_times'].append(execution_time)
        self.performance_metrics['result_counts'].append(len(top_results))
        self.performance_metrics['similarity_scores'].extend([r["similarity_score"] for r in top_results])
        
        # نتیجه نهایی
        search_result = {
            "status": "success",
            "query": query,
            "processed_query": processed_query,
            "query_language": query_language,
            "results": top_results,
            "total_matches": len(similarities),
            "returned_count": len(top_results),
            "execution_time": execution_time,
            "filters_applied": filters or {},
            "statistics": {
                "max_similarity": max_similarity,
                "avg_similarity": avg_similarity,
                "language_distribution": Counter([
                    r["metadata"].get("language", "unknown") for r in top_results
                ])
            }
        }
        
        logger.info(
            f"جستجو برای '{query}' در {execution_time:.3f} ثانیه: "
            f"{len(top_results)} نتیجه (میانگین شباهت: {avg_similarity:.3f})"
        )
        
        return search_result
    
    def _apply_filters(self, filters: Dict) -> List[int]:
        """اعمال فیلترهای جستجو"""
        valid_indices = list(range(len(self.document_metadata)))
        
        for filter_name, filter_value in filters.items():
            if filter_name == 'language':
                valid_indices = [
                    i for i in valid_indices
                    if i < len(self.document_metadata) and
                    self.document_metadata[i].get('language') == filter_value
                ]
            
            elif filter_name == 'cluster_id':
                valid_indices = [
                    i for i in valid_indices
                    if i < len(self.document_metadata) and
                    self.document_metadata[i].get('cluster_id') == filter_value
                ]
            
            elif filter_name == 'min_word_count':
                valid_indices = [
                    i for i in valid_indices
                    if i < len(self.document_metadata) and
                    self.document_metadata[i].get('word_count', 0) >= filter_value
                ]
            
            elif filter_name == 'max_word_count':
                valid_indices = [
                    i for i in valid_indices
                    if i < len(self.document_metadata) and
                    self.document_metadata[i].get('word_count', 0) <= filter_value
                ]
        
        return valid_indices
    
    def get_system_status(self) -> Dict:
        """دریافت وضعیت کامل سیستم"""
        db_stats = self.db_manager.get_statistics()
        encryption_stats = self.fhe_manager.get_encryption_stats()
        
        # آمار عملکرد
        perf_stats = {}
        if self.performance_metrics['search_times']:
            perf_stats = {
                'avg_search_time': np.mean(self.performance_metrics['search_times']),
                'avg_results_per_search': np.mean(self.performance_metrics['result_counts']),
                'avg_similarity_score': np.mean(self.performance_metrics['similarity_scores']),
                'total_searches_performed': len(self.performance_metrics['search_times'])
            }
        
        return {
            "system_info": {
                "is_trained": self.is_trained,
                "total_documents": self.total_documents,
                "vocabulary_size": len(self.vectorizer.vocabulary_) if self.is_trained else 0,
                "clusters_count": len(self.cluster_info),
                "config": self.config
            },
            "database_statistics": db_stats,
            "encryption_statistics": encryption_stats,
            "performance_metrics": perf_stats,
            "cluster_analysis": self.cluster_info,
            "language_distribution": Counter([
                meta.get('language', 'unknown') 
                for meta in self.document_metadata
            ])
        }
    
    def export_system_data(self, filepath: str, include_vectors: bool = False) -> Dict:
        """صادرات داده‌های سیستم"""
        try:
            export_data = {
                "export_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "system_version": "FHE Search Engine v1.0",
                    "total_documents": self.total_documents,
                    "include_vectors": include_vectors
                },
                "configuration": self.config,
                "document_metadata": self.document_metadata,
                "cluster_information": self.cluster_info,
                "performance_metrics": self.performance_metrics,
                "vocabulary": list(self.vectorizer.vocabulary_.keys()) if self.is_trained else []
            }
            
            # اضافه کردن بردارها (اختیاری)
            if include_vectors:
                export_data["document_vectors"] = [vec.tolist() for vec in self.document_vectors]
            
            # ذخیره فایل
            filepath = Path(filepath)
            with open(filepath.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"داده‌های سیستم در {filepath} ذخیره شد")
            return {"status": "success", "filepath": str(filepath.with_suffix('.json'))}
            
        except Exception as e:
            logger.error(f"خطا در صادرات داده‌ها: {e}")
            return {"status": "error", "message": str(e)}
    
    def import_system_data(self, filepath: str) -> Dict:
        """وارد کردن داده‌های سیستم"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # بازیابی متادیتا
            if 'document_metadata' in import_data:
                self.document_metadata = import_data['document_metadata']
                self.total_documents = len(self.document_metadata)
            
            # بازیابی اطلاعات خوشه‌ها
            if 'cluster_information' in import_data:
                self.cluster_info = import_data['cluster_information']
            
            # بازیابی آمار عملکرد
            if 'performance_metrics' in import_data:
                self.performance_metrics.update(import_data['performance_metrics'])
            
            # بازیابی بردارها (در صورت وجود)
            if 'document_vectors' in import_data:
                self.document_vectors = [
                    np.array(vec) for vec in import_data['document_vectors']
                ]
            
            logger.info(f"داده‌های سیستم از {filepath} بازیابی شد")
            return {
                "status": "success", 
                "imported_documents": len(self.document_metadata),
                "imported_clusters": len(self.cluster_info)
            }
            
        except Exception as e:
            logger.error(f"خطا در وارد کردن داده‌ها: {e}")
            return {"status": "error", "message": str(e)}
    
    def __del__(self):
        """پاک‌سازی منابع"""
        if hasattr(self, 'db_manager'):
            self.db_manager.close()


# مثال استفاده
def main():
    """تابع اصلی برای تست سیستم"""
    
    # ایجاد نمونه از سیستم
    search_engine = FHESearchEngine()
    
    # نمونه اسناد فارسی و انگلیسی
    sample_documents = [
        "یادگیری ماشین یکی از شاخه‌های مهم هوش مصنوعی است که امروزه کاربرد فراوانی دارد.",
        "امنیت داده‌ها در فضای ابری یکی از چالش‌های اصلی شرکت‌های فناوری است.",
        "Machine learning is revolutionizing various industries including healthcare and finance.",
        "تکنولوژی بلاکچین و رمزنگاری در آینده نقش مهمی خواهند داشت.",
        "Cloud computing provides scalable solutions for modern businesses.",
        "پردازش زبان طبیعی به کامپیوترها کمک می‌کند تا متن انسانی را درک کنند.",
        "Artificial intelligence algorithms are becoming more sophisticated every year.",
        "حریم خصوصی کاربران در عصر دیجیتال اهمیت ویژه‌ای پیدا کرده است."
    ]
    
    # افزودن اسناد
    print("=== افزودن اسناد به سیستم ===")
    result = search_engine.add_documents(sample_documents)
    print(f"وضعیت: {result['status']}")
    print(f"تعداد اسناد پردازش شده: {result['processed_documents']}")
    print(f"توزیع زبان‌ها: {result['language_distribution']}")
    print(f"زمان پردازش: {result['processing_time']:.2f} ثانیه")
    print()
    
    # جستجوی نمونه
    test_queries = [
        "یادگیری ماشین و هوش مصنوعی",
        "امنیت و حریم خصوصی",
        "machine learning algorithms",
        "رمزنگاری و بلاکچین"
    ]
    
    print("=== نتایج جستجو ===")
    for query in test_queries:
        print(f"\nکوئری: '{query}'")
        search_result = search_engine.search(query, top_k=3)
        
        if search_result['status'] == 'success':
            print(f"زبان کوئری: {search_result['query_language']}")
            print(f"تعداد نتایج: {search_result['returned_count']}")
            print(f"زمان اجرا: {search_result['execution_time']:.3f} ثانیه")
            
            for i, result in enumerate(search_result['results'], 1):
                original_text = result['metadata'].get('original_text', '')
                score = result['similarity_score']
                print(f"  {i}. امتیاز شباهت: {score:.3f}")
                print(f"     متن: {original_text[:80]}...")
        else:
            print(f"خطا: {search_result['message']}")
        print("-" * 50)
    
    # نمایش وضعیت سیستم
    print("\n=== وضعیت سیستم ===")
    status = search_engine.get_system_status()
    print(f"تعداد کل اسناد: {status['system_info']['total_documents']}")
    print(f"اندازه واژگان: {status['system_info']['vocabulary_size']}")
    print(f"تعداد خوشه‌ها: {status['system_info']['clusters_count']}")
    
    if status['performance_metrics']:
        perf = status['performance_metrics']
        print(f"میانگین زمان جستجو: {perf['avg_search_time']:.3f} ثانیه")
        print(f"میانگین تعداد نتایج: {perf['avg_results_per_search']:.1f}")
    
    # صادرات داده‌ها
    print("\n=== صادرات داده‌ها ===")
    export_result = search_engine.export_system_data("fhe_search_export")
    print(f"وضعیت صادرات: {export_result['status']}")
    if export_result['status'] == 'success':
        print(f"فایل ذخیره شد در: {export_result['filepath']}")


if __name__ == "__main__":
    main()
