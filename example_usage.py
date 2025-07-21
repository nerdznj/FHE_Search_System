#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
مثال استفاده از سیستم جستجوی FHE
===================================

این فایل نحوه استفاده از سیستم جستجوی رمزنگاری هومورفیک را نشان می‌دهد.
"""

from fhe_search import FHESearchEngine
import json
import time

def load_sample_documents():
    """بارگذاری اسناد نمونه"""
    documents = [
        # اسناد فارسی
        "هوش مصنوعی و یادگیری ماشین در حال تغییر دنیای فناوری هستند.",
        "امنیت سایبری و حفاظت از داده‌ها در عصر دیجیتال اهمیت بالایی دارد.",
        "رمزنگاری کوانتومی آینده امنیت اطلاعات را متحول خواهد کرد.",
        "پردازش زبان طبیعی به کامپیوترها کمک می‌کند تا انسان‌ها را بهتر درک کنند.",
        "اینترنت اشیاء و شهرهای هوشمند زندگی ما را تسهیل می‌کنند.",
        "بلاکچین و ارزهای دیجیتال سیستم مالی جهان را دگرگون می‌کنند.",
        "واقعیت مجازی و واقعیت افزوده تجربه‌های جدیدی خلق می‌کنند.",
        "محاسبات ابری و edge computing مرکز ثقل پردازش را تغییر می‌دهند.",
        
        # اسناد انگلیسی
        "Artificial intelligence is transforming healthcare through predictive analytics.",
        "Machine learning algorithms help in fraud detection and financial security.",
        "Deep learning models are revolutionizing computer vision applications.",
        "Natural language processing enables better human-computer interaction.",
        "Quantum computing promises to solve complex optimization problems.",
        "Blockchain technology ensures transparency in supply chain management.",
        "Cloud computing provides scalable infrastructure for modern applications.",
        "Internet of Things connects billions of devices worldwide.",
        
        # اسناد مختلط
        "استفاده از AI در پزشکی برای تشخیص بیماری‌ها کاربرد فراوانی دارد.",
        "Machine learning در تحلیل داده‌های مالی نقش مهمی ایفا می‌کند.",
        "Cybersecurity و امنیت سایبری دو روی یک سکه هستند.",
        "استفاده از blockchain در smart contracts انقلابی ایجاد کرده است."
    ]
    
    # متادیتای نمونه
    metadata = []
    for i, doc in enumerate(documents):
        meta = {
            "document_id": f"doc_{i+1:03d}",
            "category": "technology" if i < 16 else "mixed",
            "source": "sample_data",
            "priority": "high" if "امنیت" in doc or "security" in doc.lower() else "medium"
        }
        metadata.append(meta)
    
    return documents, metadata

def demonstrate_basic_usage():
    """نمایش استفاده پایه از سیستم"""
    print("🔍 راه‌اندازی سیستم جستجوی FHE...")
    
    # ایجاد نمونه سیستم
    engine = FHESearchEngine(config_file='config.json')
    
    # بارگذاری اسناد نمونه
    documents, metadata = load_sample_documents()
    
    print(f"📄 افزودن {len(documents)} سند به سیستم...")
    start_time = time.time()
    
    # افزودن اسناد
    result = engine.add_documents(documents, metadata)
    
    processing_time = time.time() - start_time
    
    if result['status'] == 'success':
        print(f"✅ پردازش موفق در {processing_time:.2f} ثانیه")
        print(f"   • اسناد پردازش شده: {result['processed_documents']}")
        print(f"   • اسناد رمزنگاری شده: {result['encrypted_documents']}")
        print(f"   • توزیع زبان‌ها: {result['language_distribution']}")
        print(f"   • اندازه واژگان: {result['vocabulary_size']}")
        if result['clustering_info']:
            print(f"   • تعداد خوشه‌ها: {len(result['clustering_info'])}")
    else:
        print(f"❌ خطا در پردازش: {result['message']}")
        return
    
    print("\n" + "="*60)
    
    # تست جستجوهای مختلف
    test_queries = [
        ("یادگیری ماشین", "فارسی"),
        ("امنیت سایبری", "فارسی"),
        ("artificial intelligence", "انگلیسی"),
        ("blockchain technology", "انگلیسی"),
        ("هوش مصنوعی و AI", "مختلط"),
        ("quantum computing", "انگلیسی")
    ]
    
    print("🔎 آزمایش جستجوهای مختلف:\n")
    
    for query, lang_type in test_queries:
        print(f"جستجو برای: '{query}' ({lang_type})")
        
        search_start = time.time()
        search_result = engine.search(query, top_k=3)
        search_time = time.time() - search_start
        
        if search_result['status'] == 'success':
            print(f"⏱️  زمان جستجو: {search_time:.3f} ثانیه")
            print(f"🌐 زبان تشخیص داده شده: {search_result['query_language']}")
            print(f"📊 تعداد نتایج: {search_result['returned_count']}")
            
            if search_result['results']:
                print("🎯 بهترین نتایج:")
                for i, result in enumerate(search_result['results'], 1):
                    score = result['similarity_score']
                    text = result['metadata']['original_text'][:80]
                    doc_lang = result['metadata']['language']
                    
                    print(f"   {i}. امتیاز: {score:.3f} | زبان: {doc_lang}")
                    print(f"      متن: {text}...")
            else:
                print("   هیچ نتیجه‌ای یافت نشد")
        else:
            print(f"❌ خطا: {search_result['message']}")
        
        print("-" * 50)

def demonstrate_advanced_features():
    """نمایش ویژگی‌های پیشرفته"""
    print("\n🚀 آزمایش ویژگی‌های پیشرفته:\n")
    
    engine = FHESearchEngine(config_file='config.json')
    documents, metadata = load_sample_documents()
    engine.add_documents(documents, metadata)
    
    # 1. جستجو با فیلتر
    print("1️⃣ جستجو با فیلتر زبان:")
    result = engine.search(
        "technology", 
        top_k=5, 
        filters={'language': 'english'}
    )
    print(f"   نتایج فقط انگلیسی: {result['returned_count']}")
    
    # 2. جستجو با فیلتر خوشه
    print("\n2️⃣ جستجو در خوشه خاص:")
    result = engine.search(
        "هوش مصنوعی", 
        top_k=3, 
        filters={'cluster_id': 0}
    )
    print(f"   نتایج در خوشه 0: {result['returned_count']}")
    
    # 3. نمایش وضعیت سیستم
    print("\n3️⃣ وضعیت کامل سیستم:")
    status = engine.get_system_status()
    
    print(f"   📚 تعداد کل اسناد: {status['system_info']['total_documents']}")
    print(f"   📖 اندازه واژگان: {status['system_info']['vocabulary_size']}")
    print(f"   🔐 آمار رمزنگاری: {status['encryption_statistics']['success_rate']:.1%} موفق")
    
    if status['performance_metrics']:
        perf = status['performance_metrics']
        print(f"   ⚡ میانگین زمان جستجو: {perf['avg_search_time']:.3f} ثانیه")
        print(f"   📈 میانگین امتیاز شباهت: {perf['avg_similarity_score']:.3f}")
    
    # 4. تحلیل خوشه‌ها
    if status['cluster_analysis']:
        print("\n4️⃣ تحلیل خوشه‌ها:")
        for cluster_id, info in status['cluster_analysis'].items():
            print(f"   خوشه {cluster_id}: {info['document_count']} سند")
            if info['common_words']:
                common = [word for word, count in info['common_words'][:3]]
                print(f"      کلمات مشترک: {', '.join(common)}")
    
    # 5. صادرات داده‌ها
    print("\n5️⃣ صادرات داده‌های سیستم:")
    export_result = engine.export_system_data("system_backup", include_vectors=False)
    if export_result['status'] == 'success':
        print(f"   ✅ داده‌ها در {export_result['filepath']} ذخیره شد")

def performance_benchmark():
    """آزمایش عملکرد سیستم"""
    print("\n⚡ آزمایش عملکرد:\n")
    
    engine = FHESearchEngine()
    
    # ایجاد مجموعه داده بزرگ‌تر
    base_docs, base_meta = load_sample_documents()
    
    # تکثیر اسناد برای تست عملکرد
    large_docs = []
    large_meta = []
    
    for i in range(5):  # 5 بار تکرار = 120 سند
        for j, (doc, meta) in enumerate(zip(base_docs, base_meta)):
            new_doc = f"{doc} (نسخه {i+1})"
            new_meta = {**meta, "document_id": f"doc_{i}_{j}", "version": i+1}
            large_docs.append(new_doc)
            large_meta.append(new_meta)
    
    print(f"📊 آزمایش با {len(large_docs)} سند...")
    
    # زمان‌سنجی افزودن اسناد
    start_time = time.time()
    result = engine.add_documents(large_docs, large_meta, batch_size=50)
    add_time = time.time() - start_time
    
    print(f"⏱️  زمان افزودن اسناد: {add_time:.2f} ثانیه")
    print(f"📈 سرعت پردازش: {len(large_docs)/add_time:.1f} سند/ثانیه")
    
    # آزمایش سرعت جستجو
    test_queries = ["هوش مصنوعی", "machine learning", "امنیت", "technology", "blockchain"]
    search_times = []
    
    print("\n🔍 آزمایش سرعت جستجو:")
    for query in test_queries:
        start_time = time.time()
        result = engine.search(query, top_k=10)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        print(f"   '{query}': {search_time:.3f}s ({result['returned_count']} نتیجه)")
    
    avg_search_time = sum(search_times) / len(search_times)
    print(f"\n📊 میانگین زمان جستجو: {avg_search_time:.3f} ثانیه")

def main():
    """تابع اصلی"""
    print("=" * 60)
    print("🌟 سیستم جستجوی رمزنگاری هومورفیک (FHE)")
    print("=" * 60)
    
    try:
        # استفاده پایه
        demonstrate_basic_usage()
        
        # ویژگی‌های پیشرفته
        demonstrate_advanced_features()
        
        # آزمایش عملکرد
        performance_benchmark()
        
        print("\n✅ همه آزمایش‌ها با موفقیت کامل شد!")
        print("📝 لاگ‌ها در فایل fhe_search.log ذخیره شده‌اند.")
        
    except Exception as e:
        print(f"\n❌ خطا در اجرا: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()