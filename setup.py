#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# خواندن فایل README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "سیستم جستجوی رمزنگاری هومورفیک"

# خواندن requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="fhe-search-system",
    version="1.0.0",
    author_email="info@fhe-search.ir",
    description="سیستم جستجوی رمزنگاری هومورفیک با پشتیبانی چندزبانه",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/FHE_Search_System",
    
    packages=find_packages(),
    py_modules=['fhe_search'],
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security :: Cryptography",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Natural Language :: Persian",
        "Natural Language :: Arabic",
        "Natural Language :: English",
    ],
    
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'flake8>=3.8',
            'black>=21.0',
            'mypy>=0.800',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5',
            'myst-parser>=0.15',
        ]
    },
    
    entry_points={
        'console_scripts': [
            'fhe-search=fhe_search:main',
        ],
    },
    
    include_package_data=True,
    package_data={
        '': ['*.json', '*.txt', '*.md'],
    },
    
    project_urls={
        "Bug Reports": "https://github.com/username/FHE_Search_System/issues",
        "Source": "https://github.com/username/FHE_Search_System",
        "Documentation": "https://fhe-search.readthedocs.io/",
    },
    
    keywords=[
        "homomorphic encryption", "FHE", "search engine", "privacy", 
        "persian nlp", "multilingual", "cryptography", "machine learning",
        "رمزنگاری هومورفیک", "جستجو", "حریم خصوصی", "پردازش زبان فارسی"
    ],
    
    zip_safe=False,
)
