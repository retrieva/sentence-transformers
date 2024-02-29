from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()


setup(
    name="retrieva-sentence-transformers",
    version="0.1.0",
    author="Satoru Katsumata",
    author_email="cof.ktmt@gmail.com",
    description="Multilingual text embeddings",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/retrieva/sentence-transformers",
    download_url="https://github.com/retrieva/sentence-transformers/",
    packages=find_packages(),
    python_requires=">=3.8.0",
    install_requires=[
        "transformers[sentencepiece]>=4.32.0,<5.0.0",
        "tqdm",
        "torch>=1.11.0",
        "numpy",
        "scikit-learn",
        "scipy",
        "nltk",
        "sentencepiece",
        "huggingface-hub>=0.15.1",
        "Pillow",
        "peft",
        "datasets",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="Transformer Networks BERT XLNet sentence embedding PyTorch NLP deep learning",
)
