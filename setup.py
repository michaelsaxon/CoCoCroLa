from setuptools import setup, find_packages

setup(name='CoCoCroLa',
    version='0.1',
    description='A package for building and evaluating the multilingual conceptual coverage of generative text-to-image models.',
    url='https://github.com/michaelsaxon/CoCoCroLa',
    author='Michael Saxon',
    author_email='saxon@ucsb.edu',
    packages=find_packages(include=['cococrola', 'cococrola.*']),
    python_requires='>=3.8',
    install_requires=[
        'diffusers',
        'transformers',
        'torch',
        'click',
        'numpy'
    ],
    extras_require={
        # 'cogview' : ['git+https://github.com/Sleepychord/Image-Local-Attention.git'],
        # https://github.com/THUDM/CogView2/tree/main
        'figures' : ['seaborn'],
        'openai' : ['openai'],
        'craiyon' : ['dalle-mini'],
        'creator' : ['babelnet', 'translators']
    },
    entry_points={
        'console_scripts': [
            'cccl-build-benchmark=cococrola.create.benchmark:main',
            'cccl-evaluate=cococrola.evaluate.evaluate_model:main'
        ]
    },    zip_safe=False
)