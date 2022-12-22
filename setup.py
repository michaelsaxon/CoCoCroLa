"""from setuptools import setup

setup(name='CoCoCroLa',
    version='0.1',
    description='A package for building and evaluating the multilingual conceptual coverage of generative text-to-image models.',
    url='https://github.com/michaelsaxon/CoCoCroLa',
    author='Michael Saxon',
    author_email='saxon@ucsb.edu',
    packages=['cococrola'],
    install_requires=[
        'diffusers',
        'transformers',
        'torch',
        'click'
    ],
    extras_require={
        'cogview' : ['git+https://github.com/Sleepychord/Image-Local-Attention.git'],
        'figures' : ['seaborn']
    },
    entry_points={
        'console_scripts': [
            'cccl-build-benchmark=cococrola.create.benchmark:main',
            'cccl-evaluate=cococrola.evaluate.evaluate_model:main'
        ]
    },    zip_safe=False
)
"""