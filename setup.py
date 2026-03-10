from setuptools import setup, find_packages

setup(
    name='MiningMedicinalTaxa',
    url='https://github.com/alrichardbollans/MiningMedicinalTaxa',
    author='Adam Richard-Bollans',
    author_email='38588335+alrichardbollans@users.noreply.github.com',
    # Needed to actually package something
    packages=find_packages(),
    package_data={
        'MiningMedicinalTaxa.literature_downloads': ['final_keywords_lists/*.txt'],
        'SciBert': ['models/ner_scibert_lora_full/*', 'models/re_scibert_lora_full/*'],
    },
    # added to incorporate taxonomy lists and scibert models

    install_requires=[
        'langchain==0.3.22',
        'langchain-core==0.3.83',
        'pydantic',
        'wcvpy >= 1.3.2',
        'openpyxl'
    ],
    # *strongly* suggested for sharing
    version='1.2',
    description='Collected packages for downloading corpora and extracting plant names',
    long_description=open('README.md').read(),
)
