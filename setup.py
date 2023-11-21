from setuptools import setup, find_packages

setup(
    name='MedicinalPlantMining',
    url='https://github.com/alrichardbollans/MedicinalPlantMining',
    author='Adam Richard-Bollans',
    author_email='38588335+alrichardbollans@users.noreply.github.com',
    # Needed to actually package something
    packages=find_packages(),

    install_requires=[
        "automatchnames == 1.2.3",
        "pandas",
        "tqdm",
        "openpyxl"
    ],
    # *strongly* suggested for sharing
    version='1.1',
    description='Collected packages for downloading corpora and extracting plant names',
    long_description=open('README.md').read(),
)
