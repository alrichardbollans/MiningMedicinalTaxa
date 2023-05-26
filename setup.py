from setuptools import setup, find_packages

setup(
    name='MedicinalPlantMining',
    url='https://github.com/alrichardbollans/MedicinalPlantMining',
    author='Adam Richard-Bollans',
    author_email='38588335+alrichardbollans@users.noreply.github.com',
    # Needed to actually package something
    packages=find_packages(),

    install_requires=[
        "automatchnames == 1.1.1"
    ],
    # *strongly* suggested for sharing
    version='1.0',
    description='A set of python packages for plant trait data',
    long_description=open('README.md').read(),
)
