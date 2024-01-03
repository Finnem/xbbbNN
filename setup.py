from setuptools import setup
setup(name='xbbbNN',
        version='1.0',
        description='Library to train NN to predict binding energies between halobenzenes and the protein backbone.',
        url='https://github.com/Finnem/xbbbNN',
        author='Finn Mier',
        license='MIT',
        packages=['xbbbNN'],
        install_requires=[
                'numpy',
                'matplotlib',
                'scipy',
                'pandas'
        ],
        long_description='',
        zip_safe=False)
