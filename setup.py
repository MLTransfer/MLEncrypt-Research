from setuptools import setup

setup(
    name='mlencrypt',
    version='0.1',
    py_modules=['mlencrypt'],
    install_requires=[
        "cffi==1.13.2",
        "Click",
        "cryptography==2.8",
        "cycler==0.10.0",
        "kiwisolver==1.1.0",
        "matplotlib==3.1.1",
        "numpy==1.17.4",
        "pyaescrypt==0.4.3",
        "pycparser==2.19",
        "pydicom==1.3.0",
        "pyparsing==2.4.5",
        "python-dateutil==2.8.1",
        "six==1.13.0",
    ],
    entry_points='''
        [console_scripts]
        mlencrypt=mlencrypt:main
    ''',
)
