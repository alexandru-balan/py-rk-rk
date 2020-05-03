import setuptools

setuptools.setup(
    name='py-rk-rk',
    version='2020.1',
    url='https://github.com/alexandru-balan/py-rk-rk',
    license='GPL-3.0-or-later',
    author='Alexandru Balan',
    author_email='balan.alexandru1997@tutanota.com',
    description='A python implementation of two algorithms for solving low-rank linear systems',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0-or-later",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8'
)
