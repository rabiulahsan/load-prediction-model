from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns a list of requirements.
    '''
    with open(file_path) as file:
        requirements = [req.strip() for req in file if req.strip() != HYPEN_E_DOT]
    return requirements


setup(
name='loan-prediction',
version='0.1.0',
author='rabiul',
author_email='rabiulahsan64@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)