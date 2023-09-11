from setuptools import find_packages, setup


def parse_requirement(requirement_file):
    with open(requirement_file, 'r') as fp:
        reqs = [r.strip() for r in fp]
    return reqs


setup(name='feature_analysis',
      version='0.0.0',
      python_requires='<3.10',
      author='Andy Wang',
      author_email='andy80136@gmail.com',
      description='Feature analysis and selection for training process',
      packages=find_packages(),
      install_requires=parse_requirement('requirements/core.txt'),
      extras_require={
          'dev': parse_requirement('requirements/dev.txt')
      })
