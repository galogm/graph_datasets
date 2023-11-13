rm -rf .env

# # install python 3.8.16 use pyenv
# pyenv install 3.8.16
# pyenv local 3.8.16

# create and activate virtual environment
python3 -m venv .env
source .env/bin/activate

# update pip
# add a source if prefered: -i https://pypi.tuna.tsinghua.edu.cn/simple/
python -m pip install -U pip

# install requirements
# add a source if prefered: -i https://pypi.tuna.tsinghua.edu.cn/simple/
python -m pip install -r requirements-dev.txt

pre-commit install

echo install DEV requirements successfully!
