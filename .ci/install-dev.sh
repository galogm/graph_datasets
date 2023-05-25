# # install python 3.7.16 use pyenv
# pyenv install 3.7.16
# pyenv local 3.7.16

# create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# update pip
python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple/

# install requirements
python -m pip install -r requirements-dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

pre-commit install

echo install DEV requirements successfully!
