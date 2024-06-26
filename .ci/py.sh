# Check if python3 command exists
flag=0
if command -v python3 &>/dev/null; then
    # Extracting the version number
    python_version=$(python3 --version 2>&1 | awk '{print $2}')

    # Splitting the version number into major and minor parts
    IFS='.' read -r -a version_parts <<< "$python_version"

    major_version="${version_parts[0]}"
    minor_version="${version_parts[1]}"

    if [ "$major_version" -eq 3 ] && [ "$minor_version" -ge 8 ]; then
        echo "Python version $python_version is greater than or equal to 3.8."
    else
        echo "Python version $python_version is less than 3.8."
        flag=1
    fi
else
    echo "Python>=3.8 is not installed."
    flag=1
fi
# # install python 3.8, i.e., using pyenv:
# pyenv install 3.8
if [ "$flag" -eq 1 ]; then
    command -v pyenv >/dev/null 2>&1 && pyenv local 3.8 && { echo >&2 "Install python 3.8 using pyenv successfully.";} || { echo >&2 "Pyenv is not installed. Did not install python 3.8 using pyenv. Please install mannually and retry"; exit 1;}
fi
