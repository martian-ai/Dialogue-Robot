sh clean.sh

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --append channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/fastai/
conda config --append channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --append channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/ 

python3 -m pip install ipykernel
python3 -m ipykernel install --user

# git clone https://github.com/dongrixinyu/chinese_keyphrase_extractor
# cd ~/chinese_keyphrase_extractor
# pip install .


# data download

# data upload