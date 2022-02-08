cd ..
find . -name '__pycache__' -type d -exec rm -rf {} \;
find . -name '.ipynb_checkpoints/' -type d -exec rm -rf {} \;