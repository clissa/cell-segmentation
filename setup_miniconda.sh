# setup conda config
cat << EOF > ~/.condarc
channels:
  - fastai
  - pytorch
  - conda-forge
  - defaults
channel_priority: strict
#auto_activate_base: false
EOF

# download latest miniconda and init
DOWNLOAD=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
wget $DOWNLOAD
bash Miniconda3-latest*.sh -b -p $1/miniconda3
$1/miniconda3/bin/conda init bash

# configure bashrc and miniconda init
RCFILE=~/.bashrc
MINICONDA_FILE=$1/.minicondainit
chmod -R o+rwX $1/miniconda3 # make it editable to non-root users

set -e
perl -n  -e 'print if     />>> conda/../<<< conda/' $RCFILE > $MINICONDA_FILE
perl -ni -e 'print unless />>> conda/../<<< conda/' $RCFILE
echo source $MINICONDA_FILE >> $RCFILE
source $MINICONDA_FILE

conda config --set auto_activate_base false

# install mamba
conda install -y -c conda-forge mamba
