# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# Usage: ./get-data-para.sh $lg_pair
#

set -e

pair=$1  # input language pair
echo $pair
# data paths
MAIN_PATH=$PWD
PARA_PATH=$PWD/data/data17langxnli/para

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py

# install tools
# ./install-tools.sh

# create directories
mkdir -p $PARA_PATH


#
# Download and uncompress data
#

# ar-en
if [ $pair = "ar-en" ]; then
  echo "I am here"
  # OpenSubtitles 2018
  # wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Far-en.txt.zip -P $PARA_PATH
  # global voices
  wget -c http://opus.nlpl.eu/download.php?f=GlobalVoices/v2017q3/moses/ar-en.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/download.php?f=GlobalVoices%2Fv2017q3%2Fmoses%2Far-en.txt.zip -d $PARA_PATH
fi


#
# Tokenize and preprocess data
#

# tokenize
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  if [ ! -f $PARA_PATH/$pair.$lg.all ]; then
    cat $PARA_PATH/*.$pair.$lg | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $PARA_PATH/$pair.$lg.all
  fi
done

# split into train / valid / test
split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NTRAIN=$(( NLINES - 10000 ));
    NVAL=$(( NTRAIN + 5000 ));
    
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN             > $2;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NVAL | tail -5000  > $3;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -5000                > $4;
}
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  split_data $PARA_PATH/$pair.$lg.all $PARA_PATH/$pair.$lg.train $PARA_PATH/$pair.$lg.valid $PARA_PATH/$pair.$lg.test
done

