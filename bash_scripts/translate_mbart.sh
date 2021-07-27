# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

tgt_lg=$1      # translated to this language
tgt_lg_mbart=$2
MODEL_DIR=$3   # path/to/model_dir
DATA_ROOT=$4   # path/to/XGLUE/NTG
max_len=$5     # 512

src_lg=en
src_lg_mbart=en_XX
PRETRAIN=$MODEL_DIR/model.pt
lang_list=$MODEL_DIR/ML50_langs.txt
SPE_MODEL=$MODEL_DIR/sentence.bpe.model
DATA_DICT_SRC=$MODEL_DIR/dict.${src_lg_mbart}.txt
DATA_DICT_TGT=$MODEL_DIR/dict.${tgt_lg_mbart}.txt

if [ ! -x results/ ]; then
   mkdir results/
fi

for pair in tgt src ; do
    DATA_PATH=$DATA_ROOT/xglue.ntg.${src_lg}.$pair.train
    echo $DATA_PATH
    python scripts/spm_encode.py \
        --model $SPE_MODEL \
        --inputs=$DATA_PATH \
        --outputs=$DATA_ROOT/train.spm.$pair.${src_lg_mbart}

  # Truncate source to 512
  python ./bash_scripts/truncate_src.py --path $DATA_ROOT/train.spm.$pair.${src_lg_mbart} --max_len $max_len
  echo "Done Truncate!"

  DATA_BIN=$DATA_ROOT/train.bin.$pair.${src_lg}
  if [ ! -x $DATA_BIN ]; then
     mkdir $DATA_BIN
  fi

  fairseq-preprocess \
      --source-lang ${src_lg_mbart} --target-lang ${tgt_lg_mbart} \
      --only-source \
      --testpref $DATA_ROOT/train.spm.$pair \
      --thresholdsrc 0 \
      --destdir $DATA_BIN \
      --srcdict $DATA_DICT_SRC \

  cp $DATA_DICT_SRC  $DATA_BIN/dict.${src_lg_mbart}.txt
  cp $DATA_DICT_TGT  $DATA_BIN/dict.${tgt_lg_mbart}.txt

  echo "Done preprocess!"

  fairseq-generate \
      $DATA_BIN \
      --batch-size 8  \
      --path $PRETRAIN \
      -s ${src_lg_mbart} -t ${tgt_lg_mbart} \
      --sacrebleu --remove-bpe 'sentencepiece' \
      --task translation_multi_simple_epoch \
      --decoder-langtok --encoder-langtok $pair \
      --gen-subset test  \
      --lang-dict $lang_list \
      --lang-pairs ${src_lg_mbart}-${tgt_lg_mbart}  > results/mbart_gen_${tgt_lg}_${pair}

  echo "Done generate!"
  grep ^H results/mbart_gen_${tgt_lg}_${pair} | sort -n -k 2 -t '-' | cut -f 3 >results/xglue.ntg.${tgt_lg}.$pair.train

done
echo "Done translate!"