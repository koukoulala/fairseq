# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

lg=$1          # translated to this language
MODEL_DIR=$2   # path/to/model_dir
DATA_ROOT=$3   # path/to/XGLUE/NTG
ckpt_name=$4   # 418M_last_checkpoint.pt
data_name=$5   # sampled_xglue.ntg.en.src.train

PRETRAIN=$MODEL_DIR/$ckpt_name
SPE_MODEL=$MODEL_DIR/spm.128k.model
MODEL_DICT=$MODEL_DIR/model_dict.128k.txt
DATA_DICT=$MODEL_DIR/data_dict.128k.txt
LANGUAGE_PAIR=$MODEL_DIR/language_pairs_small_models.txt
DATA_PATH=$DATA_ROOT/$data_name

if [ ! -x results/ ]; then
   mkdir results/
fi

DATA_BIN=$DATA_ROOT/en.spm.dest

fairseq-generate \
    $DATA_BIN \
    --batch-size 16 \
    --path $PRETRAIN \
    --fixed-dictionary $MODEL_DICT \
    -s en -t $lg \
    --remove-bpe 'sentencepiece' \
    --beam 5 \
    --task translation_multi_simple_epoch \
    --lang-pairs $LANGUAGE_PAIR \
    --decoder-langtok --encoder-langtok src \
    --gen-subset train  > results/gen_out_$lg


cd ./examples/m2m_100
cat results/gen_out_$lg | grep -P "^H" | sort -V | cut -f 3- | sh tok.sh $lg > results/hyp_$lg

