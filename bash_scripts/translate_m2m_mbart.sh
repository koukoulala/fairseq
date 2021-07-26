# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

lg=$1          # translated to this language
MODEL_DIR=$2   # path/to/model_dir
DATA_ROOT=$3   # path/to/XGLUE/NTG

PRETRAIN=$MODEL_DIR/model.pt
lang_list=$MODEL_DIR/ML50_langs.txt

if [ ! -x results/ ]; then
   mkdir results/
fi

DATA_BIN=$DATA_ROOT/en.spm.dest
#DATA_BIN=$DATA_ROOT/en.spm.dest/train.src-tgt.src.bin

fairseq-generate \
    $DATA_BIN \
    --batch-size 8 \
    --path $PRETRAIN \
    -s en -t $lg \
    --remove-bpe 'sentencepiece' \
    --task translation_multi_simple_epoch \
    --decoder-langtok --encoder-langtok src \
    --gen-subset test  \
    --lang-dict $lang_list \
    --skip-invalid-size-inputs-valid-test > results/mbart_gen_out_$lg

echo "Done generate!"
grep ^H results/mbart_gen_out_$lg | sort -n -k 2 -t '-' | cut -f 3 >results/mbart_translated_fr

echo "Done translate!"