# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

lg=$1          # translated to this language
MODEL_DIR=$2   # path/to/model_dir
DATA_ROOT=$3   # path/to/XGLUE/NTG
data_name=$4   # small_test.en

PRETRAIN=$MODEL_DIR/model.pt
lang_list=$MODEL_DIR/ML50_langs.txt
SPE_MODEL=$MODEL_DIR/sentence.bpe.model
DATA_DICT=$DATA_ROOT/dict.en_XX.txt
DATA_PATH=$DATA_ROOT/$data_name

if [ ! -x results/ ]; then
   mkdir results/
fi

# : << !
for lang in en_XX ; do
    echo $DATA_PATH
    python scripts/spm_encode.py \
        --model $SPE_MODEL \
        --inputs=$DATA_PATH \
        --outputs=$DATA_ROOT/small_spm_mbart.${lang}
done

if [ ! -x $DATA_ROOT/small_en_XX.spm_mbart.dest/ ]; then
   mkdir $DATA_ROOT/small_en_XX.spm_mbart.dest/
fi

fairseq-preprocess \
    --source-lang en_XX --target-lang $lg \
    --only-source \
    --testpref $DATA_ROOT/small_spm_mbart \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir $DATA_ROOT/small_en_XX.spm_mbart.dest \
    --srcdict $DATA_DICT  --tgtdict $DATA_DICT \

echo "Done preprocess!"
# !

DATA_BIN=$DATA_ROOT/small_en_XX.spm_mbart.dest

fairseq-generate \
    $DATA_BIN \
    --max-sentences 2  \
    --path $PRETRAIN \
    -s en_XX -t $lg \
    --sacrebleu --remove-bpe 'sentencepiece' \
    --task translation_multi_simple_epoch \
    --decoder-langtok --encoder-langtok src \
    --gen-subset test  \
    --lang-dict $lang_list \
    --lang-pairs en_XX-fr_XX  > results/mbart_gen_out_$lg_all

echo "Done generate!"
grep ^H results/mbart_gen_out_$lg | sort -n -k 2 -t '-' | cut -f 3 >results/mbart_translated_$lg_all

echo "Done translate!"