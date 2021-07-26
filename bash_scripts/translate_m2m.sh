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

# sacrebleu --echo src -l en-$lg -t $DATA_PATH | head -n 20 > $DATA_ROOT/raw_input.en-$lg.en

for lang in en ; do
  for pair in en $lg; do
      echo $DATA_PATH.$pair
      python scripts/spm_encode.py \
          --model $SPE_MODEL \
          --output_format=piece \
          --inputs=$DATA_PATH.$pair \
          --outputs=$DATA_ROOT/spm.${lang}.$pair
  done
done

mkdir -p $DATA_ROOT/en.spm.dest
fairseq-preprocess \
    --source-lang en --target-lang $lg \
    --only-source \
    --testpref $DATA_ROOT/spm.en \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir $DATA_ROOT/en.spm.dest \
    --srcdict $DATA_DICT  --tgtdict $DATA_DICT \

echo "Done preprocess!"

DATA_BIN=$DATA_ROOT/en.spm.dest

fairseq-generate \
    $DATA_BIN \
    --batch-size 2 \
    --path $PRETRAIN \
    --fixed-dictionary $MODEL_DICT \
    -s en -t $lg \
    --remove-bpe 'sentencepiece' \
    --beam 5 \
    --task translation_multi_simple_epoch \
    --lang-pairs $LANGUAGE_PAIR \
    --decoder-langtok --encoder-langtok src \
    --gen-subset test  \
    --skip-invalid-size-inputs-valid-test > results/gen_out_$lg

echo "Done generate!"
cat ./results/gen_out_$lg | grep -P "^H" | sort -V | cut -f 3- | sh ./examples/m2m_100/tok.sh $lg > results/hyp_$lg

echo "Done translate!"