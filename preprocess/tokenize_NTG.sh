
DATA=/home/work/xiaoyu/fairseq/data/NTG
MODEL_DIR=/home/work/xiaoyu/fairseq/models/mbart.cc25.v2
CODE_ROOT=/home/work/xiaoyu/fairseq

SPE_MODEL=$MODEL_DIR/sentence.bpe.model
DICT=$MODEL_DIR/dict.txt

# Binarize

for lg in en es fr de ru; do
    echo $lg
    mkdir -p ${DATA}/${lg}.spm.dest
	fairseq-preprocess  \
	--source-lang src \
    --target-lang tgt \
    --only-source \
	--testpref ${DATA}/${lg}.spm/$test.spm \
	--destdir ${DATA}/${lg}.spm.dest \
	--thresholdtgt 0 \
	--thresholdsrc 0 \
	--srcdict ${DICT} \
	--workers 70
done


for lg in en; do
    echo $lg
    mkdir -p $DATA_BIN/$lg
    python $CODE_ROOT/preprocess.py \
    --source-lang src \
    --target-lang tgt \
    --trainpref ${DATA}/${lg}.spm/train.spm \
    --validpref ${DATA}/${lg}.spm/dev.spm \
    --destdir ${DATA}/${lg}.spm.dest \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${DICT} \
    --tgtdict ${DICT} \
    --workers 70
done

for lg in es fr de ru; do
    echo $lg
    mkdir -p $DATA_BIN/$lg
    python $CODE_ROOT/preprocess.py \
    --source-lang src \
    --target-lang tgt \
    --validpref ${DATA}/${lg}.spm/dev.spm \
    --destdir ${DATA}/${lg}.spm.dest \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${DICT} \
    --tgtdict ${DICT} \
    --workers 70
done

echo "Done!"
