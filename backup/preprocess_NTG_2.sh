
CODE_ROOT=$1     # path to code root
MODEL_DIR=$2     # path/to/saved_model_dir
DATA=$3          # path/to/XGLUE/NTG

#CODE_ROOT=/home/work/xiaoyu/fairseq
#MODEL_DIR=/home/work/xiaoyu/fairseq/models/mbart.cc25.v2
#DATA=/home/work/xiaoyu/fairseq/data/NTG

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
	--testpref ${DATA}/${lg}.spm/test.spm \
	--destdir ${DATA}/${lg}.spm.dest \
	--thresholdtgt 0 \
	--thresholdsrc 0 \
	--srcdict ${DICT} \
	--workers 70
done


for lg in en; do
    echo $lg
    fairseq-preprocess \
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
    fairseq-preprocess \
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
