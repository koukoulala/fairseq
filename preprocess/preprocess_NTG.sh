
DATA=/home/work/xiaoyu/fairseq/data/NTG
MODEL_DIR=/home/work/xiaoyu/fairseq/models/mbart.cc25.v2
CODE_ROOT=/home/work/xiaoyu/fairseq

SPE_MODEL=$MODEL_DIR/sentence.bpe.model
DICT=$MODEL_DIR/dict.txt

# copy data to specific language folder
for lg in en; do
    mkdir ${DATA}/${lg}
    cp ${DATA}/xglue.ntg.$lg.src.train ${DATA}/${lg}/train.src
    cp ${DATA}/xglue.ntg.$lg.tgt.train ${DATA}/${lg}/train.tgt
done

for lg in en es fr de ru; do
    mkdir ${DATA}/${lg}
    cp ${DATA}/xglue.ntg.$lg.src.dev ${DATA}/${lg}/dev.src
    cp ${DATA}/xglue.ntg.$lg.tgt.dev ${DATA}/${lg}/dev.tgt
    cp ${DATA}/xglue.ntg.$lg.src.test ${DATA}/${lg}/test.src
    cp ${DATA}/xglue.ntg.$lg.tgt.test ${DATA}/${lg}/test.tgt
done


# Tokenize
echo "Tokenize"
for lg in en; do
    for split in train dev; do
        for pair in tgt src; do
            echo $lg.$pair.$split
            mkdir ${DATA}/${lg}.spm
            python $CODE_ROOT/scripts/spm_encode.py --model $SPE_MODEL \
                --inputs ${DATA}/${lg}/$split.$pair --outputs ${DATA}/${lg}.spm/$split.spm.$pair
        done
    done
done

for lg in es fr de ru; do
    for split in dev; do
        for pair in tgt src; do
            echo $lg.$pair.$split
            mkdir ${DATA}/${lg}.spm
            python $CODE_ROOT/scripts/spm_encode.py --model $SPE_MODEL \
                --inputs ${DATA}/${lg}/$split.$pair --outputs ${DATA}/${lg}.spm/$split.spm.$pair
        done
    done
done


for lg in en es fr de ru; do
    for split in test; do
        for pair in src; do
            echo $lg.$pair.$split
            mkdir ${DATA}/${lg}.spm
            python $CODE_ROOT/scripts/spm_encode.py --model $SPE_MODEL \
                --inputs ${DATA}/${lg}/$split.$pair --outputs ${DATA}/${lg}.spm/$split.spm.$pair
        done
    done
done


# Binarize

for lg in en es fr de ru; do
    echo $lg
    mkdir -p ${DATA}/${lg}.spm.dest
	python $CODE_ROOT/preprocess.py \
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
