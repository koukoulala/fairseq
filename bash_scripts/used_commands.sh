# 7.26
# using m2m to translate NET-en
nohup bash bash_scripts/translate_m2m.sh fr ../ckpt/m2m_100 ../datasets/trans 418M_last_checkpoint.pt small_test &> logs/trans_m2m.out &
