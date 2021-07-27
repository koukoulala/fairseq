# 7.26
# using m2m to translate NET-en
nohup bash bash_scripts/translate_m2m.sh fr ../ckpt/m2m_100 ../datasets/trans 418M_last_checkpoint.pt small_test &> logs/trans_m2m.out &
CUDA_VISIBLE_DEVICES=0,1 nohup bash bash_scripts/translate_m2m_only.sh fr ../ckpt/m2m_100 ../datasets/trans 418M_last_checkpoint.pt small_test &> logs/trans_m2m_only.out &
CUDA_VISIBLE_DEVICES=0,1 nohup bash bash_scripts/translate_m2m_mbart.sh fr_XX ../ckpt/mbart50.ft.1n ../datasets/trans small_test &> logs/trans_m2m_mbart.out &
CUDA_VISIBLE_DEVICES=0,1 nohup bash bash_scripts/translate_m2m_mbart.sh fr_XX ../ckpt/mbart50.ft.1n ../datasets/trans sampled_xglue.ntg.en.src.train &> logs/all_trans_m2m_mbart.out &
