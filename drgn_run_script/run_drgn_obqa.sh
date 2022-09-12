export CUDA_VISIBLE_DEVICES=0,1
export TRANSFORMERS_CACHE=/localscratch/chen.zheng/data/transformer_cache/
dt=`date '+%Y%m%d_%H%M%S'`


dataset="obqa"
model='roberta-large'
# shift
# shift
args=$@


elr="1e-5"
dlr="1e-3"
bs=128
n_epochs=100

k=5 #num of gnn layers
gnndim=200

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref

###### Training ######
for seed in 0; do
  python3 -u ../qa_drgn.py --dataset $dataset \
      --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs --seed $seed \
      --n_epochs $n_epochs --max_epochs_before_stop 30  \
      --train_adj /localscratch/chen.zheng/data/drgn/data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   /localscratch/chen.zheng/data/drgn/data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  /localscratch/chen.zheng/data/drgn/data/${dataset}/graph/test.graph.adj.pk \
      --train_statements /localscratch/chen.zheng/data/drgn/data/${dataset}/statement/train.statement.jsonl \
      --dev_statements   /localscratch/chen.zheng/data/drgn/data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  /localscratch/chen.zheng/data/drgn/data/${dataset}/statement/test.statement.jsonl \
      --save_model \
      --save_dir /localscratch/chen.zheng/data/drgn/${save_dir_pref}/${dataset}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \
  > train_${dataset}__enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt}.log.txt
done
