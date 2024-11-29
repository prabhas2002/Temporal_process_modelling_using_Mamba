device=7

#book order
# data=data/data_book_order/fold1/
# data = data/data_book_order/fold2/
# data = data/data_book_order/fold3/
# data = data/data_book_order/fold4/
# data = data/data_book_order/fold5/

# #cont time
# data = data/data_cont_time

# #hawkes
# data = data/data_hawkes

# #mimic data
# data=data/data_mimic/fold1/
# data = data/data_mimic/fold2/
# data = data/data_mimic/fold3/
# data = data/data_mimic/fold4/
# data = data/data_mimic/fold5/

# #retewwt
data=data/data_missing/

#stack overflow
#data=data/data_so/fold1/
# data = data/data_so/fold2/
# data = data/data_so/fold3/
# data = data/data_so/fold4/
# data = data/data_so/fold5/

#data=data/data_conttime/
batch=16
n_head=2
n_layers=2
d_model=512
d_rnn=64
d_inner=1024
d_k=512
d_v=512
dropout=0.1
lr=1e-4
smooth=0.1
epoch=50
log=log.txt

export PYTHONDONTWRITEBYTECODE=1

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python3 Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log 
