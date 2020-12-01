PYTHON="/home/jmeng15/anaconda3/bin/python3"
depth=18
model=resnet_2bit
act_precision=2
w_prec=2
optimizer=SGD
mode=sawb
k=2
save_path="resnet${depth}_binary_inflation1_pactquant_A${act_precision}bit_W${w_prec}_mode${q_mode}_k${k}_${optimizer}"

$PYTHON -W ignore main_binary.py \
    --model ${model} \
    --save ${save_path} \
    --dataset cifar10 \
    --depth ${depth} \
    --act_prec ${act_precision} \
    --w_prec ${w_prec} \
    --q_mode ${mode} \
    --k ${k} \
    --optimizer ${optimizer} \
    --b 128 \
