PYTHON="/home/jianmeng/anaconda3/bin/python3"
depth=18
model=resnet_binary_act_quant
act_precision=4
save_path="resnet${depth}_binary_inflation1_pactact_${act_precision}bit"

$PYTHON main_binary.py --model ${model} \
    --save ${save_path} \
    --dataset cifar10 \
    --depth ${depth} \
    --clp \
    --a_lambda 0.001 \
    --act_prec ${act_precision}
