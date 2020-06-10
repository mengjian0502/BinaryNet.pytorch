PYTHON="/home/mengjian/anaconda3/bin/python3"
save_path="resnet20_binary_inflation5_act_quant"
model=resnet_binary_act_quant
act_precision=4

$PYTHON main_binary.py --model ${model} \
    --save ${save_path} \
    --dataset cifar10 \
    --depth 20 \
    --clp \
    --a_lambda 0.001 \
    --act_prec ${act_precision}