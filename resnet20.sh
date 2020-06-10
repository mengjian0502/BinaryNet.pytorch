PYTHON="/home/mengjian/anaconda3/bin/python3"
save_path="resnet20_binary_inflation5"
model=resnet_binary

$PYTHON main_binary.py --model ${model} \
    --save ${save_path} \
    --dataset cifar10 \
    --depth 20
