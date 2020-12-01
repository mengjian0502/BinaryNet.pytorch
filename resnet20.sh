PYTHON="/home/jmeng15/anaconda3/bin/python3"
save_path="resnet18_binary"
model=resnet_binary

$PYTHON main_binary.py --model ${model} \
    --save ${save_path} \
    --dataset cifar10 \
    --depth 18
