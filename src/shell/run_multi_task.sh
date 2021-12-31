echo 'START EXPERIMENTS'  
n=1

for lr in 0.01 0.001 0.0001
do
    echo Experiment number $n learning rate $lr
    python3 run_multi_task.py --exp resnet-baseline-multi-task --epochs 100 --lr $lr
    ((n=n+1))
done

for lr in 0.01 0.001 0.0001
do
    echo Experiment number $n learning rate $lr and freeze activated
    python3 run_multi_task.py --exp resnet-baseline-multi-task --epochs 100 --lr $lr --freeze
    ((n=n+1))
done

echo 'END EXPERIMENTS'  