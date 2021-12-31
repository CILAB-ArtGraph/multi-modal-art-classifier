echo 'START EXPERIMENTS'  
n=1

for label in "style" "genre"
do
    for lr in 0.01 0.001 0.0001
    do
        echo Experiment number $n with $label and learning rate $lr
        python3 run.py --exp resnet-baseline --label $label --epochs 100 --lr $lr
        ((n=n+1))
    done
done

for label in "style" "genre"
do
    for lr in 0.01 0.001 0.0001
    do
        echo Experiment number $n with $label, learning rate $lr and freeze activate
        python3 run.py --exp resnet-baseline --label $label --epochs 100 --freeze --lr $lr
        ((n=n+1))
    done
done

echo 'END EXPERIMENTS'  