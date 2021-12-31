echo 'START EXPERIMENTS'  
n=1

for label in "genre" "style"
do
    for lr in 0.001 0.0001
    do
        for lambda in 0.9 0.8 0.7
        do
            echo Experiment number $n with $label and learning rate $lr and lambda $lambda
            python3 run_kg.py --exp resnet-baseline --label $label --epochs 100 --lr $lr --lamb $lambda
            ((n=n+1))
        done
    done
done

echo 'END EXPERIMENTS'  