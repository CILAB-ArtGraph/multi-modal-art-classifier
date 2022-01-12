echo 'START EXPERIMENTS'  
n=1

for label in "style" "genre"
do
    for lr in 3e-3 3e-4 3e-5  
    do
        echo Experiment number $n with $label and learning rate $lr
        python3 ../run_baseline.py --exp resnet-baseline --label $label --epochs 100 --lr $lr
        ((n=n+1))
    done
done

echo 'END EXPERIMENTS'  