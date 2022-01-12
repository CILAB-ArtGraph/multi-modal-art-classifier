echo 'START EXPERIMENTS'  
n=1

for lr in 3e-4 3e-5
do
    echo Experiment number $n multitask context-net with learning rate $lr and embedding artwork
    python3 run_baseline_context_multitask.py --exp baseline-context-net-multitask --epochs 100 --lr $lr --embedding artwork --net context-net
    ((n=n+1))
done


for lr in 3e-4 3e-5
do
    echo Experiment number $n multitask sansaro model with learning rate $lr and embedding artwork
    python3 run_baseline_context_multitask.py --exp baseline-sansaro-multitask --epochs 100 --lr $lr --embedding artwork --net multi-modal
    ((n=n+1))
done

echo 'END EXPERIMENTS'  