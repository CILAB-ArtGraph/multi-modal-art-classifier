echo 'START EXPERIMENTS'  
n=1

for lr in 3e-4 3e-5
do
    echo Experiment number $n with style and learning rate $lr and embedding artwork
    python3 run_baseline_context.py --exp baseline-context-net --label style --epochs 100 --lr $lr --embedding artwork --net context-net
    ((n=n+1))
done

for lr in 3e-4 3e-5
do
    echo Experiment number $n with genre and learning rate $lr and embedding artwork
    python3 run_baseline_context.py --exp baseline-context-net --label genre --epochs 100 --lr $lr --embedding artwork --net context-net
    ((n=n+1))
done


for lr in 3e-4 3e-5
do
    echo Experiment number $n with style and learning rate $lr and embedding artwork
    python3 run_baseline_context.py --exp baseline-sansaro --label style --epochs 100 --lr $lr --embedding artwork --net multi-modal
    ((n=n+1))
done

for lr in 3e-4 3e-5
do
    echo Experiment number $n with genre and learning rate $lr and embedding artwork
    python3 run_baseline_context.py --exp baseline-sansaro --label genre --epochs 100 --lr $lr --embedding artwork --net multi-modal
    ((n=n+1))
done

echo 'END EXPERIMENTS'  