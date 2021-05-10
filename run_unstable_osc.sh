for seed in {1..20}
do
  julia src/experiments/train_stable_oscillator_kras.jl $seed;

done
for seed in {1..20}
do
  julia src/experiments/train_stable_oscillator_unstable.jl $seed;

done
