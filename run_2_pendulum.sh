for seed in {1..20}
do
  sleep 30
  julia src/experiments/train_n_pendulum_unstable.jl $seed 2;
done
for seed in {1..20}
do
  sleep 30
  julia src/experiments/train_n_pendulum_kras.jl $seed 2;
done
