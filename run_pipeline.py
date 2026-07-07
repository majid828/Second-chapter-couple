from src.generator import generate_all
from src.btc_inverse import recover_kernels
from src.snapshot_inverse import snapshot_moments
from src.plotting import make_plots

btc,snap,truth=generate_all()
rec=recover_kernels(btc)
print(snapshot_moments(snap))
make_plots(btc,snap,truth,rec)
