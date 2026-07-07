
from src.synthetic_generator import generate_data
from src.btc_kernel_recovery import recover_memory
from src.spatial_kernel_recovery import recover_spatial
from src.joint_reconstruction import reconstruct
from src.metrics import rmse
from src.plotting import plot_all

btc,snap,true = generate_data()

rec_h = recover_memory(btc)
rec_G = recover_spatial(snap)

pred = reconstruct(rec_G, rec_h)

print("Memory kernel RMSE:", rmse(true["H"], rec_h["H"]))
print("Pipeline completed")

plot_all(btc,snap,true,rec_h,rec_G,pred)
