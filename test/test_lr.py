import matplotlib
import os, sys
from os.path import dirname, abspath, join
TEST_DIR = dirname(__file__)
sys.path.insert(0, abspath(join(TEST_DIR, "../")))
from vlkit.lr import CosAnnealingLR, MultiStepLR
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
for ax in axes:
    ax.grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')

epochs = 120
loader_len = 5004
warmup_epochs = 5

lr_scheduler = CosAnnealingLR(loader_len=loader_len, epochs=epochs,
                                max_lr=0.1, min_lr=0.01, warmup_epochs=warmup_epochs)
lr_record = []
for i in range(epochs*loader_len):
    lr_record.append(lr_scheduler.step())

lr_scheduler.restart(0.05)
for i in range(epochs*loader_len):
    lr_record.append(lr_scheduler.step())

lr_scheduler.restart(0.01)
for i in range(epochs*loader_len):
    lr_record.append(lr_scheduler.step())

axes[0].plot(lr_record, color="blue", linewidth=2)
axes[0].set_xlabel("Iter", fontsize=12)
axes[0].set_ylabel("Learning Rate", fontsize=12)
axes[0].set_title("Warmup + Cosine Annealing")

# multi step lr
lr_scheduler = MultiStepLR(loader_len=loader_len, milestones=[30, 60, 90], gamma=0.1, warmup_epochs=warmup_epochs)
lr_record = []
for i in range(epochs*loader_len):
    lr_record.append(lr_scheduler.step())
axes[1].plot(lr_record, color="blue", linewidth=2)
axes[1].set_xlabel("Iter", fontsize=12)
axes[1].set_ylabel("Learning Rate", fontsize=12)
axes[1].set_title("Warmup + Multistep LR")

plt.tight_layout()
plt.savefig("lr_scheduler.svg")