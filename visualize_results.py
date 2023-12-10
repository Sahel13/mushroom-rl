import numpy as np
import matplotlib.pyplot as plt


env = "cartpole"
models = ["ppo", "trpo"]

plt.figure()

for model in models:
    data = np.loadtxt(f"results/{model}_{env}.csv", delimiter=",", skiprows=1)
    np.save(f"results/{model}_{env}.npy", data[:, 1:])

    x_axis = data[:, 0]
    mean_reward = np.mean(data[:, 1:], axis=1)
    std_reward = np.std(data[:, 1:], axis=1)

    plt.errorbar(x_axis, mean_reward, yerr=std_reward, fmt="-o", label=f"{model}")

plt.legend()
plt.xlabel("Number of samples")
plt.ylabel("Expected reward per episode")
plt.title(f"{env}")
plt.show()
