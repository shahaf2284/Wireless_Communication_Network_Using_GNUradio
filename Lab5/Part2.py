import numpy as np
import matplotlib.pyplot as plt

P = 5  # Total power of the transmitted vector
Mr = 4  # Number of receive antennas
Mt = 4  # Number of transmit antennas
M = 5000  # Number of Monte Carlo simulations

H_true = np.random.normal(0, 1, (Mr, Mt))  # True channel matrix
N_range = np.arange(Mr, 5 * Mr + 1)  # Range of training symbol numbers
MMSE = []  # List to store the MMSE values

for N in N_range:
    mse_sum = 0

    for _ in range(M):
        # Generate N training symbols randomly from the set {-4, -3, ..., 3, 4}
        training_symbols = np.random.choice([-4, -3, -2, -1, 0, 1, 2, 3, 4], size=(Mt, N))

        # Normalize the training symbols and allocate power uniformly
        # such that the total power of the transmitted vector is P
        power_per_symbol = P / (N * Mt)
        transmitted_symbols = np.sqrt(power_per_symbol) * training_symbols / np.sqrt(10)

        # Add zero-mean, 0.5-variance additive noise to the training symbols
        noise = np.random.normal(0, 0.5, (Mr, N))
        received_symbols = np.dot(H_true, transmitted_symbols) + noise

        # Estimate H and calculate per-component MSE (to average later)
        H_estimated = np.dot(received_symbols, np.linalg.pinv(transmitted_symbols))
        mse = np.mean(np.square(H_true - H_estimated))
        mse_sum += mse

    # Calculate average MMSE
    mmse = (1 / (Mr * Mt)) * (mse_sum / M)
    MMSE.append(mmse)

# Convert MMSE to dB
MMSE_dB = 10 * np.log10(MMSE)

# Create figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the results
ax.plot(N_range * Mt, MMSE_dB, 'b.-', linewidth=2, markersize=8)

# Set labels and title
ax.set_xlabel('Number of Training Symbols')
ax.set_ylabel('MMSE (dB)')
ax.set_title('MMSE vs. Number of Training Symbols')

# Show grid
ax.grid(True)

# Display plot
plt.show()
