import numpy as np


def fft(x):
    """
    Compute the Fast Fourier Transform (FFT) of a 1D array x.

    Parameters:
    x (np.array): The input array (1D complex array).

    Returns:
    np.array: The FFT of the input array.
    """
    N = len(x)
    if N <= 1:
        return x

    # Divide the array into even and odd parts
    even = fft(x[0::2])
    odd = fft(x[1::2])

    # Compute the combined terms using the twiddle factor
    T = np.exp(-2j * np.pi * np.arange(N) / N)[: N // 2] * odd
    return np.concatenate([even + T, even - T])


def ifft(X):
    """
    Compute the Inverse Fast Fourier Transform (IFFT) of a 1D array X.

    Parameters:
    X (np.array): The input array in frequency domain (1D complex array).

    Returns:
    np.array: The IFFT of the input array (time domain).
    """
    N = len(X)
    if N <= 1:
        return X

    # Apply conjugate, FFT, and scale by 1/N
    X_conj = np.conjugate(X)
    X_fft = fft(X_conj)
    return np.conjugate(X_fft) / N


# Example usage:
# Create a sample signal of 8 points
# x = np.random.random(8)  # Input signal (real values)
x = np.array([1, 2, 3, 4, 5, 6, 0, 0])
x_fft = fft(x)  #
x_ifft = ifft(x_fft)  # IFFT (reconstructed signal)

print("Input signal:", x)
print("FFT:", x_fft)
print("Reconstructed signal (IFFT):", x_ifft)
