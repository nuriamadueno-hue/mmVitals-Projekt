import numpy as np
import matplotlib.pyplot as plt

rd = np.load("adc_data_Test_rd.npy")

rd_db = 20 * np.log10(rd + 1e-6)

plt.figure()
plt.imshow(rd_db, aspect="auto", origin="lower")
plt.xlabel("Range bin")
plt.ylabel("Doppler bin")
plt.colorbar(label="Magnitude [dB]")
plt.title("Range-Doppler Map")
plt.show()