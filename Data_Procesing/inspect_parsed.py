import numpy as np

data = np.load("adc_data_Test_parsed.npz")

cube = data["cube_chirps"]
rd = np.load("adc_data_Test_rd.npy")

print("ADC cube shape:", cube.shape)
print("Range-Doppler map shape:", rd.shape)

print("Example complex ADC sample:")
print(cube[0, 0, 0])