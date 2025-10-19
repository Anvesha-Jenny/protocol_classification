import numpy as np
import random
import csv

# --- Configuration ---
num_samples_per_class = 1000  # total = 3 * this number
output_file = "protocol_dataset.csv"

# --- Helper function to generate realistic values ---
def generate_protocol_data(protocol, n):
    data = []

    for _ in range(n):
        if protocol == "UART":
            data_rate = random.uniform(9.6e3, 1e6)
            payload = random.randint(8, 128)
            noise = random.uniform(25, 40)
            latency = random.uniform(8, 20)
            array_rate = data_rate * random.uniform(2, 4)

        elif protocol == "SPI":
            data_rate = random.uniform(0.5e6, 50e6)
            payload = random.randint(8, 512)
            noise = random.uniform(10, 25)
            latency = random.uniform(1, 5)
            array_rate = data_rate * random.uniform(2, 10)

        elif protocol == "I2C":
            data_rate = random.uniform(1e5, 3.4e6)
            payload = random.randint(8, 256)
            noise = random.uniform(30, 45)
            latency = random.uniform(10, 30)
            array_rate = data_rate * random.uniform(2, 6)

        data.append([data_rate, payload, noise, latency, array_rate, protocol])

    return data

# --- Generate dataset ---
uart_data = generate_protocol_data("UART", num_samples_per_class)
spi_data = generate_protocol_data("SPI", num_samples_per_class)
i2c_data = generate_protocol_data("I2C", num_samples_per_class)

# Combine and shuffle
dataset = uart_data + spi_data + i2c_data
random.shuffle(dataset)

# --- Save to CSV ---
header = ["Data_Rate_bps", "Payload_bits", "Noise_Level_dB", "Latency_ms", "Array_Rate_Hz", "Label"]

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(dataset)

print(f"✅ Dataset generated successfully: {output_file}")
print(f"Total samples: {len(dataset)}")
