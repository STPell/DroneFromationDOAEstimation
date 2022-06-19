"""
    Generate random complex noise signals with AWGN added to it.

    Author: Samuel Pell, sam.pell@canterbury.ac.nz
    Date:   14/03/22
"""

import numpy as np
import samUtilities as su

TX_POWER_DB = 10            # dBm, Transmit power of the transmitter we are trying to locate
FREQUENCY = 2.45e9          # Hz, Transmit frequency
SNR = 5                     # dB, the signal to noise ratio
SAMPLES_PER_SECTOR = 100    # Number of samples per sector of the system
SAMPLING_BANDWIDTH = 2.5e6  # Hz, Sampling bandwidth
RX_NOISE_FIGURE = 8         # dB, the noise figure of the receiver


def generate_rx_powers_fixed(snr, rx_x, rx_y, ant, tx_x, tx_y, rel_bearing):
    """
       Generate the signal powers for each different sector of the antenna based on a fixed SNR value.

       Inputs:
           - snr:         The signal-to-noise ratio of the environment
           - rx_x:        The receiver's position on the x-axis
           - rx_y:        The receiver's position on the y-axis
           - ant:         The SectoredAntenna the receiver has
           - tx_x:        The transmitter's location on the x-axis
           - tx_y:        The transmitter's location on the y-axis
           - rel_bearing: The relative bearing between the transmitter and receiver.
    """
    # Assume the transmitter is continuously transmitting a random signal with a set energy level. This makes
    # the first stage of simulation easier. To get the received signal energy we assume the transmitter has a
    # perfectly omnidirectional antenna and that there is no shadowing.
    fspl = su.freespace_path_loss(np.linalg.norm(np.subtract([rx_x, rx_y], [tx_x, tx_y])),
                                  FREQUENCY)
    average_signal_rx_power = TX_POWER_DB - fspl

    # Add 3dB to the noise SNR to halve if, accounting for the noise being complex
    noise_var = su.db_to_nat(average_signal_rx_power) / su.db_to_nat(snr + 3)

    sector_rx_powers = []
    for sector in range(ant.num_sectors()):
        # The signal received in each sector is independent as we are modelling a switched signal so we generate
        # a new one for each sector. The sector is complex random noise with a variance equal to to the power
        # in the signal. This then has complex AWGN noise added to simulate incoherent interference + thermal
        # noise in each frontend.
        # NB: Subtract 3dB from the signal RX power to halve if as we have a complex signal so the real part and
        #     the imaginary part both contribute to the signal power.
        average_signal_rx_power_lin = su.db_to_nat(average_signal_rx_power)
        signal_stdev = np.sqrt(average_signal_rx_power_lin / 2)
        rx_signal = np.random.normal(loc=0, scale=signal_stdev, size=SAMPLES_PER_SECTOR) + \
                    (1j * np.random.normal(loc=0, scale=signal_stdev, size=SAMPLES_PER_SECTOR))

        # Generate the noise for the signal
        noise = np.random.normal(loc=0, scale=np.sqrt(noise_var), size=SAMPLES_PER_SECTOR) + \
                (1j * np.random.normal(loc=0, scale=np.sqrt(noise_var), size=SAMPLES_PER_SECTOR))

        # Assemble the signal from the received signal
        rx_signal = (rx_signal * ant.sector_gain(sector, rel_bearing)) + noise

        sector_rx_power = su.nat_to_db(np.mean(np.power(np.abs(rx_signal), 2)))
        sector_rx_powers.append(sector_rx_power)

    return sector_rx_powers, noise_var


def generate_rx_powers(rx_x, rx_y, ant, tx_x, tx_y, rel_bearing):
    """
       Generate the signal powers for each different sector of the antenna with the noise on the received signal based
       on the sampling bandwidth and the noise figure of the device.

       Inputs:
           - rx_x:        The receiver's position on the x-axis
           - rx_y:        The receiver's position on the y-axis
           - ant:         The SectoredAntenna the receiver has
           - tx_x:        The transmitter's location on the x-axis
           - tx_y:        The transmitter's location on the y-axis
           - rel_bearing: The relative bearing between the transmitter and receiver.
    """
    # Assume the transmitter is continuously transmitting a random signal with a set energy level. This makes
    # the first stage of simulation easier. To get the received signal energy we assume the transmitter has a
    # perfectly omnidirectional antenna and that there is no shadowing.
    fspl = su.freespace_path_loss(np.linalg.norm(np.subtract([rx_x, rx_y], [tx_x, tx_y])), FREQUENCY)
    average_signal_rx_power_db = TX_POWER_DB - fspl

    thermal_noise_power_db = -174 + (10 * np.log10(SAMPLING_BANDWIDTH))
    snr = average_signal_rx_power_db - (thermal_noise_power_db + RX_NOISE_FIGURE)

    # Add 3dB to the noise SNR to halve if, accounting for the noise being complex
    noise_var = su.db_to_nat(average_signal_rx_power_db) / su.db_to_nat(snr + 3)

    sector_rx_powers = []
    for sector in range(ant.num_sectors()):
        # The signal received in each sector is independent as we are modelling a switched signal so we generate
        # a new one for each sector. The sector is complex random noise with a variance equal to to the power
        # in the signal. This then has complex AWGN noise added to simulate incoherent interference + thermal
        # noise in each frontend.
        # NB: Subtract 3dB from the signal RX power to halve if as we have a complex signal so the real part and
        #     the imaginary part both contribute to the signal power.
        average_signal_rx_power_lin = su.db_to_nat(average_signal_rx_power_db)
        signal_stdev = np.sqrt(average_signal_rx_power_lin / 2)
        rx_signal = np.random.normal(loc=0, scale=signal_stdev, size=SAMPLES_PER_SECTOR) + \
                    (1j * np.random.normal(loc=0, scale=signal_stdev, size=SAMPLES_PER_SECTOR))

        # Generate the noise for the signal
        noise = np.random.normal(loc=0, scale=np.sqrt(noise_var), size=SAMPLES_PER_SECTOR) + \
                (1j * np.random.normal(loc=0, scale=np.sqrt(noise_var), size=SAMPLES_PER_SECTOR))

        # Assemble the signal from the received signal
        rx_signal = (rx_signal * ant.sector_gain(sector, rel_bearing)) + noise

        sector_rx_power = su.nat_to_db(np.mean(np.power(np.abs(rx_signal), 2)))
        sector_rx_powers.append(sector_rx_power)

    return sector_rx_powers, noise_var