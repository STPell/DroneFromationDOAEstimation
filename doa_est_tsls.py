"""
   An implementation of the the Three-stage Simplified Least Squares (TSLS) algorithm as documented in [1].

   Author: Samuel Pell, sam.pell@canterbury.ac.nz
   Date:   11/01/22

   -----------------------
   References:
       [1] J. Werner et al., "Sectorized Antenna-based DoA Estimation and Localization: Advanced Algorithms and
           Measurements," in IEEE Journal on Selected Areas in Communications, vol. 33, no. 11, pp. 2272-2286,
           Nov. 2015, doi: 10.1109/JSAC.2015.2430292.
"""

import numpy as np
from typing import List, Tuple
import sectoredAntenna as sa
import samUtilities as su
import itertools


def maxE_bearing_est_hakkarainen(antenna: sa.SectoredAntenna, sector_powers: List[float]) -> float:
    """
       Implements the Maximum Energy (maxE) algorithm as defined in [1]. This specific algorithm is specified by [2] as
       a fall-back algorithm to determine the direction-of-arrival of the transmitter.

       Inputs:
           - ant:           The SectoredAntenna object representing the sector antenna model
           - sector_powers: The receiver power for each sector of the antenna in decibels.

       References:
           [1] A. Hakkarainen et al., "Reconfigurable antenna based doa estimation and localization in cognitive radios:
               Low complexity algorithms and practical measurements," 2014 9th International Conference on Cognitive
               Radio Oriented Wireless Networks and Communications (CROWNCOM), 2014, pp. 454-459,
               doi: 10.4108/icst.crowncom.2014.255730.
           [2] J. Werner et al., "Sectorized Antenna-based DoA Estimation and Localization: Advanced Algorithms and
               Measurements," in IEEE Journal on Selected Areas in Communications, vol. 33, no. 11, pp. 2272-2286,
               Nov. 2015, doi: 10.1109/JSAC.2015.2430292.
    """
    max_sector = 0
    max_power = -np.inf

    for sector in range(antenna.num_sectors()):
        alpha = 0
        step_size = 0.1
        for dirr in np.arange(0, 360, step_size):
            alpha += su.db_to_nat(antenna.sector_gain(sector, dirr)) * step_size
        alpha = alpha / 360

        adjusted_power = su.db_to_nat(sector_powers[sector]) / alpha
        if max_power < adjusted_power:
            max_sector = sector
            max_power = adjusted_power

    return antenna.sector_direction(max_sector)


def _tsls_ssl_step(ant: sa.SectoredAntenna, sector_powers: List[float], L: int) -> List[int]:
    """Helper function to perform the SSL step of the TSLS algorithm implemented in `tsls_bearing_est`"""
    max_pair = (0, 0)
    max_value = -np.inf
    # Start from -1 to match the last sector with the first sector as they are adjacent sectors
    for i in range(-1, len(sector_powers) - 1):
        combined_power = su.nat_to_db(su.db_to_nat(sector_powers[i]) * su.db_to_nat(sector_powers[i + 1]))
        if combined_power > max_value:
            max_value = combined_power
            max_pair = (i, i + 1)

    masp_average_rssi = su.nat_to_db(0.5 * (su.db_to_nat(sector_powers[max_pair[0]]) +
                                            su.db_to_nat(sector_powers[max_pair[1]])))

    # Need to select the L - 2 remaining sectors. If L is odd we use a slightly different algorithm to do this rather
    # than if L is even.
    if L % 2 == 0:
        # L is even, therefore we can just select the (L - 2) / 2 sectors either side of the MASP.
        q = max_pair[0]
        L_k = list(range(q - int(L / 2) + 1, q + int(L / 2) + 1))
    else:
        # L is odd. Allocate the extra left over sector to the side with the higher power
        q = max_pair[0]
        if sector_powers[max_pair[0]] > sector_powers[max_pair[1]]:
            L_k = list(range(q - int((L + 1) / 2) + 1, q + int((L - 1) / 2) + 1))
        else:
            L_k = list(range(q - int((L - 1) / 2) + 1, q + int((L + 1) / 2) + 1))

    # Need to map the sectors back to their real value to account for the -1 to match the
    L_k = np.mod(L_k, ant.num_sectors())

    return L_k, masp_average_rssi


def _tsls_dbs_g_func(ant: sa.SectoredAntenna, i: int, ncsp_i: float, j: int, ncsp_j: float):
    """
       Implements equation (9) from [1]. Used as a helper equation in `_tsls_dbs_spdoa_func`.

       -----------------------
       References:
           [1] J. Werner et al., "Sectorized Antenna-based DoA Estimation and Localization: Advanced Algorithms and
               Measurements," in IEEE Journal on Selected Areas in Communications, vol. 33, no. 11, pp. 2272-2286,
               Nov. 2015, doi: 10.1109/JSAC.2015.2430292.
    """
    dir_i = ant.sector_direction(i)
    dir_j = ant.sector_direction(j)

    # When dealing with sectors on the edge of wrapping around we need to be careful when we calculate
    # the average as there is a significant difference between the average of 360 and 350 and 0 and 350.
    if dir_i > 180 and dir_j < 180:
        dir_j += 360
    elif dir_j > 180 and dir_i < 180:
        dir_i += 360

    delta_v_ij = np.deg2rad(dir_i) - np.deg2rad(dir_j)
    delta_beta_ij = np.power(ant.beam_width(i, rad=True), 2) - np.power(ant.beam_width(j, rad=True), 2)
    alpha_i = ant.sector_gain(i, ant.sector_direction(i))
    alpha_j = ant.sector_gain(j, ant.sector_direction(j))

    g = np.lib.scimath.sqrt(np.power(delta_v_ij, 2) - (delta_beta_ij * np.log(alpha_i / alpha_j)) +
                (0.5 * delta_beta_ij * np.log(ncsp_i / ncsp_j)))
    return g


def _tsls_dbs_spdoa_func(ant: sa.SectoredAntenna, i: int, ncsp_i: float, j: int, ncsp_j: float):
    """
       Implements the SDE step of TSLS for the different beamwidth sectors (DBS) case. Performs the
       direction-of-arrival (DoA) step for a single sector pair, ij. To do this, we first solve equation (8) in [1] to
       get our two possible DoA solutions. Then we select the DoA estimate which minimises the difference between the
       two predicted RSSIs for sector i and sector j.

       -----------------------
       References:
           [1] J. Werner et al., "Sectorized Antenna-based DoA Estimation and Localization: Advanced Algorithms and
               Measurements," in IEEE Journal on Selected Areas in Communications, vol. 33, no. 11, pp. 2272-2286,
               Nov. 2015, doi: 10.1109/JSAC.2015.2430292.
    """
    g = _tsls_dbs_g_func(ant, i, ncsp_i, j, ncsp_j)

    if np.imag(g) != 0:
        return None

    dir_i = ant.sector_direction(i)
    dir_j = ant.sector_direction(j)
    beta_i = ant.beam_width(i, rad=True)
    beta_j = ant.beam_width(j, rad=True)

    # When dealing with sectors on the edge of wrapping around we need to be careful when we calculate
    # the average as there is a significant difference between the average of 360 and 350 and 0 and 350.
    if dir_i > 180 and dir_j < 180:
        dir_j += 360
    elif dir_j > 180 and dir_i < 180:
        dir_i += 360

    delta_beta_ij = np.power(beta_i, 2) - np.power(beta_j, 2)
    b_ij = (beta_i * beta_j) / delta_beta_ij
    lambda_ij = ((np.power(beta_i, 2) * np.deg2rad(dir_j)) -
                 (np.power(beta_j, 2) * np.deg2rad(dir_i))) / delta_beta_ij

    est_dir_1 = np.rad2deg(lambda_ij + (b_ij * g))
    est_dir_2 = np.rad2deg(lambda_ij - (b_ij * g))

    # Estimate the RSSI for each possible direction, first do the first estimated direction
    rho_ij_i = np.power(ant.sector_gain(i, est_dir_1), 2)
    gamma_ij_i = ncsp_i / rho_ij_i
    rho_ij_j = np.power(ant.sector_gain(j, est_dir_1), 2)
    gamma_ij_j = ncsp_j / rho_ij_j

    delta_gamma_dir_1 = np.abs(gamma_ij_i - gamma_ij_j)

    # Then the second estimated direction
    rho_ij_i = np.power(ant.sector_gain(i, est_dir_2), 2)
    gamma_ij_i = ncsp_i / rho_ij_i
    rho_ij_j = np.power(ant.sector_gain(j, est_dir_2), 2)
    gamma_ij_j = ncsp_j / rho_ij_j

    delta_gamma_dir_2 = np.abs(gamma_ij_i - gamma_ij_j)

    if delta_gamma_dir_1 < delta_gamma_dir_2:
        return np.mod(est_dir_1, 360)
    else:
        return np.mod(est_dir_2, 360)


def tsls_bearing_est(ant: sa.SectoredAntenna, sector_powers: List[float], noise_power_est: float, L: int=3) -> Tuple[int, float]:
    """
       Performs the Three-stage Simplified Least Squares (TSLS) algorithm as documented in [1]. This algorithm consists
       of three stages:
           1) Sector Selection - Finds the susbset L_k which contains the L sectors best suited to estimate the
              (SSL)              transmitter direction with. This is the L-2 sectors surrounding the Maximum Adjacent
                                 Sector Pair (MASP). The MASP are the two adjacent sectors which have the highest
                                 received power values.
           2) Sector-Pair DoA  - Pair up every possible combination of the sectors in L_k with non-negative
              Estimation (SDE)   noise-centred sector-powers (NCSP). I.E., these are sectors where the estimated noise
                                 power is not as strong as the signal from the transmitter. Then use the SLS method to
                                 estimate the direction-of-arrival (DoA) for every pair.
           3) DoA Fusion (DFU) - Fuse the DoA estimates from SDE step together to produce a single esimate of the DoA.
                                 NB: [1] doesn't provide a single, optimal way of doing this step and suggests three
                                 possible methods for fusing the estimates together. However, the authors remark that
                                 they found that the sector-power-weighting method they outline performed the best for
                                 their setup. As such, they used it for to evaluate the system.
       Once these steps have been completed, the function returns the estimate DoA in degrees. If the algorithm for some
       reason cannot return a valid estimate (either there were not enough sectors with non-negative sector-powers or
       all the valid sector pairs produced imaginary bearing estimations), [1] specifies a fall-back algorithm which is
       documented in [3] and uses maximum energy DoA estimation approach.

       Inputs:
           - ant:             The SectoredAntenna object representing the sector antenna model
           - sector_powers:   The receiver power for each sector of the antenna in decibels.
           - noise_power_est: An estimate of the noise power on the signal in decibels.
           - L:               The number of sectors to include in the DoA estimation. For L=2, the TSLS DoA estimation
                              algorithm performs the same as the Simplified Least Squares (SLS) algorithm documented in
                              [2]. The default value is 4 assuming that there are at least six sectors on the antenna.

       -----------------------
       References:
           [1] J. Werner et al., "Sectorized Antenna-based DoA Estimation and Localization: Advanced Algorithms and
               Measurements," in IEEE Journal on Selected Areas in Communications, vol. 33, no. 11, pp. 2272-2286,
               Nov. 2015, doi: 10.1109/JSAC.2015.2430292.
           [2] J. Werner, J. Wang, A. Hakkarainen, M. Valkama and D. Cabric, "Primary user localization in cognitive
               radio networks using sectorized antennas," 2013 10th Annual Conference on Wireless On-demand Network
               Systems and Services (WONS), 2013, pp. 155-161, doi: 10.1109/WONS.2013.6578341.
           [3] A. Hakkarainen et al., "Reconfigurable antenna based doa estimation and localization in cognitive radios:
               Low complexity algorithms and practical measurements," 2014 9th International Conference on Cognitive
               Radio Oriented Wireless Networks and Communications (CROWNCOM), 2014, pp. 454-459,
               doi: 10.4108/icst.crowncom.2014.255730.
    """
    # Stage 1: SSL
    L_k, masp_rssi = _tsls_ssl_step(ant, sector_powers, L)

    # Stage 2: SDE
    # Remove all sectors with a negative noise-centred sector-power (NCSP)
    remaining_sectors = []
    for sector in L_k:
        ncsp = su.db_to_nat(sector_powers[sector]) - su.db_to_nat(noise_power_est)
        if ncsp > 0:
            remaining_sectors.append((sector, ncsp))

    if len(remaining_sectors) < 2:
        # If there less than two sectors with a positive NCSP give up and return None. In [1] they specify a fall-back
        # algorithm to use which is described in [3].
        return maxE_bearing_est_hakkarainen(ant, sector_powers), masp_rssi

    unfused_doa_estimates = []
    # There are two different solutions to the SLS problem based on whether each sector of the antenna has equal
    # beamwidths. As such we need to treat them differently.
    for (i, ncsp_i), (j, ncsp_j) in itertools.permutations(remaining_sectors, 2):
        if ant.has_equal_beamwidths() or ant.beam_width(i) == ant.beam_width(j):
            gain_i = su.db_to_nat(ant.sector_gain(i, ant.sector_direction(i)))
            gain_j = su.db_to_nat(ant.sector_gain(j, ant.sector_direction(j)))

            dir_i = ant.sector_direction(i)
            dir_j = ant.sector_direction(j)

            # When dealing with sectors on the edge of wrapping around we need to be careful when we calculate
            # the average as there is a significant difference between the average of 360 and 350 and 0 and 350.
            if dir_i > 180 and dir_j < 180:
                dir_j += 360
            elif dir_j > 180 and dir_i < 180:
                dir_i += 360

            a = np.log(ncsp_i / ncsp_j) - (2 * np.log(gain_i / gain_j))
            kappa_ij = (ant.beam_width(i) ** 2.0) / (4 * (dir_i - dir_j))
            v_bar_ij = (dir_i + dir_j) / 2
            est_doa = v_bar_ij + kappa_ij * a
            unfused_doa_estimates.append((est_doa, (ncsp_i, ncsp_j)))
        else:
            if i != j:
                est_doa = _tsls_dbs_spdoa_func(ant, i, ncsp_i, j, ncsp_j)
                if est_doa is not None:
                    unfused_doa_estimates.append((est_doa, (ncsp_i, ncsp_j)))

    if len(unfused_doa_estimates) == 0:
        # If we dont have any estimates to fuse together, we just return None. [1] specifies that their system uses a
        # fall-back algorithm which is defined in [3].
        return maxE_bearing_est_hakkarainen(ant, sector_powers), masp_rssi
    elif len(unfused_doa_estimates) == 1:
        # If we only have one estimate, just return that as we don't need to fuse it.
        return unfused_doa_estimates[0][0], masp_rssi
    else:
        # We have more than one valid estimate for the DoA, so we need to fuse them together to produce a hopefully more
        # accurate estimate. Use the sector-power weighted fusion scheme described in section III-D-2 of [1].
        numerator = 0
        denominator = 0
        for est, (ncsp_i, ncsp_j) in unfused_doa_estimates:
            w_ij = ncsp_i * ncsp_j
            numerator += w_ij * np.sin(np.deg2rad(est))
            denominator += w_ij * np.cos(np.deg2rad(est))

        fused_doa = np.rad2deg(np.arctan2(numerator, denominator))
        return fused_doa, masp_rssi
