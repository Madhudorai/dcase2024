import numpy as np
import soundfile as sf
from scipy import signal
import os
import scipy.special as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.special import factorial, lpmv
from plot_tools import plot_SSL_results
import pyroomacoustics as pra

def soundread(sound_filepath):
    """
    Returns the contents of a sound file
    :param sound_filepath: path to sound_file to be read
    :return: (signal, sampling rate, number of channels)
    """
    mic_signals, fs = sf.read(sound_filepath, dtype='float32')
    if len(mic_signals.shape) == 1:
        num_channels = 1
    else:
        num_channels = mic_signals.shape[0]
    return mic_signals, fs, num_channels

def sph2cart(r, theta, phi):
    """
    Converts coordinate in spherical coordinates to Cartesian coordinates
    :param r: Radius
    :param theta: Azimuth angle
    :param phi: Inclination angle
    :return: Coordinates in Cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    car_coor = np.array([x, y, z])
    return car_coor


def detect_signal_type(signal,fs, threshold_ratio=0.5):
    """
    This function is used to determine if the signal is single frequency or multiple
    :param signal: input signal
    :param fs: sampling rate
    :param threshold_ratio:  The ratio between the maximum magnitude and other magnitudes in frequency domain
    :return: "Single" and its frequency or "Multiple"
    """
    window = np.hanning(len(signal))
    A = np.fft.fft(signal)
    #
    A = A[:len(A)//2]

    # Calculate the absolute value of the frequency components
    magnitude = np.abs(A)

    # Find the index of the largest frequency component
    max_index = np.argmax(magnitude)

    # Calculate the threshold (part of the maximum value)
    threshold = threshold_ratio * magnitude[max_index]

    # Remove maximum frequency components
    magnitude[magnitude == magnitude[max_index]] = 0

    # Find out if there are frequency components above the threshold in the remaining portion
    second_peak = np.max(magnitude)

    # If the remaining part has no frequency component exceeding the threshold, it is judged as a single-frequency signal, otherwise it is a multi-frequency signal
    if second_peak < threshold:
        frequency = max_index * fs / (len(A)*2)  # convert the index to actual frequency
        return "Single", frequency
    else:
        return "Multiple", None


def SphHarmonic(n, theta, phi):
    y = np.zeros(2*n+1, dtype=complex)
    for m in range(-n, n+1):
        temp = np.sqrt((2*n+1)/(4*np.pi) * factorial(n-abs(m)) / factorial(n+abs(m))) * np.exp(1j*m*phi)
        if m >= 0:
            y[m+n] = temp * lpmv(m, n, np.cos(theta))
        else:
            fm = -m
            y[m+n] = temp * lpmv(fm, n, np.cos(theta)) * (-1)**fm
    return y

    
def setRoom(room_dim, mic_arrays_car, source_pos_car_list, signal_list, typ, rt60_tgt=None):
    """
    The function `setRoom` sets up a room with specified dimensions, microphone arrays, source
    positions, and signals, and simulates the room acoustics either in an anechoic or reverberant
    environment.
    
    :param room_dim: The dimensions of the room in meters (length, width, height)
    :param mic_arrays_car: The `mic_arrays_car` parameter is a list of microphone array positions in
    Cartesian coordinates. Each element in the list represents the position of a microphone array. The
    position of each microphone array is represented as a 2D array, where each row represents the x, y,
    and z coordinates of a
    :param source_pos_car_list: The `source_pos_car_list` parameter is a list of source positions in
    Cartesian coordinates. Each element in the list represents the position of a source in 3D space
    :param signal_list: The `signal_list` parameter is a list of audio signals that will be used as the
    source signals in the simulation. Each element in the list represents a different source signal
    :param typ: The "typ" parameter specifies the type of room simulation to be performed. It can have
    two possible values: "Anechoic" or "Reverb"
    :param rt60_tgt: The parameter "rt60_tgt" represents the target reverberation time (RT60) for the
    room. RT60 is a measure of how quickly sound decays in a room, and it is commonly used to
    characterize the level of reverberation in a space. In this function, if the
    :return: two values: `room` and `rt60_est`.
    """
    if typ == "Anechoic":
        for array_id, mic_pos in enumerate(mic_arrays_car):
            room = pra.AnechoicRoom(fs=24000)
            mic_pos = mic_pos.transpose()
            room.add_microphone_array(mic_pos)
            for i, (source_pos_car, signal) in enumerate(zip(source_pos_car_list, signal_list)):
                room.add_source(source_pos_car, signal=signal, delay=0)
            room.simulate()
            output_dir = '/content/dcase2024/Anechoic/Array_output_{}/'.format(array_id)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for i in range(len(source_pos_car_list)):
                room.mic_array.to_wav(os.path.join(output_dir, 'source{}.wav'.format(i)), norm=True, bitdepth=np.float32)
        return room, None

    elif typ == "Reverb":
        for array_id, mic_pos in enumerate(mic_arrays_car):
            e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
            room = pra.ShoeBox(room_dim, fs=24000, materials=pra.Material(e_absorption), max_order=max_order)
            mic_pos = mic_pos.transpose()
            room.add_microphone_array(mic_pos)
            for i, (source_pos_car, signal) in enumerate(zip(source_pos_car_list, signal_list)):
                room.add_source(source_pos_car, signal=signal, delay=0)
            room.simulate()
            output_dir = '/content/dcase2024/Reverberant/Array_output_{}/'.format(array_id)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for i in range(len(source_pos_car_list)):
                room.mic_array.to_wav(os.path.join(output_dir, 'source{}.wav'.format(i)), norm=True, bitdepth=np.float32)
        # Estimate the real T60 using the pyroomacoustics function
        rt60_est = np.mean(room.measure_rt60())  # get the average value for all frequency bands
        return room, rt60_est
    



def ssl_SHmethod_broad(mic_signals, fs, mic_pos_sph, Theta_l, Phi_l, method, sphere_config, plot_method, resolution, num_sources=1):
    """
    This script is stored some classical SSL algorithms in the spherical domain for source localization
    :param mic_signals: The received signals from microphone array
    :param fs: Sampling frequency
    :param mic_pos_sph:  The spherical coordinate of microphone array
    :param Theta_l:  The elevation of the source
    :param Phi_l:  The azimuth of the source
    :param method:  The chosen algorithm
    :param sphere_config: The configuration of sphere
    :param plot_method: The method of plotting (2D or 3D)
    :param resolution: The resolution for the grid in the space
    :param num_sources: The number of sound sources
    :return:  A figure, estimate azimuth , estimate elevation
    """
    # Transform the inputs to lists
    if not isinstance(Theta_l, list):
        Theta_l = [Theta_l]
    if not isinstance(Phi_l, list):
        Phi_l = [Phi_l]
    radius = 0.042
    c = 343       # Velocity of sound
    # num_mics = mic_pos_sph.shape[0]
    num_mics = 32
    K = 2048  # The length of signal frame
    Mic_Theta = mic_pos_sph[:,1]
    Mic_Phi = mic_pos_sph[:,2]
    signal_single_channel = mic_signals[:, 0]
    signal_type, freq = detect_signal_type(signal_single_channel, fs,0.9)
    theta = np.arange(0, np.pi, resolution / 180 * np.pi)
    phi = np.arange(0, 2 * np.pi, resolution / 180 * np.pi)
    x = []
    y = []

    if signal_type == 'Single':
        N = 1
        ka = 2 * np.pi * 500 / c * radius

        FrameNumber = mic_signals.shape[0] // K
        # label the effective data
        FrameFlag = np.zeros(FrameNumber + 1)
        count = 0
        flags = []
        for num in range(FrameNumber + 1):
            # determine this frame if is effective
            if np.sum(mic_signals[(num - 1) * K:num * K, 0] ** 2) > 10 * 1e-5:
                FrameFlag[num] = 1
                count += 1
                # Append the beginning position of effective frame
                flags.append((num - 1) * K + 1)
        # Calculate bn(ka)
        if sphere_config == "rigid":
            bn = np.zeros(N+1, dtype=complex)
            for n in range(N+1):
                jn_ka = sp.spherical_jn(n, ka)
                jn_ka_der = sp.spherical_jn(n, ka, derivative=True)
                yn_ka = sp.spherical_yn(n, ka)
                yn_ka_der = sp.spherical_yn(n, ka, derivative=True)

                # Compute the second kind of Hankel function and its derivative
                hn2_ka = jn_ka - 1j * yn_ka
                hn2_ka_der = jn_ka_der - 1j * yn_ka_der

                bn[n] = 4 * np.pi * 1j ** n * (jn_ka - jn_ka_der / hn2_ka_der * hn2_ka)

        if sphere_config == "open":
            bn = np.zeros(N+1, dtype=complex)
            for n in range(N+1):
                bn[n] = 4 * np.pi * 1j ** n * sp.spherical_jn(n, ka)

        # Calculate spherical harmonics
        Y_nm = np.zeros(((N+1)**2, num_mics), dtype=complex)
        for num in range(num_mics):
            for n in range(N + 1):
                Y_nm[n ** 2:(n + 1) ** 2, num] = SphHarmonic(n, Mic_Theta[num], Mic_Phi[num])

        # Define the signal in the time domain and freqeuncy domain
        x_p = np.zeros((K, num_mics), dtype=complex)
        X = np.zeros((K, num_mics), dtype=complex)

        for m in range(num_mics):
            x_p[:, m] = mic_signals[flags[0]:(flags[0] + K), m]
            X[:, m] = np.fft.fft(x_p[:, m])
        X_half = X[:K // 2, :]  # Keep only first half of spectrum

        I = int(np.floor(500 / fs * K)) + 1# Spectral line positions corresponding to single frequencies


        # Do spherical harmonic transform
        p_nm = 4 * np.pi / num_mics * Y_nm.conj() @ X_half.T
        if method == 'SHMVDR':
            sphCOV = np.dot(p_nm, p_nm.T.conj())
            lambda_reg = 1e-2  # 正则化因子，需要根据你的应用进行调整
            sphCOV += lambda_reg * np.eye(sphCOV.shape[0])
            # print(sphCOV.shape)
            out = []
            P_mvdr = np.zeros((len(theta), len(phi)), dtype=complex)  # MUSIC谱
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    U_nm = np.zeros(((N + 1) ** 2), dtype=complex)
                    for n in range(N + 1):
                        U_nm[n ** 2:(n + 1) ** 2] = bn[n] * SphHarmonic(n, theta[num1], phi[num2]).conj()
                    U_nm = U_nm[:, np.newaxis]
                    invA_b = np.linalg.solve(sphCOV, U_nm)
                    b_invA_b = np.dot(U_nm.T.conj(), invA_b)
                    w_mvdr = invA_b / b_invA_b
                    P_mvdr[num1, num2] = w_mvdr.T.conj() @ sphCOV @ w_mvdr

            # Convert to dB and clip values below -22 dB
            out = 10 * np.log10(np.abs(P_mvdr))
            out = out - np.max(out)
            x, y = np.where(out == np.max(out))
            source_positions = [(i * resolution, j * resolution) for i, j in zip(x, y)]

        if method == "SHMUSIC":
            Bn = np.zeros(((N + 1) ** 2, 1), dtype=complex)
            for n in range(N + 1):
                Bn[n ** 2:(n + 1) ** 2] = bn[n]
            a_nm = p_nm / Bn
            S = np.dot(a_nm, a_nm.T.conj())
            # Calculate the Eigenvalue and Eigenvector
            D, V = np.linalg.eigh(S)
            # Sort the eigenvalue and get the index
            i = np.argsort(D)
            Y = np.diag(D)[i]
            # Calculate the noise subspace
            E = V[:, i[:-num_sources]]
            P_music = np.zeros((len(theta), len(phi)), dtype=complex)  # MUSIC谱
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    y_nm = np.zeros(((N + 1) ** 2), dtype=complex)
                    for n in range(N + 1):
                        y_nm[n ** 2:(n + 1) ** 2] = SphHarmonic(n, theta[num1], phi[num2]).T
                    P_music[num1, num2] = 1 / (y_nm @ E @ E.T.conj() @ y_nm.T.conj())

            # Convert to dB and clip values below -22 dB
            out = 10 * np.log10(np.abs(P_music))
            out = out - np.max(out)
            x, y = np.where(out == np.max(out))
            source_positions = [(i * resolution, j * resolution) for i, j in zip(x, y)]

        if method == "SHMLE":
            out = []
            Out = np.zeros((len(theta), len(phi)), dtype=complex)
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    D_nm = np.zeros(((N + 1) ** 2), dtype=complex)
                    for n in range(N + 1):
                        D_nm[n ** 2:(n + 1) ** 2] = bn[n] * SphHarmonic(n, theta[num1], phi[num2]).conj()
                    D_nm = D_nm[:, np.newaxis]
                    Out[num1, num2] = np.linalg.norm(p_nm - D_nm @ np.linalg.pinv(D_nm) @ p_nm)

            out = -20 * np.log10(np.abs(Out))
            out = out - np.max(out)
            x, y = np.where(out == np.max(out))
            out = np.clip(out, -10, None)
            indices = []
            out_copy = np.copy(out)
            for _ in range(num_sources):
                max_index = np.argmax(out_copy)
                indices.append(np.unravel_index(max_index, out_copy.shape))
                out_copy[indices[-1]] = -np.inf

            # Convert the indices to the desired resolution
            source_positions = [(i * resolution, j * resolution) for i, j in indices]

        plot_SSL_results(out, Theta_l, Phi_l, plot_method, method, signal_type, sphere_config, vmin_value=None,
                         vmax_value=None, source_est=source_positions)
        return out, source_positions



    if signal_type == "Multiple":
        N = 4
        # Calculate the range of frequency: ka
        freq_up = round(K*c*N/(fs*2*np.pi*radius))
        freq_low = round(freq_up/2)+1

        ## divide the received signal into frames
        FrameNumber = mic_signals.shape[0] // K
        # label the effective data
        FrameFlag = np.zeros(FrameNumber + 1)
        count = 0
        flags = []
        for num in range(FrameNumber + 1):
            # determine this frame if is effective
            if np.sum(mic_signals[(num - 1) * K:num * K, 0] ** 2) > 10 * 1e-5:
                FrameFlag[num] = 1
                count += 1
                # Append the beginning position of effective frame
                flags.append((num - 1) * K + 1)

        if sphere_config == "rigid":
            # Calculate Bn
            bn = np.zeros(((N + 1) ** 2, freq_up), dtype=complex)

            for k in range(freq_up):
                ka = 2 * np.pi * k / K * fs / c * radius
                for n in range(N + 1):
                    jn_ka = sp.spherical_jn(n, ka)
                    jn_ka_der = sp.spherical_jn(n, ka, derivative=True)
                    yn_ka = sp.spherical_yn(n, ka)
                    yn_ka_der = sp.spherical_yn(n, ka, derivative=True)

                    # Compute the second kind of Hankel function and its derivative
                    hn2_ka = jn_ka - 1j * yn_ka
                    hn2_ka_der = jn_ka_der - 1j * yn_ka_der

                    bn[n ** 2:(n + 1) ** 2, k] = 4 * np.pi * (1j) ** n * (jn_ka - jn_ka_der / hn2_ka_der * hn2_ka)

        if sphere_config == "open":
            # Calculate Bn
            bn = np.zeros(((N + 1) ** 2, freq_up), dtype=complex)

            for k in range(freq_up):
                ka = 2 * np.pi * k / K * fs / c * radius
                for n in range(N + 1):
                    bn[n ** 2:(n + 1) ** 2, k] = 4 * np.pi * (1j) ** n * sp.spherical_jn(n, ka)

        # Calculate spherical harmonics
        Y_nm = np.zeros(((N+1)**2, num_mics), dtype=complex)
        for n in range(N+1):
            for m in range(num_mics):
                Y_nm[n ** 2:(n + 1) ** 2, m] = SphHarmonic(n, Mic_Theta[m], Mic_Phi[m])

        # Define the signal in the time domain and freqeuncy domain
        # 将信号变换到频域
        x_p = np.zeros((K, num_mics), dtype=complex)
        X = np.zeros((K, num_mics), dtype=complex)
        for m in range(num_mics):
            x_p[:, m] = mic_signals[flags[10]:(flags[10] + K), m]
            X[:, m] = np.fft.fft(x_p[:, m])

        # Do spherical transform
        p_nm = np.zeros((freq_up, (N + 1) ** 2), dtype=complex)
        for k in range(int(freq_low-1),int(freq_up)):
            p_nm[k, :] = 4 * np.pi / num_mics * X[k+1, :].dot(Y_nm.T.conj())
        p_nm = p_nm.T


        if method == "PWD":
            Out = np.zeros((len(theta), len(phi)), dtype=complex)
            B_n = np.zeros(((N + 1) ** 2), dtype=complex)
            D_nm = np.zeros(((N + 1) ** 2, 1), dtype=complex)
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    temp = 0
                    for n in range(N + 1):
                        D_nm[n ** 2:(n + 1) ** 2, 0] = SphHarmonic(n, theta[num1], phi[num2])
                    for k in range(int(freq_low - 1), int(freq_up)):
                        a_nm = np.diag(1 / bn[:, k]) @ p_nm[:, k]
                        temp += np.linalg.norm(D_nm.T @ a_nm)
                    Out[num1, num2] = temp
            # Convert to dB and clip values below -22 dB
            out = 20 * np.log10(np.abs(Out))
            out = out - np.max(out)
            out = np.clip(out, -10, None)
            x, y = np.where(out == np.max(out))
            source_positions = [(i * resolution, j * resolution) for i, j in zip(x, y)]

        if method == "DAS":
            Out = np.zeros((len(theta), len(phi)), dtype=complex)
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    temp = 0
                    D_nm = np.zeros(((N + 1) ** 2, 1), dtype=complex)
                    for n in range(N + 1):
                        D_nm[n ** 2:(n + 1) ** 2, 0] = SphHarmonic(n, theta[num1], phi[num2]).conj()
                    for k in range(int(freq_low - 1), int(freq_up)):
                        d_nm = np.diag(bn[:, k]) @ D_nm
                        temp += np.linalg.norm(d_nm.T.conj() @ p_nm[:, k])
                    Out[num1, num2] = temp
            # Convert to dB and clip values below -22 dB
            out = 20 * np.log10(np.abs(Out))
            out = out - np.max(out)
            out = np.clip(out, -10, None)
            x, y = np.where(out == np.max(out))
            source_positions = [(i * resolution, j * resolution) for i, j in zip(x, y)]


        if method == "SHMVDR":
            lambda_reg = 1e-2
            P_mvdr = np.zeros((len(theta), len(phi)), dtype=complex)
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    temp = 0
                    y_nm = np.zeros(((N + 1) ** 2), dtype=complex)
                    for n in range(N + 1):
                        y_nm[n ** 2:(n + 1) ** 2] = SphHarmonic(n, theta[num1], phi[num2]).conj()
                    for k in range(int(freq_low - 1), int(freq_up)):
                        U_nm = np.diag(bn[:, k]) @ y_nm
                        U_nm = U_nm[:, np.newaxis]
                        sphCOV = np.outer(p_nm[:,k], p_nm[:,k].T.conj())
                        Lambda_reg = 1e-2
                        sphCOV += lambda_reg * np.eye(sphCOV.shape[0])
                        invA_b = np.linalg.solve(sphCOV , U_nm)
                        b_invA_b = np.dot(U_nm.T.conj(), invA_b)
                        w_mvdr = invA_b / b_invA_b
                        temp += np.linalg.norm(w_mvdr.T.conj() @ sphCOV @ w_mvdr)
                    P_mvdr[num1, num2] = temp

            # Convert to dB and clip values below -22 dB
            out = 10 * np.log10(np.abs(P_mvdr))
            out = out - np.max(out)
            out = np.clip(out, -10, None)

            indices = []
            out_copy = np.copy(out)
            for _ in range(num_sources):
                max_index = np.argmax(out_copy)
                indices.append(np.unravel_index(max_index, out_copy.shape))
                out_copy[indices[-1]] = -np.inf

            # Convert the indices to the desired resolution
            source_positions = [(i * resolution, j * resolution) for i, j in indices]


        if method == "SHMUSIC":
            Out = np.zeros((len(theta), len(phi)), dtype=complex)
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    temp = 0
                    y_nm = np.zeros(((N + 1) ** 2), dtype=complex)
                    for n in range(N + 1):
                        y_nm[n ** 2:(n + 1) ** 2] = SphHarmonic(n, theta[num1], phi[num2])
                    for k in range(int(freq_low - 1), int(freq_up)):
                        a_nm = np.diag(1 / bn[:, k]) @ p_nm[:, k]
                        S = np.outer(a_nm, a_nm.T.conj())
                        D, V = np.linalg.eigh(S)
                        i = np.argsort(D)
                        E = V[:, i[:-num_sources]]
                        temp += np.linalg.norm(y_nm.T @ E @ E.T.conj() @ y_nm.conj())
                    Out[num1, num2] = temp

            # Convert to dB and clip values below -22 dB
            out = -10 * np.log10(np.abs(Out))
            out = out - np.max(out)
            out = np.clip(out, -10, None)

            indices = []
            out_copy = np.copy(out)
            for _ in range(num_sources):
                max_index = np.argmax(out_copy)
                indices.append(np.unravel_index(max_index, out_copy.shape))
                out_copy[indices[-1]] = -np.inf

            # Convert the indices to the desired resolution
            source_positions = [(i * resolution, j * resolution) for i, j in indices]

        if method == "SHMLE":
            Out = np.zeros((len(theta), len(phi)), dtype=complex)
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    temp = 0
                    P_nm = np.zeros((N + 1) ** 2, dtype=complex)
                    for n in range(N + 1):
                        P_nm[n ** 2:(n + 1) ** 2] = SphHarmonic(n, theta[num1], phi[num2])
                    for k in range(int(freq_low - 1), int(freq_up)):
                        d_nm = np.diag(bn[:, k]) @ np.conj(P_nm)
                        d_nm = d_nm[:, np.newaxis]
                        temp += np.linalg.norm(p_nm[:, k] - d_nm @ np.linalg.pinv(d_nm) @ p_nm[:, k]) ** 2
                    Out[num1, num2] = temp

            # Convert to dB and clip values below -22 dB
            out = -10 * np.log10(np.abs(Out))
            out = out - np.max(out)
            out = np.clip(out, -10, None)
            x, y = np.where(out == np.max(out))
            source_positions = [(i * resolution, j * resolution) for i, j in zip(x, y)]


        plot_SSL_results(out, Theta_l, Phi_l, plot_method, method, 'Real Data', sphere_config, vmin_value=None,
                              vmax_value=None, source_est=source_positions)
        return out, source_positions
    

def ssl_SHmethod_broad2(ambisonic_signals, fs, Theta_l, Phi_l, method, plot_method, resolution, num_sources=1):
    """
    This script implements SSL algorithms in the spherical domain using B-format input
    :param ambisonic_signals: B-format signals (W,X,Y,Z) with shape (num_frames, 4)
    :param fs: Sampling frequency
    :param Theta_l: The elevation of the source
    :param Phi_l: The azimuth of the source
    :param method: The chosen algorithm
    :param plot_method: The method of plotting (2D or 3D)
    :param resolution: The resolution for the grid in the space
    :param num_sources: The number of sound sources
    :return: A figure, estimate azimuth, estimate elevation
    """
    # Transform the inputs to lists
    if not isinstance(Theta_l, list):
        Theta_l = [Theta_l]
    if not isinstance(Phi_l, list):
        Phi_l = [Phi_l]

    # Constants
    c = 343  # Velocity of sound
    K = int(fs * 0.1)  # Frame length for 100ms hop sizframe_length = int(fs * 0.1)  # 100ms = 0.1 sec
    num_samples = ambisonic_signals.shape[0]
    num_frames = num_samples // K
    
    # Create search grid
    theta = np.arange(0, np.pi, resolution / 180 * np.pi)
    phi = np.arange(0, 2 * np.pi, resolution / 180 * np.pi)
    
    # Process only the 10th frame
    frame_idx = 10
    if frame_idx >= num_frames:
        frame_idx = num_frames - 1  # Use last frame if 10th frame doesn't exist
    
    # Get B-format coefficients for the 10th frame
    W = ambisonic_signals[frame_idx, 0]  # 0th order
    X = ambisonic_signals[frame_idx, 1]  # 1st order x
    Y = ambisonic_signals[frame_idx, 2]  # 1st order y
    Z = ambisonic_signals[frame_idx, 3]  # 1st order z
    
    # Initialize output
    Out = np.zeros((len(theta), len(phi)), dtype=complex)
    
    if method == "DAS":
        for num1 in range(len(theta)):
            for num2 in range(len(phi)):
                # Calculate steering vector for current direction
                # For first-order Ambisonics, we use the spherical harmonics directly
                Y00 = 1  # 0th order
                Y1m1 = np.sin(theta[num1]) * np.cos(phi[num2])  # 1st order x
                Y10 = np.cos(theta[num1])  # 1st order z
                Y11 = np.sin(theta[num1]) * np.sin(phi[num2])  # 1st order y
                
                # Calculate beamformer output
                temp = (W * Y00 + X * Y1m1 + Y * Y11 + Z * Y10)
                Out[num1, num2] = np.abs(temp)
    
    # Convert to dB and clip values
    out = 20 * np.log10(np.abs(Out))
    out = out - np.max(out)
    out = np.clip(out, -10, None)
    
    # Find source positions
    x, y = np.where(out == np.max(out))
    source_positions = [(i * resolution, j * resolution) for i, j in zip(x, y)]
    
    # Plot results
    plot_SSL_results(out, Theta_l, Phi_l, plot_method, method, 'Real Data', 'open', vmin_value=None,
                    vmax_value=None, source_est=source_positions)
    
    return out, source_positions
  