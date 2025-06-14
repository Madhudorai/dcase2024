import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
import matplotlib.cm as cm
import librosa
import librosa.display


"""
plot_tools: This script contains several tools for plotting, i.e. in time domain, frequency domain, STFT domain
"""
def plot_wave(signal, fs):
    """
    Plot the wave in time domain

    :param signal: signal in time domain
    :param fs: sampling rate of signal
    """
    time = np.arange(len(signal)) / fs
    # plot the wave
    plt.figure()
    plt.plot(time, signal)
    plt.title('Waveform in time domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

def plot_stft(signals, fs):
    """
    The function `plot_stft` takes in a signal and its sampling rate, computes the Short-Time Fourier
    Transform (STFT), and plots the magnitude spectrogram.
    
    :param signals: The signals parameter is the input audio signal that you want to analyze using the
    Short-Time Fourier Transform (STFT). It can be a 1-dimensional array representing the audio waveform
    :param fs: The parameter "fs" represents the sampling rate of the audio signal. It is the number of
    samples per second in the audio signal
    """
    # Transfer from time domain to stft domain
    D = librosa.stft(signals)
    # Decompose it to the magnitude and phase
    magnitude, phase = librosa.magphase(D)
    # plot the STFT
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), sr=fs, x_axis='time', y_axis='log')

    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar = plt.colorbar(format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show()

def plot_option(room, mic_arrays_car, source_pos_car_list):
    """
    Plot the microphone array in the room
    :param room:
    :param mic_arrays_car: The list of cartesian positions of microphone arrays
    :param source_pos_car_list: The list of cartesian positions of sound sources
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the room
    for wall in room.walls:
        corners = wall.corners.T
        ax.plot(corners[[0, 1, 2, 3, 0], 0], corners[[0, 1, 2, 3, 0], 1], corners[[0, 1, 2, 3, 0], 2], 'k')

    # Plot the microphone array
    for i, mic_pos_car in enumerate(mic_arrays_car):
        ax.scatter(mic_pos_car[:, 0], mic_pos_car[:, 1], mic_pos_car[:, 2], c='g', marker='o',
               label=f'Microphone Array {i+1}')
    # Plot the source locations
    for i, source_pos_car in enumerate(source_pos_car_list):
        ax.scatter(source_pos_car[0], source_pos_car[1], source_pos_car[2], c='r', marker='x', label=f'Source {i+1}', s=100)
    # Set axis labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    # Show the legend
    ax.legend()

    # Show the plot
    plt.show()

def plot_SSL_results(out, Theta_l_rad_list, Phi_l_rad_list, plot_type, method, signal_type, sphere_config, vmin_value=None, vmax_value=None, source_est=None):
    """
    This script is used to plot SSL algorithm result
    :param out: The output of algorithm
    :param Theta_l_rad_list:  A list of the real sound source elevations
    :param Phi_l_rad_list:  A list of the real sound source azimuths
    :param plot_type:  The type of plotting
    :param method:  The name of algorithm
    :param signal_type:  The type of signal
    :param sphere_config: The configuration of the sphere
    :param vmin_value: The minimum spectrum
    :param vmax_value: The maximum spectrum
    :param source_est: The estimated source position
    :return: A plot
    """
    Theta_l_deg_list = [angle * 180 / np.pi for angle in Theta_l_rad_list]
    Phi_l_deg_list = [angle * 180 / np.pi for angle in Phi_l_rad_list]
    if plot_type == "2D":
        plt.figure(figsize=(10, 8), dpi=250)
        im = plt.imshow(out, origin='lower', extent=[-180, 180, -90, 90], aspect='auto', cmap='jet', vmin=vmin_value,
                   vmax=vmax_value)
        cb = plt.colorbar(im, label='[dB]')
        cb.ax.tick_params(labelsize=16)  
        cb.set_label('[dB]', size=25)  
        plt.xlabel(r'$\phi$ [deg]', fontsize=25)
        plt.ylabel(r'$\theta$ [deg]', fontsize=25)
        plt.xticks(fontsize=18)  
        plt.yticks(fontsize=18)  
        for Theta_l_deg, Phi_l_deg in zip(Theta_l_deg_list, Phi_l_deg_list):
            plt.plot(Phi_l_deg, Theta_l_deg, 'bx', markersize=10, markeredgewidth=3, label='Real Source Position')
        if source_est is not None:
            for idx, est in enumerate(source_est):
                label = 'Estimated Source Position' if idx == 0 else ""  
                plt.plot(est[1], est[0], 'ro', markersize=10, fillstyle='none', markeredgewidth=3, label=label)
        plt.legend(loc='upper right', fontsize=16)
        plt.show()

    elif plot_type == "3D":
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        phi, theta = np.meshgrid(np.linspace(-np.pi, np.pi, 360), np.linspace(-np.pi/2, np.pi/2, 180))
        phi_deg = np.rad2deg(phi)
        theta_deg = np.rad2deg(theta)

        surface = ax.plot_surface(phi_deg, theta_deg, out, cmap='jet')
        cb = fig.colorbar(surface, label='[dB]', shrink=0.5, aspect=5)
        cb.ax.tick_params(labelsize=16)  
        cb.set_label('[dB]', size=18)  
        ax.set_xlabel('Phi [deg]', fontsize=18)
        ax.set_ylabel(r'$\theta$ [deg]', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)  

        for Theta_l_deg, Phi_l_deg in zip(Theta_l_deg_list, Phi_l_deg_list):
            source = ax.scatter3D(Phi_l_deg, Theta_l_deg, np.max(out), color='ForestGreen', s=100, edgecolor='DarkBlue',
                                  linewidth=1.5, label='Real Source Position')
            ax.text(Phi_l_deg, Theta_l_deg, np.max(out),
                    '({:.1f}, {:.1f}, {:.1f})'.format(Phi_l_deg, Theta_l_deg, np.max(out)), color='ForestGreen',
                    fontsize=14)

        if source_est is not None:
            for idx, est in enumerate(source_est):
                label = 'Estimated Source Position' if idx == 0 else ""
                est_source = ax.scatter3D(est[1], est[0], np.max(out), color='DarkRed', s=100, edgecolor='DarkBlue',
                                          linewidth=1.5, label=label, marker='o', facecolors='none')

        plt.legend()
        plt.show()

    else:
        print("Invalid plot type. Please select either '2D' or '3D'.")

        
def plot_spherical_grid(resolution):
    """
    Plots a spherical grid with a given resolution.
    
    :param resolution: The resolution in degrees for the grid points
    """
    # Create meshgrid for the polar and azimuthal angles
    theta, phi = np.mgrid[0:np.pi:complex(0, 180/resolution), 0:2*np.pi:complex(0, 360/resolution)]

    # Convert polar and azimuthal angles to cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the title for the subplot
    ax.set_title(f'Simulated grid for possible points in the space with grid resolution = {resolution}°')

    # Show the plot
    plt.show()


