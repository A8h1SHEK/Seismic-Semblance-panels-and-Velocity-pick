"""Copyright 2024 Abhishek Singh Chauhan"""

import subprocess

libraries = ['segysak', 'pykonal']
for lib in libraries:
    try:
        __import__(lib)
    
    except ImportError:
        subprocess.call(['pip', 'install', lib])

# If having some error in istalling pykonal run the command on terminal :  conda update conda 
# and then try to install again

# Libraries for computation 
import numpy as np
from joblib import Parallel, delayed
import os
from numpy import sum
import time
from scipy.interpolate import CubicSpline
import pandas as pd
import collections

# Libraries for plotting and interfaces
import tqdm.auto
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pkg_resources

# Library for loading segy files
import segysak
from segysak import segy
from segysak.segy import segy_loader, segy_header_scrape

# Library for solving ekional eq.
import pykonal


class Semblance:
    
    """This class of Semblance consist of various steps, like looking for the various headers , loading the segy files, visualizing shots,
       cmp sorting and visualizing it, calculation of semblance panels and visualizing it, travel-time calculation, ray-tracing,
       nmo-correction and it's visualization."""

    path = "D:/germany/seismic.segy"  # path of segy file

    def __init__(self, header=None):
        if header!=None:
            self.header = header

    def text_header(self):
        """
        Input : 
            path : str
                takes path of segy file as input

        Returns:
            text_header : text headrs
        """        
        return segy.get_segy_texthead(Semblance.path)

    def bin_scrap(self):
        """
        Input : 
            path : str
                takes path of segy file as input

        Returns:
            values in bin : dict
                            In the form of dictonary 
        """
        return segy.segy_bin_scrape(Semblance.path)

    def header_scan(self):
        """
        Input : 
            path : str
                takes path of segy file as input

        Returns:
            header_values_byte_locations : DataFrame 
                                           header values byte locations in pandas DataFrame
        """
        return segy.segy_header_scan(Semblance.path)

    def header_scrap(self):
        """
        Input : 
            path : str
                takes path of segy file as input

        Returns:
            header values  : DataFrame
                             header values in pandas DataFrame
        """
        return segy_header_scrape(Semblance.path)

    def get_header_values(self):
        """
        Input : 
            header name : str
                        which header info  you want

        Returns:
            header values : np.ndarray 
                            ndarray of header value
        """
        return Semblance.header_scrap(self)[self.header].to_numpy()

    def loader(self):
        """
        Input : 
            path : str
                takes path of segy file as input

        Returns:
            gather : xarray
                     Contain all data and information in the form of xarray 
        """
        return segy_loader(Semblance.path, vert_domain='TWT')

    @staticmethod
    def shot_plotter(shot_gather : np.ndarray, shot_no : int, offset_traces : np.ndarray, trace_number : np.ndarray, times: np.ndarray):
        """
        Input : 
            shot_gather : np.ndarray
                        Contain all the traces corresponding to the shots in increasing order.
            
            shot_no :  int
                    Shot number which you want to plot
            
            offset_traces : np.ndarray
                            array containing offsets of all the traces
            
            trace_number : np.ndarray
                        array containing tracl no. of all the traces
            
            times : np.ndarray
                    array containing recording time of the shots

        Returns:
            plot : plt
                     Plot of a shot with resperct to tracl and offsets
        """
        
        
        nrcv = 120
        trace_number = trace_number - 1
        if shot_no == 1:
            shot = shot_gather[:, :nrcv]
            tr_no0 = trace_number[0]
            tr_no1 = trace_number[nrcv-1]
            off_tr0 = offset_traces[0]
            off_tr1 = offset_traces[nrcv-1]
        else:
            shot = shot_gather[:, (shot_no - 1)*nrcv:shot_no*nrcv]
            tr_no0 = trace_number[nrcv*(shot_no - 1)]
            tr_no1 = trace_number[nrcv*shot_no - 1]
            off_tr0 = offset_traces[nrcv*(shot_no - 1)]
            off_tr1 = offset_traces[nrcv*shot_no - 1]
        
        fig = plt.figure(figsize=(15, 8))
        
        vmin, vmax = np.percentile(shot, [15, 85])

        ax = plt.subplot(121)

        ax.set_title(f'for Shot = {shot_no}',fontsize=13)
        ax.set_xlabel('trace number',fontsize=15)
        ax.set_ylabel('time (sec) ',fontsize=15)
        ax.imshow(shot, extent=[tr_no0, tr_no1, times[-1], 0],
          aspect='auto', cmap='Greys', vmin=vmin, vmax=vmax)

        ax = plt.subplot(122)
        ax.set_title(f'for Shot = {shot_no}',fontsize=13)
        ax.set_xlabel('offsets (m) ',fontsize=15)
        ax.set_ylabel('time (sec) ',fontsize=15)
        ax.imshow(shot, extent=[off_tr0, off_tr1, times[-1], 0],
          aspect='auto', cmap='Greys', vmin=vmin, vmax=vmax)

        plt.show()

    @staticmethod
    def cdp_trace(traces : np.ndarray, cdp_traces : np.ndarray, cdp_no : int) -> np.ndarray :
        """
        Input : 
            traces : np.ndarray
                    array containing tracl no. of all the traces
            
            cdp_traces : np.ndarray
                        array containing cdp no. of all the traces
                        
            cdp_no: int
                    unique cdp no. corresponding to which you want trace numbers

        Returns:
            cdp_traces : np.ndarray
                        Contain cdp_trace no. corresponding to the unique cdp no.
        """
        return traces[cdp_traces==cdp_no]

    @staticmethod
    def cdp_offset(offset_traces : np.ndarray, cdp_traces : np.ndarray, cdp_no : int) -> np.ndarray :
        """
        Input : 
            offset_traces : np.ndarray
                    array containing offsets of all the traces
            
            cdp_traces : np.ndarray
                        array containing cdp no. of all the traces
                        
            cdp_no: int
                    unique cdp no. corresponding to which you want trace numbers

        Returns:
            cdp_offset : np.ndarray
                        Contain cdp_offset's corresponding to the unique cdp no.
        """
        return offset_traces[cdp_traces==cdp_no]

    @staticmethod
    def cdp_sort(shot_gather : np.ndarray, cdp_trace : np.ndarray) -> np.ndarray :
        """
        Input : 
            shot_gather : np.ndarray
                    array containing offsets of all the traces
            
            cdp_traces : np.ndarray
                        array containing tracl no's. for a unique cdp no.
                        
        Returns:
            cdp_gather : np.ndarray
                        Contain cdp_gather corresponding to the unique cdp no. for which the tracl no. are the cdp_traces
        """
        return shot_gather[:,cdp_trace]

    @staticmethod
    def cdp_plotter(cdp : np.ndarray, cdp_traces : np.ndarray, cdp_offsets : np.ndarray, cdp_no : int, times : np.ndarray):
        """
        Input : 
            cdp : np.ndarray
                  Contain cdp gather for a unique cdp.
            
            cdp_traces :  np.ndarray
                          array containing tracl no's. for a unique cdp no.
            
            cdp_offsets : np.ndarray
                          Contain cdp_offset's corresponding to the unique cdp no.
            
            cdp_no : int
                     unique cdp no. corresponding to which you want trace numbers
            
            times : np.ndarray
                    array containing recording time of the shots

        Returns:
            plot : plt
                     Plot of a cdp gather with resperct to tracl and offsets
        """
        
        fig = plt.figure(figsize=(15, 8))
        vmin, vmax = np.percentile(cdp, [15, 85])

        ax = plt.subplot(121)

        ax.set_title(f'CMP {cdp_no}',fontsize=30)
        ax.set_xlabel('trace number',fontsize=20)
        ax.set_ylabel('time (sec)' ,fontsize=20)
        ax.imshow(cdp, extent=[cdp_traces[0], cdp_traces[-1], times[-1], 0],
                  aspect='auto', cmap='Greys', vmin=vmin, vmax=vmax)
        
        ax = plt.subplot(122)
        ax.set_title(f'CMP {cdp_no}',fontsize=30)
        ax.set_xlabel('offsets (m) ',fontsize=20)
        ax.set_ylabel('time (sec) ',fontsize=20)
        ax.imshow(cdp, extent=[cdp_offsets[0], cdp_offsets[-1], times[-1], 0],
                  aspect='auto', cmap='Greys', vmin=vmin, vmax=vmax)
        plt.show()

    @staticmethod
    def velocity(min_vel : int, max_vel : int, interval : int) -> np.ndarray :
        """
        Input : 
            min_vel : int
                      minimum value of the velocity you want
            
            max_vel : int
                      maximum value of the velocity you want
                        
            interval : int
                       step in velocity array you want

        Returns:
            velocity : np.ndarray
                       Velocity array containg the velocity
        """
        return np.arange(min_vel, max_vel, interval)

    @staticmethod
    def reflection_time(time : float, offset : np.ndarray, velocity : float) -> np.ndarray :
        """
        Input : 
            time : float
                   zeor-offset time 
            
            offset : np.ndarray
                     array of offset's
                        
            velocity : float
                       a value of velocity 

        Returns:
            reflection_time : np.ndarray
                              array containg the reflection time corresponding to a time, velocity and array of offset's
        """
        t = np.sqrt(time**2 + offset**2/velocity**2)
        return t

    @staticmethod
    def interpolate_amplitude(cmp : np.ndarray, trajectory : np.ndarray, sample_rate : float) -> np.ndarray :
        """
        Input : 
            cmp : np.ndarray
                cdp gather for a unique cdp
            
            trajectory : np.ndarray
                        reflection time corresponding to a time, velocity and array of offset's
                        
            sample_rate : float
                        sampling interval

        Returns:
            interpolated_amplitude : np.ndarray
                                    array containg the amplitude's at reflection time (trajectory)
        """  

        amp = np.zeros(len(trajectory))
        n_samples, n_traces = cmp.shape
        dt = sample_rate

        time = trajectory/dt

        traj = time.astype('i')

        mask = (traj < n_samples - 1 ) & ( traj >= 0 )

        t_before = traj
        t_after = t_before + 1
        ntr = np.arange(n_traces)

        amp0 = cmp[t_before[mask], ntr[mask]]
        amp1 = cmp[t_after[mask], ntr[mask]]

        interp = (amp0 * (time - t_after)[mask])/ ((t_before - t_after)[mask]) + (amp1 * (time - t_before)[mask]) / ((t_after - t_before)[mask])

        return interp

    @staticmethod
    def semblance_IA(cmp : np.ndarray, times : np.ndarray, offsets : np.ndarray, velocities : np.ndarray, window : int, sample_rate : float) -> np.ndarray :
        """
        Input : 
            cmp : np.ndarray
                  Contain cdp gather for a unique cdp.
                  
            times : np.ndarray
                    array containing recording time of the shots
                    
            offsets : np.ndarray
                      Contain cdp_offset's corresponding to the unique cdp no.
                          
            velocities : np.ndarray
                         Velocity array containg the velocity
            
            window : int
                     sample window or time window length over which you want to compute semblance
                     
            sample_rate : float
                          sampling interval
            
        Returns:
            semblance : np.ndarray
                        Contain semblance corresponding to the cdp gather.
        """

        n_samples, n_traces = cmp.shape
        semb = np.zeros((len(times), len(velocities)))
        l1 = np.arange(-window//2, 0)
        l2 = np.arange(0, window//2)
        ntr = np.arange(60)

        ncpu = 4 #os.cpu_count()
        time_batch = np.array_split(times, ncpu)
        
        res = Parallel(n_jobs=-1)(delayed(Semblance._process)(
        time_batch[i], velocities, offsets, cmp, sample_rate, n_samples, 
        l1, l2, ntr) for i in range(ncpu))

        return np.concatenate(res)

    @staticmethod
    def _process(time_batch : np.ndarray, velocities : np.ndarray, offsets : np.ndarray, cmp : np.ndarray, sample_rate :float, n_samples : int, 
                l1 : np.ndarray, l2 : np.ndarray, ntr : np.ndarray):
        """
        Input :
            time_batch : np.ndarray
                         a batch of time which we get by splitting the times array into batches
                         
            velocities : np.ndarray
                         Velocity array containg the velocity
            
            offsets : np.ndarray
                      Contain cdp_offset's corresponding to the unique cdp no.
            
            cmp : np.ndarray
                  Contain cdp gather for a unique cdp.
                     
            sample_rate : float
                          sampling interval
                          
            n_samples : int
                        no. of samples in a trace
                        
            l1 : np.ndarray
                 array containing upper window range
                 
            l2 : np.ndarray
                 array containing lower window range
            
            ntr : np.ndarray
                  array containing trace no's. in a cdp gather
            
        Returns:
            semblance_batch : np.ndarray
                              Contain semblance_batch corresponding to the cdp gather and time_batch.
        """
        semb = np.zeros((len(time_batch), len(velocities)))
        for n, time in enumerate(time_batch):
            for q, velocity in enumerate(velocities):

                trajectory = Semblance.reflection_time(time, offsets, velocity)

                amp = Semblance.interpolate_amplitude(cmp, trajectory, sample_rate)

                l = np.round(trajectory / sample_rate).astype('i')

                win1 = (l[:, None] + l1).T.astype('i')
                win2 = (l[:, None] + l2).T.astype('i')

                win11 = np.logical_and(win1 >= 0, win1 < n_samples)
                win22 = np.logical_and(win2 >= 0, win2 < n_samples)

                amp1 = np.zeros(win11.shape)

                amp2 = np.zeros(win22.shape)

                for i, w in enumerate(win11):
                        amp1[i, :sum(w)] = cmp[win1[i, w], ntr[w]]
                        
                for i, w in enumerate(win22):
                        amp2[i, :sum(w)] = cmp[win2[i, w], ntr[w]]
                
                num = sum(sum(amp1, axis=1)**2) + sum(amp)**2 + sum(sum(amp2, axis=1)**2)
                den = sum(sum(amp1**2, axis=1)*sum(win11, axis=1)) + sum(amp**2)*len(amp) + sum(sum(amp2**2, axis=1)*sum(win22, axis=1))
                
                semb[n,q] = num/(den) # + 1e-10)
                
        return semb

    @staticmethod
    def semblance_plotter(semblance : np.ndarray, cdp_no : np.integer, window : int):
        """
        Input : 
            semblance : np.ndarray
                        Contain semblance corresponding to the cdp gather.
            
            cdp_no : int
                     unique cdp no. corresponding to which semblance is calculated
            
            window : int
                     sample window or time window length over which the semblance is calculated

        Returns:
            plot : plt
                     Plot of a semblance (velocity panel) for a cdp gather
        """

        fig = plt.figure(figsize=(5, 7))

        plt.title(f'Semblance for cdp={cdp_no}, window={window}', fontsize=13)
        plt.xlabel('Velocity (m/sec) ',fontsize=15)
        plt.ylabel('Time (sec) ',fontsize=15)
        aspect = 'auto'
        clip=0.9
        img = plt.imshow(semblance, extent=[1200, 3600, 5.996, 0],
          aspect=aspect, cmap='jet', vmin=0, vmax=clip*np.max(semblance))
        plt.colorbar(img, label="semblance",shrink=0.5, ax=plt.gca())
  
        plt.show()
        
    @staticmethod
    def threshold_semblance(semblance_matrix : np.ndarray, threshold_value : float, cdp_no : int) -> np.ndarray :
        """
        Input : 
            semblance_matrix : np.ndarray
                        Contain semblance corresponding to the cdp gather.
                        
            threshold_value : float
                              value upto which you want to threshold the semblance values. (semb_value <= threshold_value will be replaced by zero)
            
            cdp_no : int
                     unique cdp no. corresponding to which semblance is calculated

        Returns:
            plot : plt
                   Plot of a threshold semblance for a cdp gather
                     
            threshold_semblance : np.ndarray
                                  array of the threshold semblance 
        """
        vth = np.where(semblance_matrix > threshold_value, semblance_matrix, 0)
        plt.figure(figsize=(5,7))
        plt.title(f'Threshold={threshold_value} Semblance for cdp={cdp_no}', fontsize=13)
        plt.xlabel('Velocity (m/sec) ',fontsize=15)
        plt.ylabel('Time (sec) ',fontsize=15)
        img = plt.imshow(vth, extent=[1200, 3600, 5.996, 0], cmap='jet', aspect='auto', vmin=0, vmax=0.9*vth.max())
        plt.colorbar(img, label="threshold_semb",shrink=0.5, ax=plt.gca())
        plt.show()
        return vth
        
    @staticmethod
    def velocity_model(threshold_semblance_matrix : np.ndarray, threshold_value : float, cdp_no : int) -> np.ndarray :
        """
        Input : 
            threshold_semblance_matrix : np.ndarray
                        Contain threshold semblance corresponding to a threshold_value.
                        
            threshold_value : float
                              value upto which you want to threshold the semblance values. (semb_value <= threshold_value will be replaced by zero)
            
            cdp_no : int
                     unique cdp no. corresponding to which semblance is calculated

        Returns:
            plot : plt
                     Plot of a velocity model
            velocity_model : np.ndarray
                            array velocity model corresponding to the threshold semblance 
        """
        v = np.exp(2*threshold_semblance_matrix)
        plt.figure(figsize=(5,7))
        plt.title(f'Thre = {threshold_value}, Velocity model for cdp={cdp_no}', fontsize=13)
        #plt.xlabel('Velocity (m/sec) ',fontsize=15)
        #plt.ylabel('Time (sec) ',fontsize=15)
        img = plt.imshow(v, extent=[1200, 3600, 5.996, 0], cmap='jet', aspect='auto', vmin=0, vmax=0.9*v.max())
        plt.colorbar(img, label="velocity",shrink=0.5, ax=plt.gca())
        plt.show() 
        return v.reshape(v.shape[0], v.shape[1], 1)
    
    @staticmethod
    def eikonal_solver(velocity_model : np.ndarray, source_index : tuple, receiver_coordinate : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] :
        """
        Input : 
            velocity_model : np.ndarray
                            array velocity model corresponding to the threshold semblance 
                        
            source_index : tuple
                           a tuple containing the source index (ix, iy, iz)
                           
            receiver_coordinate : np.ndarray
                                  an array containing the receiver coordinates np.array([xc, yc, zc], dtype=float)

        Returns:
            travel_time : np.ndarray
                        array containing the travel time values for the velocity model for the source at given location
                        
            ray_time : np.ndarray
                       array containing the time values for the ray trace from source to receiver
            
            ray_velocity : np.ndarray
                       array containing the velocity values for the ray trace from source to receiver
        """
        
        velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
        velocity.min_coords = 0, 0, 0
        velocity.node_intervals = 0.004, 0.01, 1
        velocity.npts = velocity_model.shape
        velocity.values = velocity_model
        
        solver_ek = pykonal.EikonalSolver(coord_sys="cartesian")
        solver_ek.vv.min_coords = velocity.min_coords
        solver_ek.vv.node_intervals = velocity.node_intervals
        solver_ek.vv.npts = velocity.npts
        solver_ek.vv.values = velocity.values

        src_idx = source_index #0, 20, 0
        solver_ek.tt.values[src_idx] = 0
        solver_ek.unknown[src_idx] = False
        solver_ek.trial.push(*src_idx)
        solver_ek.solve()
        
        tt_value = solver_ek.tt.values
        
        ray = solver_ek.tt.trace_ray(receiver_coordinate)
        r1 = ray[:, 0]
        r2 = (ray[:, 1]/0.01) *10 + 1200
        
        ray_times = np.arange(0,6,0.004)
        ray_velocity = np.interp(ray_times, r1, r2)
        
        return tt_value, ray_times, ray_velocity
    
    @staticmethod
    def travel_time_plotter(velocity_model : np.ndarray, threshold_value : float, TT_values : np.ndarray, source_index : tuple, cdp_no : int):
        """
        Input : 
            velocity_model : np.ndarray
                            array velocity model corresponding to the threshold semblance 
            
            threshold_value : float
                              value upto which you want to threshold the semblance values. (semb_value <= threshold_value will be replaced by zero)
            
            TT_values : np.ndarray
                        array containing the travel time values for the velocity model for the source at given location
                        
            source_index : tuple
                           a tuple containing the source index (ix, iy, iz)
            
            cdp_no : int
                     unique cdp no. corresponding to which semblance is calculated

        Returns:
            plot : plt
                     Plot of travel time and travel time contours.
        """
        
        plt.figure(figsize=(15,8))
        
        ax1 = plt.subplot(121)
        ax1.set_title(f'Thre = {threshold_value}, Travel time for cdp={cdp_no}', fontsize=13)
        img = ax1.imshow(TT_values, cmap='jet', aspect='auto', vmin=0, vmax=0.9*TT_values.max())
        ax1.scatter(source_index[1], source_index[0], marker='*', c='r', s=300, alpha=1)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        plt.colorbar(img, label="travel_time",shrink=0.5, ax=ax1)

        ax2 = plt.subplot(122)
        ax2.set_title(f'Thre = {threshold_value}, TT contours for cdp={cdp_no}', fontsize=13)
        img = ax2.imshow(velocity_model, extent=[1200, 3600, 5.996, 0], cmap='jet', aspect='auto', vmin=0, vmax=0.9*velocity_model.max())
        ax2.contour(np.arange(1200, 3610, 10), np.arange(0,6,0.004), TT_values[:, :, 0], colors="k",
                    levels=np.arange(0, TT_values.max(), 0.50), linewidths=0.5, linestyles="--")
        ax2.scatter(source_index[1]*10 + 1200, source_index[0], marker='*', c='r', s=300, alpha=1)
        plt.colorbar(img, label="velocity",shrink=0.5, ax=ax2)
        plt.show()
        
    @staticmethod
    def ray_trace_plotter(velocity_model : np.ndarray, semblance_matrix : np.ndarray, threshold_value : float, TT_values : np.ndarray, ray_times : np.ndarray, ray_velocity : np.ndarray, source_index : tuple, receiver_coordinate : np.ndarray, cdp_no : int, window : int):
        """
        Input : 
            velocity_model : np.ndarray
                            array velocity model corresponding to the threshold semblance 
                            
            semblance_matrix : np.ndarray
                        Contain semblance corresponding to the cdp gather.
            
            threshold_value : float
                              value upto which you want to threshold the semblance values. (semb_value <= threshold_value will be replaced by zero)
            
            TT_values : np.ndarray
                        array containing the travel time values for the velocity model for the source at given location
            
            ray_times : np.ndarray
                       array containing the time values for the ray trace from source to receiver
            
            ray_velocity : np.ndarray
                       array containing the velocity values for the ray trace from source to receiver
                        
            source_index : tuple
                           a tuple containing the source index (ix, iy, iz)
            
            receiver_coordinate : np.ndarray
                                  an array containing the receiver coordinates np.array([xc, yc, zc], dtype=float)
            
            cdp_no : int
                     unique cdp no. corresponding to which semblance is calculated
                     
            window : int
                     sample window or time window length over which the semblance is calculated

        Returns:
            plot : plt
                     Plot of travel time contours and ray trace (velocity picks).
        """
        plt.figure(figsize=(15,8))

        ax1 = plt.subplot(121)
        ax1.set_title(f'Thre = {threshold_value}, TT contours for cdp={cdp_no}', fontsize=13)
        ax1.imshow(velocity_model, extent=[1200, 3600, 5.996, 0], cmap='jet', aspect='auto', vmin=0, vmax=0.9*velocity_model.max())
        ax1.contour(np.arange(1200, 3610, 10), np.arange(0,6,0.004), TT_values[:, :, 0], colors="k",
                    levels=np.arange(0, TT_values.max(), 0.50), linewidths=0.5, linestyles="--")
        ax1.plot(source_index[1]*10 + 1200, source_index[0], marker='*', linestyle='None', color='r', markersize=20)
        ax1.plot((receiver_coordinate[1]/0.01)*10 + 1200, receiver_coordinate[0], marker='v', linestyle='None', color='b', markersize=10)
        ax1.plot(ray_velocity, ray_times, color="k", linewidth=1)

        ax2 = plt.subplot(122)
        ax2.set_title(f'Semblance for cdp={cdp_no}, window={window}', fontsize=13)
        ax2.set_xlabel('Velocity (m/sec) ',fontsize=15)
        ax2.set_ylabel('Time (sec) ',fontsize=15)
        img = ax2.imshow(semblance_matrix, extent=[1200, 3600, 5.996, 0], cmap='jet', aspect='auto', vmin=0, vmax=0.9*semblance_matrix.max())
        ax2.plot(ray_velocity, ray_times, color="k", linewidth=1)
        plt.colorbar(img, label="semblance",shrink=0.5, ax=ax2)

        plt.show()
        
    @staticmethod
    def differential_velocity_plotter(semblance_matrix : np.ndarray, velocity : np.ndarray, times : np.ndarray, list_of_percent : list, cdp_no : int, window : int):
        """
        Input :                 
            semblance_matrix : np.ndarray
                               Contain semblance corresponding to the cdp gather.
            
            velocity : np.ndarray
                       array containing the velocity values for the ray trace from source to receiver
            
            times : np.ndarray
                    array containing the time values for the ray trace from source to receiver
            
            list_of_percent : list
                              list of percentage value by which you want to change the velocity picks values i.e. [-3, 3, 6]
            
            cdp_no : int
                     unique cdp no. corresponding to which semblance is calculated
                     
            window : int
                     sample window or time window length over which the semblance is calculated

        Returns:
            plot : plt
                   Plot of ray trace (velocity picks) for different percentage change on semblance panel.
        """
        plt.figure(figsize=(6,8))
        
        plt.title(f'Semblance for cdp={cdp_no}, window={window}', fontsize=13)
        plt.xlabel('Velocity (m/sec) ',fontsize=15)
        plt.ylabel('Time (sec) ',fontsize=15)
        img = plt.imshow(semblance_matrix, extent=[1200, 3600, 5.996, 0], cmap='jet', aspect='auto', vmin=0, vmax=0.9*semblance_matrix.max())
        plt.plot(velocity, times, color="k", linewidth=1, label=f"V")
        plt.plot(velocity + velocity*(list_of_percent[0]/100), times, color="m", linewidth=1, label=f"V -- {-list_of_percent[0]}%V")
        plt.plot(velocity +  velocity*(list_of_percent[1]/100), times, color="r", linewidth=1, label=f"V + {list_of_percent[1]}%V")
        plt.plot(velocity + velocity*(list_of_percent[2]/100), times, color="g", linewidth=1, label=f"V + {list_of_percent[2]}%V")
        plt.legend()
        plt.colorbar(img, label="semblance",shrink=0.5, ax=plt.gca())
        plt.show()
    
    @staticmethod
    def nmo_correction(cmp : np.ndarray, times : np.ndarray, cmp_offsets : np.ndarray, velocities : np.ndarray, stretch_mute=None) -> np.ndarray :
        """
        Input : 
            cmp : np.ndarray
                  Contain cdp gather for a unique cdp.
                  
            times : np.ndarray
                    array containing the time values for the ray trace from source to receiver
                    
            cmp_offsets : np.ndarray
                      Contain cdp_offset's corresponding to the unique cdp no.
                          
            velocities : np.ndarray
                       array containing the velocity values for the ray trace from source to receiver
            
            stretch_mute : default None, float
                           value above which you want to apply the mute
            
        Returns:
            nmo_correct : np.ndarray
                          Contain nmo corrected gather corresponding to the cdp gather.
        """
        nmo = np.zeros_like(cmp)
        for j, x in enumerate(cmp_offsets):
            t = [Semblance.reflection_time(t0, x, velocities[i]) for i, t0 in enumerate(times)]
            interpolator = CubicSpline(times, cmp[:, j], extrapolate=False)
            amps = np.nan_to_num(interpolator(t), copy=False)
            nmo[:, j] = amps
            if stretch_mute is not None:
                nmo[np.abs((times-t)/(times+1e-10)) > stretch_mute, j] = 0
        return nmo
    
    @staticmethod
    def nmo_correct_plotter(cmp : np.ndarray, nmo_correct : np.ndarray, nmo_mute : np.ndarray, cdp_no : int):
        """
        Input : 
            cmp : np.ndarray
                  Contain cdp gather for a unique cdp.
                  
            nmo_correct : np.ndarray
                          Contain nmo corrected gather corresponding to the cdp gather.
                              
            nmo_mute : np.ndarray
                       Contain nmo strech mute gather corresponding to the nmo corrected gather.
            
            cdp_no : int
                     unique cdp no. corresponding to which semblance is calculated
            
        Returns:
            plot : plt
                   Plot of cmp and corresponding nmo_correct and nmo_corrected_stretch_mute.
        """
        plt.figure(figsize=(15,8))

        ax0 = plt.subplot(131)
        ax0.set_title(f'cdp = {cdp_no}', fontsize=13)
        ax0.set_xlabel('traces ',fontsize=15)
        ax0.set_ylabel('Time (sec) ',fontsize=15)
        vmin, vmax = np.percentile(cmp, [15, 85])
        ax0.imshow(cmp, cmap='Greys', extent=[0, 59, 5.996, 0], aspect='auto', vmin=vmin, vmax=vmax)

        ax1 = plt.subplot(132)
        ax1.set_title(f'nmo corrected, cdp = {cdp_no}', fontsize=13)
        ax1.set_xlabel('traces ',fontsize=15)
        vmin, vmax = np.percentile(nmo_correct, [15, 85])
        ax1.imshow(nmo_correct, cmap='Greys', extent=[0, 59, 5.996, 0], aspect='auto', vmin=vmin, vmax=vmax)

        ax2 = plt.subplot(133)
        ax2.set_title(f'stretch mute, cdp = {cdp_no}', fontsize=13)
        ax2.set_xlabel('traces ',fontsize=15)
        vmin, vmax = np.percentile(nmo_mute, [15, 85])
        ax2.imshow(nmo_mute, cmap='Greys', extent=[0, 59, 5.996, 0], aspect='auto', vmin=vmin, vmax=vmax)
        plt.show()
    
    
    @staticmethod
    def nmo_correct_diff_velocity_plotter(nmo_velocit0 : np.ndarray, nmo_velocit1 : np.ndarray, nmo_velocit2 : np.ndarray, nmo_velocit3 : np.ndarray, list_of_percent : list):
        """
        Input : 
            nmo_velocit0 : np.ndarray
                           Contain nmo strech mute gather corresponding to the nmo corrected gather by using the actual picked velocity.
            
            nmo_velocit1 : np.ndarray
                           Contain nmo strech mute gather corresponding to the nmo corrected gather by using the list_of_percent[0] times of actual velocity + actual velocity.
 
            nmo_velocit2 : np.ndarray
                           Contain nmo strech mute gather corresponding to the nmo corrected gather by using the list_of_percent[1] times of actual velocity + actual velocity.

            nmo_velocit3 : np.ndarray
                           Contain nmo strech mute gather corresponding to the nmo corrected gather by using the list_of_percent[2] times of actual velocity + actual velocity.

            list_of_percent : list
                              list of percentage value by which you want to vhange the velocity picks values i.e. [-3, 3, 6]
            
        Returns:
            plot : plt
                   Plot of nmo_corrected_stretch_mute for different %change in actual picked velocity.
        """
        plt.figure(figsize=(15,8))

        ax0 = plt.subplot(141)
        ax0.set_title(f'stretch mute0, for v= V', fontsize=13)
        ax0.set_ylabel('Time (sec) ',fontsize=15)
        ax0.set_xlabel('traces ',fontsize=15)
        vmin, vmax = np.percentile(nmo_velocit0, [15, 85])
        ax0.imshow(nmo_velocit0, cmap='Greys', extent=[0, 59, 5.996, 0], aspect='auto', vmin=vmin, vmax=vmax)

        ax1 = plt.subplot(142)
        ax1.set_title(f'stretch mute1, for v= V -- {-list_of_percent[0]}%V', fontsize=13)
        ax1.set_xlabel('traces ',fontsize=15)
        vmin, vmax = np.percentile(nmo_velocit1, [15, 85])
        ax1.imshow(nmo_velocit1, cmap='Greys', extent=[0, 59, 5.996, 0], aspect='auto', vmin=vmin, vmax=vmax)

        ax2 = plt.subplot(143)
        ax2.set_title(f'stretch mute2, for v= V + {list_of_percent[1]}%V', fontsize=13)
        ax2.set_xlabel('traces ',fontsize=15)
        vmin, vmax = np.percentile(nmo_velocit2, [15, 85])
        ax2.imshow(nmo_velocit2, cmap='Greys', extent=[0, 59, 5.996, 0], aspect='auto', vmin=vmin, vmax=vmax)
        
        ax3 = plt.subplot(144)
        ax3.set_title(f'stretch mute3, for v= V + {list_of_percent[2]}%V', fontsize=13)
        ax3.set_xlabel('traces ',fontsize=15)
        vmin, vmax = np.percentile(nmo_velocit3, [15, 85])
        ax3.imshow(nmo_velocit3, cmap='Greys', extent=[0, 59, 5.996, 0], aspect='auto', vmin=vmin, vmax=vmax)
        plt.show()