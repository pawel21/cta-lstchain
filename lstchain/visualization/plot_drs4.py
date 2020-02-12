
from matplotlib import pyplot as plt
from traitlets.config.loader import Config
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from ctapipe.io import event_source
from lstchain.calib.camera.r0 import LSTR0Corrections, NullR0Calibrator
from ctapipe.instrument import  CameraGeometry
from ctapipe.image.extractor import LocalPeakWindowSum

from ctapipe.calib.camera.pedestals import PedestalIntegrator
from ctapipe.visualization import CameraDisplay

from lstchain.calib.camera.pulse_time_correction import PulseTimeCorrection
from lstchain.calib.camera.drs4 import PulseCalibCheck


__all__ = ['plot_pedestals',
           ]

channel = ['HG', 'LG']


def plot_pedestals(data_file, pedestal_file, run=0 , plot_file="none", tel_id=1, offset_value=400):
    """
     plot pedestal quantities quantities

     Parameters
     ----------
     data_file:   pedestal run

     pedestal_file:   file with drs4 corrections

     run: run number of data to be corrected

     plot_file:  name of output pdf file

     tel_id: id of the telescope

     offset_value: baseline off_set
     """

    # plot open pdf
    if plot_file != "none":
        pp = PdfPages(plot_file)

    plt.rc('font', size=15)

    # r0 calibrator
    r0_calib = LSTR0Corrections(pedestal_path=pedestal_file, offset=offset_value,
                                r1_sample_start=2, r1_sample_end=38, tel_id=tel_id )

    # event_reader
    reader = event_source(data_file, max_events=1000)
    t = np.linspace(2, 37, 36)

    # configuration for the charge integrator
    charge_config = Config({
        "FixedWindowSum": {
            "window_start": 12,
            "window_width": 12,
        }

    })
    # declare the pedestal component
    pedestal = PedestalIntegrator(tel_id=tel_id,
                                  sample_size=1000,
                                  sample_duration=1000000,
                                  charge_median_cut_outliers=[-10, 10],
                                  charge_std_cut_outliers=[-10, 10],
                                  charge_product="FixedWindowSum",
                                  config=charge_config)

    for i, event in enumerate(reader):
        if tel_id != event.r0.tels_with_data[0]:
            raise Exception(f"Given wrong telescope id {tel_id}, files has id {event.r0.tels_with_data[0]}")

        # move from R0 to R1
        r0_calib.calibrate(event)

        ok = pedestal.calculate_pedestals(event)
        if ok:
            ped_data = event.mon.tel[tel_id].pedestal
            break

    camera = CameraGeometry.from_name("LSTCam", 2)

    # plot open pdf
    if plot_file != "none":
        pp = PdfPages(plot_file)

    plt.rc('font', size=15)

    ### first figure
    fig = plt.figure(1, figsize=(12, 24))
    plt.tight_layout()
    n_samples = charge_config["FixedWindowSum"]['window_width']
    fig.suptitle(f"Run {run}, integration on {n_samples} samples", fontsize=25)
    pad = 420

    image = ped_data.charge_median
    mask = ped_data.charge_median_outliers
    for chan in (np.arange(2)):
        pad += 1
        plt.subplot(pad)
        plt.tight_layout()
        disp = CameraDisplay(camera)
        mymin = np.median(image[chan]) - 2 * np.std(image[chan])
        mymax = np.median(image[chan]) + 2 * np.std(image[chan])
        disp.set_limits_minmax(mymin, mymax)
        disp.highlight_pixels(mask[chan], linewidth=2)
        disp.image = image[chan]
        disp.cmap = plt.cm.coolwarm
        # disp.axes.text(lposx, 0, f'{channel[chan]} pedestal [ADC]', rotation=90)
        plt.title(f'{channel[chan]} pedestal [ADC]')
        disp.add_colorbar()

    image = ped_data.charge_std
    mask = ped_data.charge_std_outliers
    for chan in (np.arange(2)):
        pad += 1
        plt.subplot(pad)
        plt.tight_layout()
        disp = CameraDisplay(camera)
        mymin = np.median(image[chan]) - 2 * np.std(image[chan])
        mymax = np.median(image[chan]) + 2 * np.std(image[chan])
        disp.set_limits_minmax(mymin, mymax)
        disp.highlight_pixels(mask[chan], linewidth=2)
        disp.image = image[chan]
        disp.cmap = plt.cm.coolwarm
        # disp.axes.text(lposx, 0, f'{channel[chan]} pedestal std [ADC]', rotation=90)
        plt.title(f'{channel[chan]} pedestal std [ADC]')
        disp.add_colorbar()

    ###  histograms
    for chan in np.arange(2):
        mean_ped = ped_data.charge_mean[chan]
        ped_std = ped_data.charge_std[chan]

        # select good pixels
        select = np.logical_not(mask[chan])

        #fig.suptitle(f"Run {run} channel: {channel[chan]}", fontsize=25)
        pad += 1
        # pedestal charge
        plt.subplot(pad)
        plt.tight_layout()
        plt.ylabel('pixels')
        plt.xlabel(f'{channel[chan]} pedestal')
        median = np.median(mean_ped[select])
        rms = np.std(mean_ped[select])
        label = f"{channel[chan]} Median {median:3.2f}, std {rms:3.2f}"
        plt.hist(mean_ped[select], bins=50, label=label)
        plt.legend()
        pad += 1
        # pedestal std
        plt.subplot(pad)
        plt.ylabel('pixels')
        plt.xlabel(f'{channel[chan]} pedestal std')
        median = np.median(ped_std[select])
        rms = np.std(ped_std[select])
        label = f" Median {median:3.2f}, std {rms:3.2f}"
        plt.hist(ped_std[select], bins=50, label=label)
        plt.legend()

    plt.subplots_adjust(top=0.94)
    if plot_file != "none":

        pp.savefig()

    pix = 0
    pad = 420
    # plot corrected waveforms of first 8 events
    for i, ev in enumerate(reader):
        for chan in np.arange(2):

            if pad == 420:
                # new figure

                fig = plt.figure(ev.r0.event_id, figsize=(12, 24))
                fig.suptitle(f"Run {run}, pixel {pix}", fontsize=25)
                plt.tight_layout()
            pad += 1
            plt.subplot(pad)

            plt.subplots_adjust(top=0.92)
            label = f"event {ev.r0.event_id}, {channel[chan]}: R0"
            plt.step(t, ev.r0.tel[tel_id].waveform[chan, pix, 2:38], color="blue", label=label)

            r0_calib.subtract_pedestal(ev,tel_id)
            label = "+ pedestal substraction"
            plt.step(t, ev.r1.tel[tel_id].waveform[chan, pix, 2:38], color="red", alpha=0.5,  label=label)

            r0_calib.time_lapse_corr(ev,tel_id)
            r0_calib.interpolate_spikes(ev,tel_id)
            label = "+ dt corr + interp. spikes"
            plt.step(t, ev.r1.tel[tel_id].waveform[chan, pix, 2:38],  alpha=0.5, color="green",label=label)
            plt.plot([0, 40], [offset_value, offset_value], 'k--',  label="offset")
            plt.xlabel("time sample [ns]")
            plt.ylabel("counts [ADC]")
            plt.legend()
            plt.ylim([-50, 500])

        if plot_file != "none" and pad == 428:
            pad = 420
            plt.subplots_adjust(top=0.92)
            pp.savefig()

        if i == 8:
            break

    if plot_file != "none":
        pp.close()


def plot_check_r0(data_file, run=0, plot_file="none", tel_id=1):
    """
     plot pedestal quantities quantities

     Parameters
     ----------
     data_file:   pedestal run

     run: run number of data to be corrected

     plot_file:  name of output pdf file

     tel_id: id of the telescope

     """

    # plot open pdf
    if plot_file != "none":
        pp = PdfPages(plot_file)

    plt.rc('font', size=15)

    # r0 null calibrator
    r0_null = NullR0Calibrator(r1_sample_start=2, r1_sample_end=38, offset=0)

    # event_reader
    reader = event_source(data_file, max_events=1000)
    t = np.linspace(2, 37, 36)

    window_width = 12
    # configuration for the charge integrator
    charge_config = Config({
        "FixedWindowSum": {
            "window_start": 12,
            "window_width": window_width,
        }

    })
    # declare the pedestal component
    pedestal = PedestalIntegrator(tel_id=tel_id,
                                  sample_size=500,
                                  sample_duration=1000000,
                                  charge_median_cut_outliers=[-10, 10],
                                  charge_std_cut_outliers=[-10, 10],
                                  charge_product="FixedWindowSum",
                                  config=charge_config)

    for i, event in enumerate(reader):
        if tel_id != event.r0.tels_with_data[0]:
            raise Exception(f"Given wrong telescope id {tel_id}, files has id {event.r0.tels_with_data[0]}")

            # move from R0 to R1
        r0_null.calibrate(event)

        ok = pedestal.calculate_pedestals(event)
        if ok:
            print(i)
            ped_data = event.mon.tel[tel_id].pedestal
            break



    fig = plt.figure(1, figsize=(12, 12))
    plt.tight_layout()
    #plt.subplot(1)
    pad = 221
    for i in range(0, 2):
        pad = pad + i
        plt.subplot(pad)
        ped_cap = ped_data.charge_mean[i, :] / window_width
        plt.plot(ped_cap, 'bo-')
        plt.xlabel("Pixel Id")
        plt.ylabel("Mean signal [ADC]")

        if i == 0:
            plt.title("HG")
        if i == 1:
            plt.title("LG")



    camera = CameraGeometry.from_name("LSTCam", 2)

    print(ped_data.charge_mean[0, :].shape)
    for chan in range(0, 2):
        image = ped_data.charge_mean[chan, :]/12
        pad = pad + i
        plt.subplot(pad)
        plt.tight_layout()
        disp = CameraDisplay(camera)
        #mymin = np.median(image[chan]) - 2 * np.std(image[chan])
        #mymax = np.median(image[chan]) + 2 * np.std(image[chan])
        #disp.set_limits_minmax(mymin, mymax)
        #disp.highlight_pixels(mask[chan], linewidth=2)
        disp.image = image
        disp.cmap = plt.cm.coolwarm
        # disp.axes.text(lposx, 0, f'{channel[chan]} pedestal std [ADC]', rotation=90)
    #    plt.title(f'{channel[chan]} pedestal std [ADC]')
        disp.add_colorbar(label="Mean charge [ADC]")
        plt.title("")
        plt.xlabel("")
        plt.ylabel("")



    fig = plt.figure(2, figsize=(12, 12))
    plt.tight_layout()

    pad = 221

    for chan in range(0, 2):
        image = ped_data.charge_std[chan, :] / 12
        print(image)
        pad = pad + chan
        print(pad)
        plt.subplot(pad)
        plt.tight_layout()
        disp = CameraDisplay(camera)
            # mymin = np.median(image[chan]) - 2 * np.std(image[chan])
            # mymax = np.median(image[chan]) + 2 * np.std(image[chan])
            # disp.set_limits_minmax(mymin, mymax)
            # disp.highlight_pixels(mask[chan], linewidth=2)
        disp.image = image
        disp.cmap = plt.cm.coolwarm
            # disp.axes.text(lposx, 0, f'{channel[chan]} pedestal std [ADC]', rotation=90)
            #    plt.title(f'{channel[chan]} pedestal std [ADC]')
        disp.add_colorbar(label="RMS of charge")
        plt.title("")
        plt.xlabel("")
        plt.ylabel("")

        if i == 0:
            plt.title("HG")
        if i == 1:
            plt.title("LG")

    pad = pad + 1
    for chan in range(0, 2):
        pad = pad + chan
        print(pad)
        plt.subplot(pad)
        image = ped_data.charge_std[chan, :] / 12
        plt.hist(image)
        plt.xlabel("Std charge")
        plt.ylabel("N")

    plt.tight_layout()
    plt.show()


def plot_drs4_pulse_correction(data_file, pedestal_file=None, time_calib_file='none', run=0, plot_file="none", tel_id=1):
    """
     plot pedestal quantities quantities

     Parameters
     ----------
     data_file:   pedestal run

     run: run number of data to be corrected

     plot_file:  name of output pdf file

     tel_id: id of the telescope

     """

    reader = event_source(data_file, max_events=1000)

    pulse_corr = PulseTimeCorrection(calib_file_path=time_calib_file)

    config = Config({
        "LSTR0Corrections": {
            "pedestal_path": pedestal_file,  # if baseline correction was done in EVB
            "tel_id": 1,
            "r1_sample_start": 2,
            "r1_sample_end": 38
        }
    })

    lst_r0 = LSTR0Corrections(config=config)
    extractor = LocalPeakWindowSum(window_width=11, window_shift=4)

    p = PulseCalibCheck()
    p_corr = PulseCalibCheck()

    for i, ev in enumerate(reader):
        if ev.r0.event_id % 500 == 0:
            print(ev.r0.event_id)

        lst_r0.calibrate(ev)  # Cut in signal to avoid cosmic events
        if ev.r0.tel[1].trigger_type == 1 and np.mean(ev.r1.tel[1].waveform[:, :, :]) > 100:
            charge, pulse = extractor(ev.r1.tel[1].waveform[:, :, :])
            pulse_corr_array = pulse_corr.get_corr_pulse(ev, pulse)

            p.fill(pulse[:, :])
            p_corr.fill(pulse_corr_array[:, :])

    p.finish()
    p_corr.finish()

    geom = CameraGeometry.from_name("LSTCam", 2)

    # RMS plots

    for chan in (np.arange(2)):
        fig = plt.figure(figsize=(10, 10))
        pad = 220

        mymin = np.median(p.rms_time_array[chan]) - 2 * np.std(p.rms_time_array[chan])
        mymax = np.median(p.rms_time_array[chan]) + 2 * np.std(p.rms_time_array[chan])

        for P in [p, p_corr]:
            pad += 1
            plt.subplot(pad)
            image = P.rms_time_array
            disp = CameraDisplay(geom)
            disp.image = image[chan]
            disp.add_colorbar(label="RMS of mean arrival time [ns]")
            disp.cmap = 'gnuplot2'

            #mymin = np.median(image[chan]) - 2 * np.std(image[chan])
            #mymax = np.median(image[chan]) + 2 * np.std(image[chan])

            disp.set_limits_minmax(mymin, mymax)
            #disp.set_limits_minmax(0, 4)
            plt.xlabel("")
            plt.ylabel("")
            plt.title("")

            pad += 1
            plt.subplot(pad)
            plt.hist(image[chan, :], bins=50, histtype='step', lw=2, range=(mymin, mymax))
            plt.xlabel("RMS of mean arrival time [ns]")
            plt.ylabel("Number of pixels")
            plt.yscale('log')
            #plt.tight_layout()
            print(f"{channel[chan]} ", end="\t")
            print("Mean RMS = {}".format(np.mean(image[chan, :])))

        if chan == 0:
            plt.suptitle("HG", fontsize=25)
        if chan == 1:
            plt.suptitle("LG", fontsize=25)



    #plt.tight_layout()
    plt.show()

    # Mean plots
    for chan in (np.arange(2)):
        fig = plt.figure(figsize=(10, 10))
        pad = 220
        sigma = np.std(p.mean_time_array[chan])
        for P in [p, p_corr]:
            pad += 1
            plt.subplot(pad)
            image = P.mean_time_array
            disp = CameraDisplay(geom)
            disp.image = image[chan, :]
            disp.add_colorbar(label="Mean arrival time [ns]")
            disp.cmap = 'gnuplot2'

            mymin = np.median(image[chan]) - 2 * sigma
            mymax = np.median(image[chan]) + 2 * sigma
            disp.set_limits_minmax(mymin, mymax)

            plt.xlabel("")
            plt.ylabel("")
            plt.title("")

            pad += 1
            plt.subplot(pad)
            plt.hist(image[chan, :], bins=80, histtype='step', lw=2, range=(mymin, mymax))
            plt.xlabel("Mean arrival time [ns]")
            plt.ylabel("Number of pixels")

            print(f"{channel[chan]} ", end = "\t")
            print("RMS of mean arrival time  = {}".format(np.std(image[chan, :])))

        if chan == 0:
            plt.suptitle("HG", fontsize=25)
        if chan == 1:
            plt.suptitle("LG", fontsize=25)


    plt.show()
