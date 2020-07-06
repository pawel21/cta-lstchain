"""
Functions to handle custom containers for the mono reconstruction of LST1
"""

import astropy.units as u
from astropy.units import Quantity
import numpy as np
from ctapipe.core import Container, Field
from ctapipe.image import timing_parameters as time
from ctapipe.image import leakage, concentration
from ctapipe.image.morphology import number_of_islands

from ..reco import utils
from numpy import nan

__all__ = [
    'DL1ParametersContainer',
    'DispContainer',
    'MetaData',
    'ThrownEventsHistogram',
    'DL1MonitoringEventIndexContainer',
    'LSTEventType'
]


class DL1ParametersContainer(Container):
    """
    TODO: maybe fields could be inherited from ctapipe containers definition
        For now I have not found an elegant way to do so
    """
    intensity = Field(None, 'total intensity (size)')
    log_intensity = Field(None, 'log of total intensity (size)')

    x = Field(None, 'centroid x coordinate', unit=u.m)
    y = Field(None, 'centroid x coordinate', unit=u.m)
    r = Field(None, 'radial coordinate of centroid', unit=u.m)
    phi = Field(None, 'polar coordinate of centroid', unit=u.rad)
    length = Field(None, 'RMS spread along the major-axis', unit=u.m)
    width = Field(None, 'RMS spread along the minor-axis', unit=u.m)
    psi = Field(None, 'rotation angle of ellipse', unit=u.rad)
    skewness = Field(None, 'measure of the asymmetry')
    kurtosis = Field(None, 'measure of the tailedness')
    disp_norm = Field(None, 'disp_norm [m]', unit=u.m)
    disp_dx = Field(None, 'disp_dx [m]', unit=u.m)
    disp_dy = Field(None, 'disp_dy [m]', unit=u.m)
    disp_angle = Field(None, 'disp_angle [rad]', unit=u.rad)
    disp_sign = Field(None, 'disp_sign')
    disp_miss = Field(None, 'disp_miss [m]', unit=u.m)
    src_x = Field(None, 'source x coordinate in camera frame', unit=u.m)
    src_y = Field(None, 'source y coordinate in camera frame', unit=u.m)
    time_gradient = Field(None, 'Time gradient in the camera')
    intercept = Field(None, 'Intercept')
    leakage1_intensity = Field(None, 'Fraction of intensity in outermost pixels')
    leakage2_intensity = Field(None, 'Fraction of intensity in two outermost rings of pixels')
    leakage1_pixel = Field(None, 'Fraction of signal pixels that are border pixels')
    leakage2_pixel = Field(None, 'Fraction of signal pixels that are in the two outermost rings of pixels')
    n_pixels = Field(None, 'Number of pixels after cleaning')
    concentration_cog = Field(None, 'Fraction of intensity in three pixels closest to the cog')
    concentration_core = Field(None, 'Fraction of intensity inside hillas ellipse')
    concentration_pixel = Field(None, 'Fraction of intensity in brightest pixel')
    n_islands = Field(None, 'Number of Islands')
    alt_tel = Field(None, 'Telescope altitude pointing', unit=u.rad)
    az_tel = Field(None, 'Telescope azimuth pointing', unit=u.rad)

    obs_id = Field(None, 'Observation ID')
    event_id = Field(None, 'Event ID')
    calibration_id = Field(None, 'ID of the employed calibration event')
    gps_time = Field(None, 'GPS time event trigger')
    dragon_time = Field(None, 'Dragon time event trigger')
    ucts_time = Field(None, 'UCTS time event trigger')
    tib_time = Field(None, 'TIB time event trigger')

    mc_energy = Field(None, 'Simulated Energy', unit=u.TeV)
    log_mc_energy = Field(None, 'log of simulated energy/TeV')
    mc_alt = Field(None, 'Simulated altitude', unit=u.rad)
    mc_az = Field(None, 'Simulated azimuth', unit=u.rad)
    mc_core_x = Field(None, 'Simulated impact point x position', unit=u.m)
    mc_core_y = Field(None, 'Simulated impact point y position', unit=u.m)
    mc_h_first_int = Field(None, 'Simulated first interaction height', unit=u.m)
    mc_type = Field(None, 'Simulated particle type')
    mc_az_tel = Field(None, 'Telescope MC azimuth pointing', unit=u.rad)
    mc_alt_tel = Field(None, 'Telescope MC altitude pointing', unit=u.rad)
    mc_x_max = Field(None, "MC Xmax value", unit=u.g / (u.cm ** 2))
    mc_core_distance = Field(None, "Distance from the impact point to the telescope", unit=u.m)
    mc_shower_primary_id = Field(None, "MC shower primary ID 0 (gamma), 1(e-),"
                                    "2(mu-), 100*A+Z for nucleons and nuclei,"
                                    "negative for antimatter.")

    hadroness = Field(None, "Hadroness")
    wl = Field(None, "width/length")

    tel_id = Field(None, "Telescope Id")
    tel_pos_x = Field(None, "Telescope x position in the ground")
    tel_pos_y = Field(None, "Telescope y position in the ground")
    tel_pos_z = Field(None, "Telescope z position in the ground")

    trigger_type = Field(None, "trigger type")
    ucts_trigger_type = Field(None, "UCTS trigger type")
    trigger_time = Field(None, "trigger time")

    # info not available in data
    #num_trig_pix = Field(None, "Number of trigger groups (sectors) listed")
    #trig_pix_id = Field(None, "pixels involved in the camera trigger")

    def fill_hillas(self, hillas):
        """
        fill Hillas parameters

        hillas: HillasParametersContainer
        # TODO : parameters should not be simply copied but inherited
        (e.g. conserving unit definition)
        """
        for key in hillas.keys():
            self[key] = hillas[key]

    def fill_mc(self, event):
        """
        fill from mc
        """
        try:
            self.mc_energy = event.mc.energy
            self.mc_alt = event.mc.alt
            self.mc_az = event.mc.az
            self.mc_core_x = event.mc.core_x
            self.mc_core_y = event.mc.core_y
            self.mc_h_first_int = event.mc.h_first_int
            # mcType = event.mc. # TODO: find type in event
            self.mc_x_max = event.mc.x_max
            self.mc_alt_tel = event.mcheader.run_array_direction[1]
            self.mc_az_tel = event.mcheader.run_array_direction[0]
        except IndexError:
            print("mc information not filled")

    def fill_event_info(self, event):
        self.gps_time = event.trig.gps_time
        self.obs_id = event.r0.obs_id
        self.event_id = event.r0.event_id

    def get_features(self, features_names):
        return np.array([
            self[k].value
            if isinstance(self[k], Quantity)
            else self[k]
            for k in features_names
        ])

    def set_mc_core_distance(self, event, telescope_id):
        tel_pos = event.inst.subarray.positions[telescope_id]
        distance = np.sqrt(
            (event.mc.core_x - tel_pos[0]) ** 2 +
            (event.mc.core_y - tel_pos[1]) ** 2
        )
        self.mc_core_distance = distance

    def set_disp(self, source_pos, hillas):
        disp = utils.disp_parameters(hillas, source_pos[0], source_pos[1])
        self.disp_norm = disp.norm
        self.disp_dx = disp.dx
        self.disp_dy = disp.dy
        self.disp_angle = disp.angle
        self.disp_sign = disp.sign
        self.disp_miss = disp.miss

    def set_timing_features(self, geom, image, pulse_time, hillas):
        try:    # if np.polyfit fails (e.g. len(image) < deg + 1)
            timepars = time.timing_parameters(geom, image, pulse_time, hillas)
            self.time_gradient = timepars.slope.value
            self.intercept = timepars.intercept
        except ValueError:
            self.time_gradient = np.nan
            self.intercept = np.nan

    def set_leakage(self, geom, image, clean):
        leakage_c = leakage(geom, image, clean)
        self.leakage1_intensity = leakage_c.leakage1_intensity
        self.leakage2_intensity = leakage_c.leakage2_intensity
        self.leakage1_pixel = leakage_c.leakage1_pixel
        self.leakage2_pixel = leakage_c.leakage2_pixel

    def set_concentration(self, geom, image, hillas_parameters):
        conc = concentration(geom, image, hillas_parameters)
        self.concentration_cog = conc.cog
        self.concentration_core = conc.core
        self.concentration_pixel = conc.pixel

    def set_n_islands(self, geom, clean):
        n_islands, islands_mask = number_of_islands(geom, clean)
        self.n_islands = n_islands

    def set_telescope_info(self, event, telescope_id):
        self.tel_id = telescope_id
        tel_pos = event.inst.subarray.positions[telescope_id]
        self.tel_pos_x = tel_pos[0]
        self.tel_pos_y = tel_pos[1]
        self.tel_pos_z = tel_pos[2]

    def set_source_camera_position(self, event, telescope_id):
        tel = event.inst.subarray.tel[telescope_id]
        source_pos = utils.get_event_pos_in_camera(event, tel)
        self.src_x = source_pos[0]
        self.src_y = source_pos[1]

    def set_mc_type(self, event):
        self.mc_type = event.mc.shower_primary_id


class DispContainer(Container):
    """
    Disp vector container
    """
    dx = Field(nan, 'x coordinate of the disp_norm vector')
    dy = Field(nan, 'y coordinate of the disp_norm vector')

    angle = Field(nan, 'Angle between the X axis and the disp_norm vector')
    norm = Field(nan, 'Norm of the disp_norm vector')
    sign = Field(nan, 'Sign of the disp_norm')
    miss = Field(nan, 'miss parameter norm')


class ExtraMCInfo(Container):
    obs_id = Field(0, "MC Run Identifier")

class ExtraImageInfo(Container):
    """ attach the tel_id """
    tel_id = Field(0, "Telescope ID")
    selected_gain_channel = Field(None, "Selected gain channel")


class ThrownEventsHistogram(Container):
    """ 2D histogram from SimTel files """
    obs_id = Field(-1, 'MC run ID')
    hist_id = Field(-1, 'Histogram ID')
    num_entries = Field(-1, 'Number of entries in the histogram')
    bins_energy = Field(None, 'array of energy bin lower edges, as in np.histogram')
    bins_core_dist = Field(None, 'array of core-distance bin lower edges, as in np.histogram')
    histogram = Field(None, "array of histogram entries, size (n_bins_x, n_bins_y)")

    def fill_from_simtel(self, hist):
        """ fill from a SimTel Histogram entry"""
        self.hist_id = hist['id']
        self.num_entries = hist['entries']
        xbins = np.linspace(hist['lower_x'], hist['upper_x'], hist['n_bins_x'] + 1)
        ybins = np.linspace(hist['lower_y'], hist['upper_y'], hist['n_bins_y'] + 1)
        self.bins_core_dist = xbins
        self.bins_energy = 10 ** ybins
        self.histogram = hist['data']
        self.meta['hist_title'] = hist['title']
        self.meta['x_label'] = 'Log10 E (TeV)'
        self.meta['y_label'] = '3D Core Distance (m)'


class MetaData(Container):
    """
    Some metadata
    """
    SOURCE_FILENAMES = Field([], "filename of the source file")
    LSTCHAIN_VERSION = Field(None, "version of lstchain")
    CTAPIPE_VERSION = Field(None, "version of ctapipe")
    CONTACT = Field(None, "Person or institution responsible for this data product")


class DL1MonitoringEventIndexContainer(Container):
    """
    Container with the calibration coefficients
    """
    tel_id = Field(1, 'Index of telescope')
    calibration_id = Field(-1, 'Index of calibration event for DL1 file')
    pedestal_id = Field(-1, 'Index of pedestal event for DL1 file')
    flatfield_id = Field(-1, 'Index of flat-field event for DL1 file')


class LSTEventType:
    """
    Class to recognize event type from trigger bits
    bit 0: Mono
    bit 1: stereo
    bit 2: Calibration
    bit 3: Single Phe
    bit 4: Softrig(from the UCTS)
    bit 5: Pedestal
    bit 6: slow control
    bit 7: busy
    """

    @staticmethod
    def is_mono(trigger_type):
        return trigger_type >> 0 & 1

    @staticmethod
    def is_stereo(trigger_type):
        return trigger_type >> 1 & 1

    @staticmethod
    def is_calibration(trigger_type):
        return trigger_type >> 2 & 1

    @staticmethod
    def is_single_pe(trigger_type):
        return trigger_type >> 3 & 1

    @staticmethod
    def is_soft_trig(trigger_type):
        return trigger_type >> 4 & 1

    @staticmethod
    def is_pedestal(trigger_type):
        return trigger_type >> 5 & 1

    @staticmethod
    def is_slow_control(trigger_type):
        return trigger_type >> 6 & 1

    @staticmethod
    def is_busy(trigger_type):
        return trigger_type >> 7 & 1

    @staticmethod
    def is_unknown(trigger_type):
        return trigger_type == -1
