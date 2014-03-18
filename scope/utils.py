# coding: utf-8

""" General utilities for SCOPE """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import numpy as np

__all__ = ["human_readable_digit", "find_spectral_overlap", "latexify",
    "unique_preserved_list"]


def latexify(labels, overwrite_common_labels=None):

    common_labels = {
        "teff": "$T_{eff}$ [K]",
        "feh": "[Fe/H]",
        "logg": "$\log{g}$",
        "alpha": "[$\alpha$/Fe]",
        "jitter": "$\delta$"
    }

    if overwrite_common_labels is not None:
        common_labels.update(overwrite_common_labels)

    latex_labels = []
    for label in labels:

        if label in common_labels.keys():
            latex_labels.append(common_labels[label])
        
        elif label.startswith("jitter."):
            aperture = label.split(".")[1]
            latex_labels.append("$\delta_{{{0}}}$".format(aperture))

        elif label.startswith("doppler_shift."):
            aperture = label.split(".")[1]
            latex_labels.append("$V_{{{0}}}$ [km/s]".format(aperture))

        elif label.startswith("smooth_model_flux."):
            aperture = label.split(".")[1]
            latex_labels.append("$\sigma_{{{0}}}$ [$\AA$]".format(aperture))

        elif label.startswith("normalise_observed."):
            aperture, coefficient = label.split(".")[1:]
            latex_labels.append("${0}_{1}$".format(aperture[0], coefficient[1:]))

        else:
            latex_labels.append(label)

    return latex_labels


def unique_preserved_list(original_list):
    seen = set()
    seen_add = seen.add
    return [x for x in original_list if x not in seen and not seen_add(x)]


def human_readable_digit(number):
    millnames = ["", "thousand", "million", "billion", "trillion"]
    millidx = max(0, min(len(millnames)-1,
                      int(np.floor(np.log10(abs(number))/3.0))))
    return "{digit:.1f} {multiple}".format(
        digit=number/10**(3*millidx),
        multiple=millnames[millidx])


def find_spectral_overlap(dispersion_maps, interval_resolution=1):
    """ Checks whether the dispersion maps overlap or not.

    Inputs
    ------
    dispersion_maps : list of list-types of length 2+
        The dispersion maps in the format [(wl_1, wl_2, ... wl_N), ..., (wl_start, wl_end)]
    
    interval_resolution : float, Angstroms
        The resolution at which to search for overlap. Any overlap less than the
        `interval_resolution` may not be detected.

    Returns
    -------
    None if no overlap is found, otherwise the wavelength near the overlap is returned.
    """

    all_min = np.min(map(np.min, dispersion_maps))
    all_max = np.max(map(np.max, dispersion_maps))

    interval_tree_disp = np.arange(all_min, 
        all_max + interval_resolution, interval_resolution)
    interval_tree_flux = np.zeros(len(interval_tree_disp))

    for dispersion_map in dispersion_maps:

        wlstart, wlend = np.min(dispersion_map), np.max(dispersion_map)
        idx = np.searchsorted(interval_tree_disp, 
            [wlstart, wlend + interval_resolution])
        interval_tree_flux[idx[0]:idx[1]] += 1

    # Any overlap?
    if np.max(interval_tree_flux) > 1:
        idx = np.where(interval_tree_flux > 1)[0]
        wavelength = interval_tree_disp[idx[0]]
        return wavelength
    
    else:
        return None

"""

def pre_cache(model, new_dispersions, kernels, smoothing_mode):
    Perform pre-caching: smooth the model fluxes and interpolate them onto a dispersion map
    which is similar to the observations.


    # TODO: get config to verify the pre-caching configuration options
    for aperture, settings in self.configuration['pre-cache'].iteritems():

        logging.info("Pre-caching '{aperture}' aperture with settings: {settings}"
            .format(aperture=aperture, settings=settings))

        # Load the dispersion map for this aperture
        original_dispersion_map = self.dispersion[aperture]

        # Load the new dispersion map that is requested
        if 'dispersion_filename' in settings:

            old_dispersion_filename = settings['dispersion_filename']
            logging.debug("Dispersion filename to interpolate onto is {filename}"
                .format(filename=old_dispersion_filename))
            cached_dispersion_map = load_model_data(old_dispersion_filename)

            # Over-sample the new dispersion map if necessary
            if 'oversample' in settings and settings['oversample']:
                
                logging.debug("Performing oversampling to minimise interpolation losses")

                num_pixels = len(cached_dispersion_map)
                cached_dispersion_map = np.linspace(
                    cached_dispersion_map[0],
                    cached_dispersion_map[-1],
                    num_pixels + num_pixels - 1)

            else:
                logging.debug("No oversampling performed")

        else:
            logging.debug("Using same dispersion map. No oversampling, just smoothing.")

            old_dispersion_filename = self.configuration['models']['dispersion_filenames'][aperture]
            cached_dispersion_map = original_dispersion_map
   
        if old_dispersion_filename.endswith('.cached'):
            # This has already been cached once before. We should warn about this.
            logging.warn("Dispersion filename '{filename}' looks like it may have been cached before."
                " Continuing to cache to '{cached_filename}' anyways."
                .format(filename=old_dispersion_filename, cached_filename=cached_dispersion_filename))

        cached_dispersion_filename = self.configuration['models']['dispersion_filenames'][aperture] + '.cached'

        # Go through all the filenames for that aperture
        if isinstance(self.flux_filenames, dict):
            # There is more than one aperture
            flux_filenames = self.flux_filenames[aperture]

        else:
            flux_filenames = self.flux_filenames


        num_flux_filenames = len(flux_filenames)
        for i, flux_filename in enumerate(flux_filenames, 1):

            logging.info("Working on model flux '{flux_filename}' ({i}/{num}).."
                .format(flux_filename=flux_filename, i=i, num=num_flux_filenames))

            model_flux = load_model_data(flux_filename)

            # For each one: Load the flux, convolve it, and interpolate to the new dispersion
            done_something = False

            # Perform any smoothing
            if 'kernel' in settings:
                kernel_fwhm = settings['kernel']
                logging.debug("Convolving with kernel {kernel_fwhm:.3f} Angstroms..".format(kernel_fwhm=kernel_fwhm))

                # Convert kernel (Angstroms) to pixel size
                kernel_sigma = kernel_fwhm / (2 * (2*np.log(2))**0.5)
                
                # The requested FWHM is in Angstroms, but the dispersion between each
                # pixel is likely less than an Angstrom, so we must calculate the true
                # smoothing value
                
                true_profile_sigma = kernel_sigma / np.mean(np.diff(original_dispersion_map))
                model_flux = ndimage.gaussian_filter1d(model_flux, true_profile_sigma)
                done_something = True

            # Perform any interpolation
            if not np.all(cached_dispersion_map == original_dispersion_map):

                logging.debug("Interpolating onto new dispersion map..")
                f = interpolate.interp1d(original_dispersion_map, model_flux, bounds_error=False)

                done_something = True
                model_flux = f(cached_dispersion_map)

            # Let's do a sanity check to ensure we've actually *done* something
            if not done_something:
                logging.warn("Model flux has not been changed by the pre-caching method. Check your configuration file.")

            # Remove non-finite values
            finite_indices = np.isfinite(model_flux)
            model_flux = model_flux[finite_indices]

            # Save the cached dispersion map
            if i == 0:
                logging.debug("Saving cached dispersion map to {cached_dispersion_filename}".format(cached_dispersion_filename=cached_dispersion_filename))
                fp = np.memmap(cached_dispersion_filename, dtype=np.float32, mode='w+', shape=cached_dispersion_map[finite_indices].shape)
                fp[:] = cached_dispersion_map[finite_indices]
                del fp

            # Save the new flux to file
            cached_flux_filename = flux_filename + '.cached'
            if flux_filename.endswith('.cached'):
                # Looks like it may have been cached before. We should warn about this.
                logging.warn("Model flux filename '{filename}' looks like it may have been cached before."
                    " Continuing to cache to '{cached_filename}' anyways."
                    .format(filename=flux_filename, cached_filename=cached_flux_filename))

            # Save cached flux to disk
            logging.debug("Saving cached model flux to {cached_flux_filename}".format(cached_flux_filename=cached_flux_filename))
            fp = np.memmap(cached_flux_filename, dtype=np.float32, mode='w+', shape=model_flux.shape)
            fp[:] = model_flux[:]
            del fp

        # Info.log that shit to recommend altering the yaml file to use cache files instead.
    logging.info("Caching complete. Model filenames for dispersion and flux have been amended to have a"
        " '.cached' extension. You must alter your configuration file to use the cached filenames.")

    return True
"""