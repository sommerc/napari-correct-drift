Napari-Correct-Drift's doc
==========================

Fiji’s Correct-3D-drift macro for Napari
----------------------------------------
In time-lapse imaging, a motorized microscope repeatedly captures images at specified positions for long periods. However, microscope stages exhibit *drift*, causing the sample to move, or to appear moving.

The reasons are complex and manifold, but it typically results from error propagation in odometry, thermal expansion, and mechanical movement. Drift poses problems for subsequent image analysis and needs to be corrected.

With Napari-correct-drift provides an extensible solution with similar functionality as Fiji’s Correct-3D-drift macro. It offers efficient cross-correlation using Fourier phase correlationmethod, improved key frame selection, and outlier handling. In Napari users can provide a regions-of-interest (ROI) to effectively stabilize objects in up-to 3D-multi-channel images. Additionally, estimated drifts can be exported, imported, or edited before applying the correction.

When to use this plugin
-----------------------

#. Your time-series images or volumes exhibit *translational* drift, i. e. rigid movement, without rotation.
#. Reference channel with fixed object (e. g. fiducial) visualizing the drift
#. Stabilizing objects of interest by using a ROIs

Without Napari viewer
---------------------

Napari-Correct-Drift can also be used without starting the Napari viewer.

.. code-block:: python

   from napari_correct_drift import CorrectDrift

   # multi-channel 2D-movie
   cd = CorrectDrift(img_in, "tcyx")

   # estimate drift table
   drifts = cd.estimate_drift(t0=0, channel=0)

   # correct drift
   img_cor = cd.apply_drifts(drifts)

Issues and contributing
-----------------------

If you have any problems or question running the plugin, please open an `issue <https://github.com/sommerc/napari-correct-drift>`_


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install
   introduction
   how_to
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
