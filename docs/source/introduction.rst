Introduction
============

Example
-------
Let's start with an example of an growing `root-tip <https://seafile.ist.ac.at/f/b05362d4f358430c8c59/?dl=1>`_ imaged in a vertical microscope:

.. raw:: html

    <video controls  width="640" src="https://seafile.ist.ac.at/f/de0c4cff54cf46dcbfbc/?dl=1"></video>

Main widget
-----------
The main widget is structured in several groups:

.. image:: _static/widget_02.png
  :width: 320
  :alt: Napari-Correct-Drift's main widget

The five groups are:

1. `Input Axes`_
2. `Correct Drift`_
3. `Key Frames`_
4. `Parameters`_
5. `Outliers`_


Input Axes
^^^^^^^^^^
Select the Napari layer to process using the drop-down element. Once the layer is selected, make sure the dimensions are correctly set. When using multidimensional images (with more than 3 dimensions), Napari-Correct-Drift needs to know which dimension corresponds to the *Z*, *Channel* and *Time* dimension. The Time dimension always needs to be assigned. Use the drop-down elements per Axis to set *Z*, *Channel* and *Time*. Napari-Correct-Drift displays the size of the selected dimension for convenience in brackets.

Correct Drift
^^^^^^^^^^^^^
Run Napari-Correct-Drift using the set parameters. One can **Estimate Drift** or **Load Drift** from a .csv file. Both options will open the Napari-Correct-Drift `Table Widget`_. The table contains the Z,Y,X drifts per time frame.

Select **Correct Drift** to apply the drifts from the table widget to your image data. A new image layer containing the corrected image will be created.

One can follow the progress of the estimation and the correctiong step in the Napari notifications.

Key Frames
^^^^^^^^^^

Parameters
^^^^^^^^^^

Outliers
^^^^^^^^

Table Widget
------------
