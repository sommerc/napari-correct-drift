import numpy as np

from napari_correct_drift import CorrectDriftDock, TableWidget


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_correct_drift_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    my_widget = CorrectDriftDock(viewer)

    assert 1 == 1


def test_table_widget():
    random_drifts = np.random.rand(16, 3)
    # viewer = make_napari_viewer()

    tw = TableWidget(viewer=None)
    tw.set_content(random_drifts)
    tw.save(filename="tmp.csv")

    tw.load(filename="tmp.csv")

    assert np.allclose(random_drifts, tw.get_content())
