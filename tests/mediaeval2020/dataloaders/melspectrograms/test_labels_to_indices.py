"""Test the label to indices convertion."""
from mediaeval2020.dataloaders.melspectrograms import labels_to_indices


class DummyLoader():
    """Dummy loader used for testing."""

    def __init__(self, classes):
        """Creates the dummy loader object.

        Args:
            classes: added to the configuration.
        """
        self.configuration = {'classes': classes}


def test_labels_to_indices():
    """Test if the label to index converter works as expected."""
    dataloader = DummyLoader(classes=[
        'mood/theme---a',
        'mood/theme---b',
        'mood/theme---c',
        'mood/theme---d',
        'mood/theme---e',
        'mood/theme---f',
        'mood/theme---g',
        'mood/theme---h',
    ])

    label_list = ['b', 'e', 'mood/theme---f', 'a', 'mood/theme---c']
    expected = [1, 4, 5, 0, 2]

    actual = labels_to_indices(dataloader=dataloader, label_list=label_list)

    assert (expected == actual).all()
