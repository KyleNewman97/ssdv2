import pytest

from ssdv2.structs.exceptions import DatasetError


class TestDatasetError:
    def test_raise(self):
        """
        Test that we can raise a DatasetError.
        """
        with pytest.raises(DatasetError):
            raise DatasetError("Error")
