from abc import ABC, abstractmethod
from ..surface import FWHSurface


class SurfaceReader(ABC):
    """
    Base class for surface data readers.

    Subclass this and implement read() for any new input format
    (SU2, OpenFOAM, HDF5, etc). The solver never needs to change.
    """

    @abstractmethod
    def read(self) -> FWHSurface:
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}()"