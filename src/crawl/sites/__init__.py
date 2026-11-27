"""Registry of crawl sources. Each module documents its access situation in its docstring."""

from .b3ta import B3ta
from .dprevived import Dprevived
from .fotokto import Fotokto
from .mineo import Mineo
from .photoshopcontest import PhotoshopContest
from .photoshopgurus import PhotoshopGurus
from .realedit import RealEdit
from .reddit import Reddit
from .voz import Voz
from .zhopped import Zhopped

SOURCES = {cls.name: cls for cls in (Dprevived, Voz, Mineo, Fotokto, Zhopped, Reddit, PhotoshopGurus, RealEdit, PhotoshopContest, B3ta)}
