import ruptures as rpt
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Union, Optional
from pathlib import Path

from musicaiz.loaders import Musa
from musicaiz.features import (
    get_novelty_func,
    musa_to_graph
)


LEVELS = ["high", "mid", "low"]


@dataclass
class PeltArgs:
    penalty: int
    model: str
    minsize: int
    jump: int


class StructurePrediction:

    def __init__(
        self,
        file: Optional[Union[str, Path]] = None,
    ):

        # Convert file into a Musa object to be processed
        if file is not None:
            self.midi_object = Musa(
                file=file,
            )
        else:
            self.midi_object = Musa(file=None)

    def notes(self, level: str) -> List[int]:
        return self._get_structure_boundaries(level)

    def beats(self, level: str) -> List[int]:
        result = self._get_structure_boundaries(level)
        return [self.midi_object.notes[n].beat_idx for n in result]

    def bars(self, level: str) -> List[int]:
        result = self._get_structure_boundaries(level)
        return [self.midi_object.notes[n].bar_idx for n in result]

    def ms(self, level: str) -> List[float]:
        result = self._get_structure_boundaries(level)
        return [self.midi_object.notes[n].start_sec * 1000 for n in result]

    def _get_structure_boundaries(
        self,
        pelt_args: PeltArgs,
    ):
        """
        Get the note indexes where a section ends.
        """
        g = musa_to_graph(self.midi_object)
        mat = nx.attr_matrix(g)[0]
        n = get_novelty_func(mat)
        nn = np.reshape(n, (n.size, 1))
        # detection
        algo = rpt.Pelt(
            model=pelt_args.model,
            min_size=pelt_args.minsize,
            jump=pelt_args.jump
        ).fit(nn)
        result = algo.predict(pen=pelt_args.penalty)
        return result


@dataclass
class WindowArgs:
    penalty: int
    model: str
    width: int


class StructurePredictionWindow:

    def __init__(
        self,
        file: Optional[Union[str, Path]] = None,
    ):

        # Convert file into a Musa object to be processed
        if file is not None:
            self.midi_object = Musa(
                file=file,
            )
        else:
            self.midi_object = Musa(file=None)

    def notes(self, level: str) -> List[int]:
        return self._get_structure_boundaries(level)

    def beats(self, level: str) -> List[int]:
        result = self._get_structure_boundaries(level)
        return [self.midi_object.notes[n].beat_idx for n in result]

    def bars(self, level: str) -> List[int]:
        result = self._get_structure_boundaries(level)
        return [self.midi_object.notes[n].bar_idx for n in result]

    def ms(self, level: str) -> List[float]:
        result = self._get_structure_boundaries(level)
        return [self.midi_object.notes[n].start_sec * 1000 for n in result]

    def _get_structure_boundaries(
        self,
        pelt_args: PeltArgs,
    ):
        """
        Get the note indexes where a section ends.
        """
        g = musa_to_graph(self.midi_object)
        mat = nx.attr_matrix(g)[0]
        n = get_novelty_func(mat)
        nn = np.reshape(n, (n.size, 1))
        # detection
        algo = rpt.Window(
            model=pelt_args.model,
            width=pelt_args.width,
        ).fit(nn)
        result = algo.predict(pen=pelt_args.penalty)
        return result