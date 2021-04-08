#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:05:06 2021

@author: syxtreme
"""

import pandas as pd
import numpy as np
import argparse
from warnings import warn
from python_tsp.exact import solve_tsp_dynamic_programming
from matplotlib import pyplot as plt
from uuid import uuid4

class Direction():
    """Class to store and manipulate directions.
    """
    STAY = "stay"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    FORWARDS = "forwards"
    BACKWARDS = "backwards"

    """   O         U
        O   O     F   R

        O   O     L   B
          O         D
    """
    LIST = [  # list ordered according to portals
        "forwards", "up", "right",
        "left", "down", "backwards"
    ]

    @classmethod
    def invert(cls, direction):  # returns the inverse of a direction
        if direction == cls.STAY:
            return cls.STAY
        if direction == cls.UP:
            return cls.DOWN
        if direction == cls.DOWN:
            return cls.UP
        if direction == cls.LEFT:
            return cls.RIGHT
        if direction == cls.RIGHT:
            return cls.LEFT
        if direction == cls.FORWARDS:
            return cls.BACKWARDS
        if direction == cls.BACKWARDS:
            return cls.FORWARDS

class Cell():
    """Class to store information about the individual cells
    """
    M_MANHATAN = 1
    M_EUCLID = 2
    M_EUCLID_SQ = 3
    useMetric = M_EUCLID  # metric for the heuristic function

    def __init__(self, number, cube):
        """Initialize a cell

        Parameters
        ----------
        number : int
            The cell number.
        cube : Cube
            The parent cube object.
        """
        self._number = number
        self._cube = cube
        self._neighbors = {key: None for key in Direction.LIST}  # create an empty neighbor list
        self._pos = None  # 3D position is unknown, should be assigned later using "set_pos"

    def set_pos(self, x, y, z):
        """ Set position in the cube, in 3D coordinate system.
            X-axis corresponds to LEFT-RIGHT (right is positive)
            Y-axis corresponds to DOWN-UP (down is positive)
            Z-axis corresponds to FORWARDS-BACKWARDS (forward is positive)

        Parameters
        ----------
        x : int
        y : int
        z : int
        """
        self._cube.grid[x, y, z] = self
        self._pos = np.r_[x, y, z]

    def set_goal(self, goal):
        """Set goal and initialize for path search.

        Parameters
        ----------
        goal : Cell
            A goal for the path search.
        """
        if type(goal) is not Cell:
            goal = self._cube[goal]
        self._goal = goal
        self.route = 0
        self.parent = None
        self._heuristic = self.__heuristicFunction()

    def _add_neighbor(self, number, direction):
        """Adds a neighbor for the cell.

        Parameters
        ----------
        number : int
            The number of the neighboring cell.
        direction : str (Direction)
            The direction of the neighbor from this cell.
        """
        if number not in self._cube:
            self._cube.create_cell(number)
        self._neighbors[direction] = self._cube[number]

    def __getitem__(self, direction):
        return self._neighbors[direction]

    def __setitem__(self, direction, number):
        self._add_neighbor(number, direction)

    @property
    def number(self):
        return self._number

    @property
    def pos(self):
        return self._pos

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def heuristic(self):
        return self._heuristic

    @property
    def value(self):
        return self.route + self.heuristic

    def __heuristicFunction(self):
        # A function to compute distance estimate to the goal
        d = self.pos - self._goal.pos
        if self.useMetric == self.M_MANHATAN:
            return np.sum(d)
        elif self.useMetric == self.M_EUCLID:
            return np.linalg.norm(d)


class Cube():
    """A class that holds the information about the cube.
    """
    SIDE = 6  # number of cells per side
    MAX_DIST = 12  # fixed max distance search parameter
    USE_FIXED_MAX_DIST = False  # True if MAX_DIST should be used instead of the maxDist from data

    def __init__(self):
        self.cells = np.empty((self.SIDE**3, ), dtype=np.object)  # create a linear list of cells
        self.grid = np.empty((self.SIDE, self.SIDE, self.SIDE), dtype=np.object)  # create 3D grid of cells

    def add_cell(self, cell):
        """Adds an existing cell to the cube.

        Parameters
        ----------
        cell : Cell
            A cell.
        """
        self.cells[cell.number - 1] = cell

    def create_cell(self, cell_number):
        """Creates and adds a new cell to the cube.

        Parameters
        ----------
        cell_number : int
            The number of the cell.
        """
        self.add_cell(Cell(cell_number, self))

    def __getitem__(self, key):
        # Convenience function to get a cell by its number
        if type(key) is int:
            return self.cells[key - 1]
            # return next((c for c in self.cells if c is not None and c.number == key))
        elif type(key) is tuple:
            return self.grid[key]

    def __contains__(self, obj):
        if type(obj) is Cell:
            return obj in self.cells
        if type(obj) is int:
            return self[obj] is not None

    # def __setitem__(self, key, val):
    #     if type(key) is int:
    #         cell = next((c in self.cells if c.number == key))
    #         cell
    #     elif type(key) is tuple:
    #         return self.grid[key]

    def _gridify(self, cell, x, y, z):
        # A recursive function that organizes all known cells to a 3D grid.
        # Probably...
        setattr(cell, "visited", True)  # this is to know that a cell was already processed
        # Next, some conditions to wrap around the cube.
        if x < 0:
            x = self.SIDE - 1
        if y < 0:
            y = self.SIDE - 1
        if z < 0:
            z = self.SIDE - 1
        if x >= self.SIDE:
            x = 0
        if y >= self.SIDE:
            y = 0
        if z >= self.SIDE:
            z = 0
        cell.set_pos(x, y, z)  # set cell 3D position and add it to the grid
        for d, n in cell.neighbors.items():  # for all neighbors
            if n is None or hasattr(n, "visited"):  # if not processed before
                continue
            # got to the neighbor, change grid pos and process it
            if d == Direction.UP:
                self._gridify(n, x, y - 1, z)
            if d == Direction.DOWN:
                self._gridify(n, x, y + 1, z)
            if d == Direction.LEFT:
                self._gridify(n, x - 1, y, z)
            if d == Direction.RIGHT:
                self._gridify(n, x + 1, y, z)
            if d == Direction.FORWARDS:
                self._gridify(n, x, y, z + 1)
            if d == Direction.BACKWARDS:
                self._gridify(n, x, y, z - 1)

    def construct_grid(self):
        """Assigns all known cells to their positions in 3D grid
        """
        startn = self[1]
        self._gridify(startn, 0, 0, 0)
        for c in self.cells:  # cleanup
            if c is not None:
                delattr(c, "visited")

    def search(self, start_number, goal_number, silent=True):
        """Searches for a shortest path from the start to the goal.

        Parameters
        ----------
        start : int
            Starting cell number
        goal_number : int
            Goal cell number
        silent : bool, optional

        Returns
        -------
        tuple(list, list)
            path = list of cells to go through (including the start and the goal cells).
            directions = list of directions to take to go from the start to the goal.
        """
        if start_number not in self:  # if start_number is not yet in the cube
            warn("The starting cell is not yet known!")
            return
        if goal_number not in self:
            warn("The goal cell is not yet known!")
            return
        # some preparations
        maxDist = 0
        goal = self[goal_number]
        for c in self.cells:
            if c is not None:
                c.set_goal(goal)
                maxDist += 1

        # this function does the actual search
        # The node is the last cell in search, so should be the goal cell.
        node = self.__searchPath(start_number, maxDist)
        # if the search yielded no result or the result is not the goal.
        if node is None or node.number != goal_number:
            warn("Search did not find any path!")
            return
        path = []
        directions = []
        while node.parent:  # trace backwards from the goal to the start
            path.append(node)
            node = node.parent
        path.append(self[start_number])  # add the starting cell
        path.reverse()  # invert so that the path is from start to goal
        n = len(path)
        for i, node in enumerate(path):  # generate directions
            if i < n - 1:
                next_num = path[i + 1].number
            else:
                break
            directions.append(next((k for k, v in node.neighbors.items() if v.number == next_num)))
        if not silent:
            print("Found path\nPath length = {}\nUsed metric: {}.".format(len(path), Cell.useMetric))
        return path, directions

    def __searchPath(self, start, maxDist):
        # This is an implementation of an A* algorithm.
        ol = list()  # open list
        cl = list()  # closed list
        ol.append(self[start])
        if self.USE_FIXED_MAX_DIST:
            maxDist = self.MAX_DIST
        limit = 0
        while True:
            best = self.__findClosest(ol, maxDist)
            if best is None:
                warn("Open list is empty, could not find path!")
                return
            if best.heuristic == 0:
                break
            ol.remove(best)
            cl.append(best)
            for n in best.neighbors.values():
                if not self.__inlist(n, cl) and not self.__inlist(n, ol):
                    n.route = best.route + 1
                    n.parent = best
                    ol.append(n)
            limit += 1
            if limit > maxDist:
                warn(f"Distance limit of {maxDist} reached, could not find path!")
                return
        return best

    def __findClosest(self, li, maxDist):
        # Helper function for the search
        val = maxDist
        best = None
        for n in li:
            if n.value < val:
                val = n.value
                best = n
        return best

    def __inlist(self, node, li):
        # Helper function for the search
        for n in li:
            if np.all(n.pos == node.pos):
                return True
        return False

    def getPath(self, points):
        """Function to find path that goes through all requested points/cells.
        When only two numbers are provided (go from X to Y), a simple A* is used.
        When more numbers are provided, all-vs-all distances are computed and
        a TSP (traveling salesman problem) optimization is done to find the shortest
        route through all nodes.

        Parameters
        ----------
        points : list of ints
            List of cell numbers to go through.

        Returns
        -------
        tuple(list, list)
            A tuple of two lists. The first lists contains all the cell numbers on the path.
            The second list contains directions to take to go through all the cells.
        """
        n_points = len(points)
        if n_points == 0:
            warn(f"No point provided, nothing to do!")
        else:
            for p in points:  # check if all the points are known
                if p not in self:
                    warn(f"The cell number {p} is not yet known!")
                    return

        if n_points == 1:
            warn(f"Only one point provided! To go from cell {points[0]} to {points[0]}, simply do nothing!")
            return [self[points[0]], self[points[0]]], [Direction.STAY]
        elif n_points == 2:
            try:
                path, directions = self.search(points[0], points[1])
            except Exception as e:
                warn(f"I can't find a path from {points[0]} to {points[1]}!")
                return
            else:
                print(f"To go from {points[0]} to {points[1]}, use these directions:\n\t{directions}\n\tPath = {[c.number for c in path]}")
                return path, directions
        elif n_points > 2:
            paths = {}  # helper vars to keep the partial results
            dirs = {}
            dmat = np.zeros((n_points, n_points))  # compute a pair-wise distance matrix
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    try:
                        path, directions = self.search(points[i], points[j])
                    except Exception as e:
                        continue
                    paths[(i, j)] = path
                    paths[(j, i)] = list(reversed(path))  # inverse is the same but inverted
                    dirs[(i, j)] = directions
                    dirs[(j, i)] = [Direction.invert(d) for d in reversed(directions)]
                    dmat[i, j] = len(directions)
                    dmat[j, i] = len(directions)

            dmat[:, 0] = 0  # needed to solve open TSP
            # print(dmat)
            nodes, distance = solve_tsp_dynamic_programming(dmat)  # magic
            t_directions = []
            t_path = []
            # generate path and directions from the TSP result
            for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
                t_directions += dirs[a, b]
                if i == 0:
                    t_path += paths[a, b]
                else:
                    t_path += paths[a, b][1:]

            t_path_numbers = [c.number for c in t_path]
            all_in = True
            for p in points:  # check if the path really contains all the points
                if p not in t_path_numbers:
                    all_in = False
                    break
            if not all_in:
                warn(f"Could not find a path connecting al the points!")
                return

            print(f"To go through {points}, use these directions:\n\t{t_directions}\n\tPath = {t_path_numbers}")
            return t_path, t_directions

    @property
    def all_known(self):  # Returns True if all cells are known
        return not np.any([c is None for c in self.cells])

    @property
    def is_complete(self):  # Returns True if all cells are known and all their neighbors are known
        return not np.any([c is None or np.any([n is None for n in c.neighbors.values()]) for c in self.cells])


# %%
parser = argparse.ArgumentParser()
parser.add_argument("input_table", help="CSV file to be processed.")
parser.add_argument("--points", "-p", help="Point of the path", type=int, nargs="+")
parser.add_argument("--draw", "-d", help="Draws the slices of the cube.", action="store_true")
parser.add_argument("--draw-gates", "-g", help="Draws the slices of the cube with gates", action="store_true")

# args = parser.parse_args(["Mapa CUBE - Hárok1.csv"] + "-p 15 63".split())
# args = parser.parse_args(["Mapa CUBE - Hárok1.csv"] + "-p 15 63 86".split())
# args = parser.parse_args(["Mapa CUBE - Hárok1.csv"] + "-p 15 63 86 104 178".split())
# args = parser.parse_args(["Mapa CUBE - Hárok1.csv", "-g"])
# args = parser.parse_args()

table = pd.read_csv(args.input_table, header=None)
columns, rows = np.ogrid[0:len(table.columns):3, 0:table.index.stop:3]

cube = Cube()
for r in rows.ravel():
    for c in columns.ravel():
        if cube.is_complete:
            break
        data = table.iloc[r:r+3, c:c+3]
        if not data.shape == (3, 3):
            continue
        cell_num = data.iloc[1, 1]
        if np.isnan(cell_num):  # invalid cell
            continue
        cell_num = int(cell_num)
        neighbors = pd.concat((data.iloc[0, 0:3], data.iloc[2, 0:3])).to_numpy()
        if np.any(np.isnan(neighbors)):  # also invalid
            continue
        neighbors = neighbors.astype(np.int).tolist()

        if cell_num not in cube:
            cube.create_cell(cell_num)
        cell = cube[cell_num]
        for n, d in zip(neighbors, Direction.LIST):
            cell[d] = n

cube.construct_grid()
#%%
# path, directions = cube.search(1, 216)
# print(directions)
# print([c.number for c in path])

if args.draw:
    fig, axs = plt.subplots(2, 3)
    i = 0
    for rax in axs:
        for ax in rax:
            ax.axis('tight')
            ax.axis('off')
            plane = []
            for row in cube.grid[:, :, i]:
                plane.append([c.number for c in row])
            ax.table(plane,loc='center')
            i += 1
    plt.savefig(f"{uuid4()}.png")

if args.draw_gates:
    for i in range(cube.SIDE):
        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')
        plane = []
        for row in cube.grid[:, :, i]:
            line = []
            for c in row:
                nn = np.asanyarray([c[d].number
                                    for d in Direction.LIST]).reshape(2, 3)
                elem = np.vstack([nn[0], [0, c.number, 0], nn[1]])
                line.append(elem)
            plane.append(np.hstack(line))
        plane = np.vstack(plane)
        ax.table(plane.tolist(), loc='center')
        plt.savefig(f"{uuid4()}_{i}.png")

if args.points is not None:
    points = np.asarray(args.points).ravel().tolist()
    print(f"Path points: {points}")
    cube.getPath(points)
