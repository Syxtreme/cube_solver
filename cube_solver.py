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
import open3d as o3d


class Direction():
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    FORWARDS = "forwards"
    BACKWARDS = "backwards"
    
    LIST = [
        "forwards", "up", "right",
        "left", "down", "backwards"
    ]
    
    @classmethod
    def invert(cls, direction):
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
    M_MANHATAN = 1
    M_EUCLID = 2
    M_EUCLID_SQ = 3

    useMetric = M_EUCLID

    
    def __init__(self, number, cube):
        self._number = number
        self._cube = cube
        self._neighbors = {key: None for key in Direction.LIST}
        self._pos = None
    
    def set_pos(self, x, y, z):
        self._pos = np.r_[x, y, z]
        
    def set_goal(self, goal):
        self._goal = goal
        self.route = 0
        self.parent = None
        self._heuristic = self.__heuristicFunction()
        
    def _add_neighbor(self, number, direction):
        if number not in self._cube:
            self._cube.create_cell(number)
        self._neighbors[direction] = self._cube[number]
        
    def __getitem__(self, key):
        return self._neighbors[key]
        
    def __setitem__(self, key, val):
        self._add_neighbor(val, key)
    
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
        d = self.pos - self._goal.pos
        if self.useMetric == self.M_MANHATAN:
            return np.sum(d)
        elif self.useMetric == self.M_EUCLID:
            return np.linalg.norm(d)

        
class Cube():
    SIDE = 6
    MAX_DIST = 12
    USE_FIXED_MAX_DIST = False

    def __init__(self):
        self.cells = np.empty((self.SIDE**3, ), dtype=np.object)
        self.grid = np.empty((self.SIDE, self.SIDE, self.SIDE), dtype=np.object)
    
    def add_cell(self, cell):
        self.cells[cell.number - 1] = cell
        
    def create_cell(self, cell_number):
        self.add_cell(Cell(cell_number, self))
        
    def __getitem__(self, key):
        if type(key) is int:
            return self.cells[key - 1]
            # return next((c for c in self.cells if c is not None and c.number == key))
        elif type(key) is tuple:
            return self.grid[key]
        
    def __contains__(self, obj):
        if type(obj) is int:
            return self[obj] is not None
        
    # def __setitem__(self, key, val):
    #     if type(key) is int:
    #         cell = next((c in self.cells if c.number == key))
    #         cell
    #     elif type(key) is tuple:
    #         return self.grid[key]
    def _gridify(self, cell, x, y, z):
        setattr(cell, "visited", True)
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
        self.grid[x, y, z] = cell
        cell.set_pos(x, y, z)
        for d, n in cell.neighbors.items():
            if n is None or hasattr(n, "visited"):
                continue
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
        startn = self[1]
        self._gridify(startn, 0, 0, 0)
        for c in self.cells:
            if c is not None:
                delattr(c, "visited")

    def search(self, start, goal, silent=True):
        maxDist = 0
        goal = self[goal]
        for c in self.cells:
            if c is not None:
                c.set_goal(goal)
                maxDist += 1
                
        node = self.__searchPath(start, maxDist)
        path = []
        if node is None:
            warn("searchPath returned did not find any path!")
        else:
            while node.parent:
                path.append(node)
                node = node.parent
            path.append(self[start])
            path.reverse()
            n = len(path)
            directions = []
            for i, node in enumerate(path):
                if i < n - 1:
                    next_num = path[i + 1].number
                else:
                    break
                directions.append(next((k for k, v in node.neighbors.items() if v.number == next_num)))
            if not silent:
                print("Found path\nPath length = {}\nUsed metric: {}.".format(len(path), Cell.useMetric))
        return path, directions

    def __searchPath(self, start, maxDist):
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
        val = maxDist
        best = None
        for n in li:
            if n.value < val:
                val = n.value
                best = n
        return best
    
    def __inlist(self, node, li):
        for n in li:
            if np.all(n.pos == node.pos):
                return True
        return False
    
    def getPath(self, points):
        n_points = len(points)
        if n_points == 2:
            try:
                path, directions = self.search(points[0], points[1])
            except Exception as e:
                warn(f"I can't find a path from {points[0]} to {points[1]}!")
                return
            else:
                print(f"To go from {points[0]} to {points[1]}, use these directions:\n\t{directions}\n\tPath = {[c.number for c in path]}")
                return path, directions
        elif n_points > 2:
            paths = {}
            dirs = {}
            dmat = np.zeros((n_points, n_points))
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    try:
                        path, directions = self.search(points[i], points[j])
                    except Exception as e:
                        continue
                    paths[(i, j)] = path
                    paths[(j, i)] = list(reversed(path))
                    dirs[(i, j)] = directions
                    dirs[(j, i)] = [Direction.invert(d) for d in reversed(directions)]
                    dmat[i, j] = len(directions)
                    dmat[j, i] = len(directions)
                    
            dmat[:, 0] = 0  # needed to solve open TSP
            # print(dmat)
            nodes, distance = solve_tsp_dynamic_programming(dmat)
            t_directions = []
            t_path = []
            for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
                t_directions += dirs[a, b]
                if i == 0:
                    t_path += paths[a, b]
                else:
                    t_path += paths[a, b][1:]

            print(f"To go through {points}, use these directions:\n\t{t_directions}\n\tPath = {[c.number for c in t_path]}")
            return t_path, t_directions
        
    @property
    def is_complete(self):
        return not np.any([c is None for c in self.cells])

        
# %%
parser = argparse.ArgumentParser()
parser.add_argument("input_table", help="CSV file to be processed.")
parser.add_argument("--points", "-p", help="Point of the path", type=int, nargs="+")

# args = parser.parse_args(["Mapa CUBE - Hárok1.csv"] + "-p 15 63".split())
# args = parser.parse_args(["Mapa CUBE - Hárok1.csv"] + "-p 15 63 86".split())
args = parser.parse_args(["Mapa CUBE - Hárok1.csv"] + "-p 15 63 86 104 178".split())
# args = parser.parse_args()

points = np.asarray(args.points).ravel().tolist()
print(f"Path points: {points}")

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
# path, directions = cube.search(1, 216)
# print(directions)
# print([c.number for c in path])

cube.getPath(points)