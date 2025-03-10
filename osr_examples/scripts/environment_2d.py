#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as pl


class TriangularObstacle(object):
  def __init__(self, x0, y0, x1, y1, x2, y2):
    self.x0 = x0
    self.y0 = y0
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2

    self.A = np.zeros((3,2))
    self.C = np.zeros(3)

    a = x1 - x0
    b = y1 - y0
    c = x2 - x0
    d = y2 - y0
    if -b*c + a*d > 0:
      self.A[0, :] = -b, a
    else:
      self.A[0, :] = b, -a
    self.C[0] = np.dot(self.A[0, :], np.array([x0,y0]))

    a = x2 - x1
    b = y2 - y1
    c = x0 - x1
    d = y0 - y1
    if -b*c + a*d > 0:
      self.A[1, :] = -b, a
    else:
      self.A[1, :] = b, -a
    self.C[1] = np.dot(self.A[1, :], np.array([x1,y1]))

    a = x0 - x2
    b = y0 - y2
    c = x1 - x2
    d = y1 - y2
    if -b*c + a*d > 0:
      self.A[2, :] = -b, a
    else:
      self.A[2, :] = b, -a
    self.C[2] = np.dot(self.A[2, :], np.array([x2,y2]))


  def contains(self, x, y):
    r = np.dot(self.A, np.array([x,y])) - self.C
    return all([i>0 for i in r])

  def plot(self):
    pl.plot([self.x0,self.x1], [self.y0,self.y1], "r" , linewidth = 2)
    pl.plot([self.x1,self.x2], [self.y1,self.y2], "r" , linewidth = 2)
    pl.plot([self.x2,self.x0], [self.y2,self.y0], "r" , linewidth = 2)        


class Environment(object):
  def __init__(self, size_x, size_y, n_obs):
    self.size_x = size_x
    self.size_y = size_y
    self.obs = []
    for i in range(n_obs):
      x0 = np.random.rand()*size_x
      y0 = np.random.rand()*size_y
      x1 = np.random.rand()*size_x
      y1 = np.random.rand()*size_y
      x2 = np.random.rand()*size_x
      y2 = np.random.rand()*size_y
      self.obs.append(TriangularObstacle(x0, y0, x1, y1, x2, y2))

  def check_collision(self, x, y):
    for ob in self.obs:
      if ob.contains(x, y):
        return True
    return False

  def line_intersection_test(self, s0, s1, t0, t1):
    p0_x, p0_y = s0
    p1_x, p1_y = s1
    p2_x, p2_y = t0
    p3_x, p3_y = t1

    s1_x = float(p1_x - p0_x)
    s1_y = float(p1_y - p0_y)
    s2_x = float(p3_x - p2_x)
    s2_y = float(p3_y - p2_y)
    outcome = 0

    s = float (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y)
    t = float ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y)

    if (s >= 0 and s <= 1 and t >= 0 and t <= 1):
        outcome = 1

    return outcome

  def check_intersect(self, s0, s1):
    for ob in self.obs:
      p0 = [ob.x0, ob.y0]
      p1 = [ob.x1, ob.y1]
      p2 = [ob.x2, ob.y2]
      
      side_1_test = self.line_intersection_test(s0, s1, p0, p1)
      side_2_test = self.line_intersection_test(s0, s1, p1, p2)
      side_3_test = self.line_intersection_test(s0, s1, p0, p2)

      result = side_1_test + side_2_test + side_3_test

      if result > 0:
        return False

    return True

  def random_query(self):
    max_attempts = 100
    found_start = False
    found_goal = False
    for i in range(max_attempts):
      x_start = np.random.rand()*self.size_x
      y_start = np.random.rand()*self.size_y
      if not self.check_collision(x_start, y_start):
        found_start = True
        break
    for i in range(max_attempts):
      x_goal = np.random.rand()*self.size_x
      y_goal = np.random.rand()*self.size_y
      if not self.check_collision(x_goal, y_goal):
        found_goal = True
        break
    if found_start and found_goal:
      return x_start, y_start, x_goal, y_goal
    else:
      return None

  def plot(self):
    pl.plot([0, self.size_x, self.size_x, 0, 0], [0, 0, self.size_y, self.size_y, 0], "k", linewidth = 2)
    for ob in self.obs:
      ob.plot()

  def plot_query(self, x_start, y_start, x_goal, y_goal):
    pl.plot([x_start], [y_start], "bs", markersize = 8)
    pl.plot([x_goal], [y_goal], "y*", markersize = 12)

class Sample(object):
  def __init__(self, x, y, group):
    self.x = x
    self.y = y
    self.group = group
    self.neighbors = []
    pl.plot([x], [y], "g.", markersize = 1)

  def get_coord(self):
    return [self.x, self.y]

  def get_group(self):
    return self.group

  def set_group(self, group):
    self.group = group

  def add_neighbors(self, sample):
    self.neighbors.append(sample)

