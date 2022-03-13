import numpy as np
import matplotlib.pyplot as pl
import sys
sys.path.append('osr_examples/scripts/')
import environment_2d
import random
import math

MAX_X = 10
MAX_Y = 6
RADIUS = 1

def cal_straight_line_dist(coord1, coord2):
  distance = math.sqrt(pow((coord1[0] - coord2[0]), 2) + pow((coord1[1] - coord2[1]), 2))
  return distance

def check_if_neighbors(coord1, coord2):
  distance = cal_straight_line_dist(coord1, coord2)
  if distance <= RADIUS:
    return True
  return False

def add_neighbors(sample1, sample2, index = 0):
  sample1.add_neighbors(sample2)
  sample2.add_neighbors(sample1)
  coord1 = sample1.get_coord()
  coord2 = sample2.get_coord()
  if index == 1:
    pl.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], 'g', linestyle="--", linewidth = 0.5)
  else:
    pl.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], 'g', linestyle="-", linewidth = 0.2)

def merge_group(sample1, sample2, groups):
  sample1_group = sample1.get_group()
  sample2_group = sample2.get_group()
  if sample1_group != sample2_group:
    groups[sample1_group] = groups[sample1_group] + groups[sample2_group]
    for change_sample in groups[sample2_group]:
      change_sample.set_group(sample1_group)
    groups.pop(sample2_group, None)

def reset_plt():
  np.random.seed(4)
  env = environment_2d.Environment(MAX_X, MAX_Y, 5)
  pl.clf()
  env.plot()
  q = env.random_query()
  if q is not None:
    x_start, y_start, x_goal, y_goal = q
    env.plot_query(x_start, y_start, x_goal, y_goal)
  return env, q

def path_planning(env, q):
  # CONDITIONS:
  # 1. qstart can be connected to a (nearby) vertex u;
  # 2. qgoal can be connected to a (nearby) vertex v;
  # 3. u and v are in the same connected component of G (in the graph-theoretic sense)

  samples = []
  groups = {}
  x_start, y_start, x_goal, y_goal = q
  start = environment_2d.Sample(x_start, y_start, -1)
  end = environment_2d.Sample(x_goal, y_goal, -1)
  run = True
  main_group = None
  running = 0

  # random sample will be generated until all conditions are fufilled
  while run:
    if running % 100 == 0:
      print("Random sample of", running,"created...")
    running = running + 1
    x = random.random() * MAX_X
    y = random.random() * MAX_Y
    if env.check_collision(x, y) == False:
      group_no = len(samples)
      new_sample = environment_2d.Sample(x, y, group_no)
      samples.append(new_sample)
      groups[group_no] = [new_sample]

      # find neighbors
      for i in range(len(samples)-1):
        sample = samples[i]
        if check_if_neighbors(new_sample.get_coord(), sample.get_coord()) == True and env.check_intersect(new_sample.get_coord(), sample.get_coord()) == True:
          add_neighbors(new_sample, sample)
          merge_group(new_sample, sample, groups)

      # check if can be connected to qstart or qgoal
      if check_if_neighbors(new_sample.get_coord(), [x_start, y_start]) == True and env.check_intersect(new_sample.get_coord(), [x_start, y_start]) == True:
        add_neighbors(new_sample, start, 1)
      elif check_if_neighbors(new_sample.get_coord(), [x_goal, y_goal]) == True and env.check_intersect(new_sample.get_coord(), [x_goal, y_goal]) == True:
        add_neighbors(new_sample, end, 1)

      for u in start.neighbors:
        for v in end.neighbors:
          if u.get_group() == v.get_group():
            run = False
            main_group = u.get_group()
            break
        if run == False:
          break
  print("Total sample of", running,"created.")
  return groups[main_group], start, end

def astar_search(nodes, start, end):
  nodes.insert(0, start)
  nodes.append(end)
  goal_key = len(nodes)-1
  mapping = {} # obj -> index
  visited = []
  parent = []
  gf = []
  hf = []
  pq = {} # index -> gf + hf

  for i in range(len(nodes)):
    mapping[nodes[i]] = i
    visited.append(0)
    parent.append(None)
    gf.append(np.Infinity)
    hf.append(np.Infinity)

  gf[0] = 0
  dist = cal_straight_line_dist(start.get_coord(), end.get_coord())
  hf[0] = dist
  pq[0] = gf[0] + hf[0]
  while len(pq) != 0:
    u_key = min(pq, key=pq.get)
    if u_key == goal_key:
      break
    visited[u_key] = 1
    pq.pop(u_key, None)
    current_node = nodes[u_key]
    for neighbors in current_node.neighbors:
      if neighbors not in mapping:
        continue
      v_key = mapping[neighbors]
      dist = cal_straight_line_dist(neighbors.get_coord(), end.get_coord())
      if visited[v_key] != 1 and gf[u_key]+1 + dist < gf[v_key] + hf[v_key]:
        gf[v_key] = gf[u_key]+1
        hf[v_key] = dist
        parent[v_key] = u_key
        pq[v_key] = gf[v_key] + hf[v_key]

  main_nodes = []
  temp = goal_key
  while True:
    temp2 = parent[temp]
    sample0 = nodes[temp]
    main_nodes.append(sample0)
    if temp2 == None:
      break
    sample1 = nodes[temp2]
    temp = temp2
    coord0 = sample0.get_coord()
    coord1 = sample1.get_coord()
    pl.plot([coord0[0], coord1[0]], [coord0[1], coord1[1]], 'b.', linestyle="-", linewidth = 0.5, markersize = 1)

  return main_nodes

def post_processing(env, nodes):
  # FOR rep = 1 TO maxrep
  # Pick two random points t1, t2 along the path     
  # IF the straight segment [q(t1),q(t2)] is collision-free THEN
  #   Replace original portion by the straight segment
  shortest_node = []
  start = 0
  while True:
    reset = False
    for i in range(start, len(nodes)):
      for j in range(len(nodes)-1, i, -1):
        node1 = nodes[i]
        node2 = nodes[j]
        if env.check_intersect(node1.get_coord(), node2.get_coord()) == True:
          coord1 = node1.get_coord()
          coord2 = node2.get_coord()
          pl.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], 'b.', linestyle="-", linewidth = 0.5, markersize = 1)
          start = j
          reset = True
          break
      shortest_node.append(nodes[i])
      if reset:
        break
    if reset == False: # no more node utilisation
      break

  return

def main():
  # path planning (PRM)
  env, q = reset_plt()
  main_group, start, end = path_planning(env, q)
  pl.show()

  # path finding (based on astar)
  env, q = reset_plt()
  path_node = astar_search(main_group, start, end)
  pl.show()
  
  # post processing
  env, q = reset_plt()
  post_processing(env, path_node)
  pl.show()

main()