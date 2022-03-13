# -----------
# User Instructions:
# 
# Modify the function search so that it returns
# a table of values called expand. This table
# will keep track of which step each node was
# expanded.
#
# Make sure that the initial cell in the grid 
# you return has the value 0.
# ----------

grid = [[0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]]
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

delta = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1]] # go right

delta_name = ['^', '<', 'v', '>']

def search(grid,init,goal,cost):
    # ----------------------------------------
    # modify code below
    # ----------------------------------------
    print(len(grid))
    closed = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
    expand = [[-1 for row in range(len(grid[0]))] for col in range(len(grid))]
    action = [[-1 for row in range(len(grid[0]))] for col in range(len(grid))]
    
    closed[init[0]][init[1]] = 1
    x = init[0]
    y = init[1]
    g = 0
    open = [[x, y, g]]
    
    found = False
    resign = False
    count = 0
    
    print('initial open list:')
    for i in range(len(open)):
        print(open[i])
        print('----')
    
    while(found == False and resign == False):
        if(len(open) == 0):
            resign = True
            print("fail")
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            print("list items:", open)
            print('taken list item:', next)
            x = next[1]
            y = next[2]
            g = next[0]
            
            expand[x][y] = count
            count += 1
            
            if(x == goal[0] and y == goal[1]):
                found = True
                print(next)
            else:
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if(x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0])):
                        if(closed[x2][y2] == 0 and grid[x2][y2] == 0):
                            g2 = g + cost
                            print('append list item:', [g2, x2, y2])
                            open.append([g2, x2, y2])
                            closed[x2][y2] = 1
                            action[x2][y2] = i
                            
    print()
                        
    for i in range(len(expand)):
        print(expand[i])
    
    print()
        
    for i in range(len(action)):
        print(action[i])
        
    print()
    
    policy = [[' ' for row in range(len(grid[0]))] for col in range(len(grid))]
    x = goal[0]
    y = goal[1]
    policy[x][y] = '*'
    while(x != init[0] or y != init[1]):
        x2 = x - delta[action[x][y]][0]
        y2 = y - delta[action[x][y]][1]
        policy[x2][y2] = delta_name[action[x][y]]
        x = x2
        y = y2
    
    for i in range(len(policy)):
        print(policy[i])
                                
search(grid,init,goal,cost)
