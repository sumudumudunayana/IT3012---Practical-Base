import math
from collections import deque
import heapq



class SearchAgent:
    """Agent implementing BFS, DFS, and UCS graph search."""

    def __init__(self):
        self.actions = ['Up', 'Down', 'Left', 'Right']

        # Step 1.3: Store the offline plan
        self.plan = []

        # Select the search algorithm
        # Change to 'DFS' or 'UCS' to compare algorithms
        #self.active_algo = 'BFS'
        self.active_algo = 'AStar'


    # =========================================================
    # STEP 1.1 - HEURISTIC FUNCTIONS
    # =========================================================
    def manhattan_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal

        return abs(x1 - x2) + abs(y1 - y2)

    def euclidean_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal

        return math.sqrt(
            (x1 - x2) ** 2 +
            (y1 - y2) ** 2
        )

    # =========================================================
    # STEP 1.3 - SENSE AND ACT
    # =========================================================
    def sense_and_act(self, percept: dict) -> str:

        # If the current plan is empty, create a new plan
        if not self.plan:

            # Get information from the percept
            start = percept['agent_pos']
            all_food = percept['all_food']
            walls = set(percept['walls'])
            grid_size = percept['grid_size']

            # If there is no food remaining, stop
            if not all_food:
                return 'Suck'

            # -------------------------------------------------
            # Find the closest food pellet
            # -------------------------------------------------
            closest_food = min(
                all_food,
                key=lambda food:
                abs(start[0] - food[0]) +
                abs(start[1] - food[1])
            )

            # -------------------------------------------------
            # Select the search algorithm
            # -------------------------------------------------
            if self.active_algo == 'BFS':

                self.plan = self.bfs_search(
                    start,
                    closest_food,
                    walls,
                    grid_size
                )

            elif self.active_algo == 'DFS':

                self.plan = self.dfs_search(
                    start,
                    closest_food,
                    walls,
                    grid_size
                )

            elif self.active_algo == 'UCS':

                self.plan = self.ucs_search(
                    start,
                    closest_food,
                    walls,
                    grid_size
                )
            elif self.active_algo == 'AStar':

                self.plan = self.astar_search(
                    start,
                    closest_food,
                    walls,
                    grid_size,
                    heuristic_type='manhattan'
                )
            

            else:
                # Invalid algorithm
                return 'Right'

        # -----------------------------------------------------
        # Execute the first action from the plan
        # -----------------------------------------------------
        if self.plan:
            return self.plan.pop(0)

        # -----------------------------------------------------
        # Fallback if no path is found
        # -----------------------------------------------------
        return 'Right'

    # =========================================================
    # BFS - Breadth First Search
    # =========================================================
    def bfs_search(self, start, goal, walls, grid_size):

        # FIFO queue
        queue = deque()

        # Store starting state and empty path
        queue.append((start, []))

        # Reached set prevents revisiting states
        reached = {start}

        while queue:

            # FIFO: remove the first item
            current, path = queue.popleft()

            # Goal test
            if current == goal:
                return path

            # Expand the current state
            for action in self.actions:

                next_state = self.get_next_position(
                    current,
                    action,
                    walls,
                    grid_size
                )

                # Only add new valid states
                if (
                    next_state is not None
                    and next_state not in reached
                ):
                    reached.add(next_state)

                    queue.append(
                        (
                            next_state,
                            path + [action]
                        )
                    )

        # No path found
        return []

    # =========================================================
    # DFS - Depth First Search
    # =========================================================
    def dfs_search(self, start, goal, walls, grid_size):

        # LIFO stack
        stack = []

        # Store starting state and empty path
        stack.append((start, []))

        # Reached set
        reached = {start}

        while stack:

            # LIFO: remove the last item
            current, path = stack.pop()

            # Goal test
            if current == goal:
                return path

            # Expand the current state
            for action in self.actions:

                next_state = self.get_next_position(
                    current,
                    action,
                    walls,
                    grid_size
                )

                # Only add new valid states
                if (
                    next_state is not None
                    and next_state not in reached
                ):
                    reached.add(next_state)

                    stack.append(
                        (
                            next_state,
                            path + [action]
                        )
                    )

        # No path found
        return []

    # =========================================================
    # UCS - Uniform Cost Search
    # =========================================================
    def ucs_search(self, start, goal, walls, grid_size):

        # Priority queue
        frontier = []

        # Counter is used to break ties between equal costs
        counter = 0

        # Starting node has cost 0
        heapq.heappush(
            frontier,
            (0, counter, start, [])
        )

        # Store the lowest cost reached for each state
        reached = {
            start: 0
        }

        while frontier:

            # Remove node with lowest path cost
            cost, _, current, path = heapq.heappop(frontier)

            # Goal test
            if current == goal:
                return path

            # Expand current state
            for action in self.actions:

                next_state = self.get_next_position(
                    current,
                    action,
                    walls,
                    grid_size
                )

                if next_state is None:
                    continue

                # Every movement currently costs 1
                new_cost = cost + 1

                # Add if state has not been reached
                # or if we found a cheaper path
                if (
                    next_state not in reached
                    or new_cost < reached[next_state]
                ):

                    reached[next_state] = new_cost

                    counter += 1

                    heapq.heappush(
                        frontier,
                        (
                            new_cost,
                            counter,
                            next_state,
                            path + [action]
                        )
                    )

        # No path found
        return []


    
    # =========================================================
    # GENERATE NEXT STATE
    # =========================================================
    def get_next_position(
        self,
        position,
        action,
        walls,
        grid_size
    ):

        x, y = position
        width, height = grid_size

        # Calculate next position
        if action == 'Up':
            next_position = (x, y + 1)

        elif action == 'Down':
            next_position = (x, y - 1)

        elif action == 'Left':
            next_position = (x - 1, y)

        elif action == 'Right':
            next_position = (x + 1, y)

        else:
            return None

        nx, ny = next_position

        # Check grid boundaries
        if nx < 0 or nx >= width:
            return None

        if ny < 0 or ny >= height:
            return None

        # Check walls
        if next_position in walls:
            return None

        # Valid next state
        return next_position


# =========================================================
# A* - A STAR SEARCH
# =========================================================
def astar_search(
    self,
    start_pos,
    goal_pos,
    walls,
    grid_size,
    heuristic_type='manhattan'
):

    # Priority queue
    frontier = []

    # Starting node
    g_cost = 0

    # Calculate heuristic
    if heuristic_type == 'manhattan':
        h_cost = self.manhattan_distance(
            start_pos,
            goal_pos
        )

    elif heuristic_type == 'euclidean':
        h_cost = self.euclidean_distance(
            start_pos,
            goal_pos
        )

    else:
        # Default to Manhattan
        h_cost = self.manhattan_distance(
            start_pos,
            goal_pos
        )

    # f(n) = g(n) + h(n)
    f_cost = g_cost + h_cost

    # Tuple format:
    # (f_cost, g_cost, current_pos, path_taken)
    heapq.heappush(
        frontier,
        (
            f_cost,
            g_cost,
            start_pos,
            []
        )
    )

    # Keep track of reached states
    reached_states = set()

    # Process priority queue
    while frontier:

        # Get node with lowest f(n)
        f_cost, g_cost, current_pos, path_taken = heapq.heappop(
            frontier
        )

        # Goal test
        if current_pos == goal_pos:
            return path_taken

        # Skip if already reached
        if current_pos in reached_states:
            continue

        # Mark current state as reached
        reached_states.add(current_pos)

        # Expand four possible actions
        for action in self.actions:

            next_pos = self.get_next_position(
                current_pos,
                action,
                walls,
                grid_size
            )

            # Ignore invalid or already reached states
            if (
                next_pos is None
                or next_pos in reached_states
            ):
                continue

            new_g_cost = g_cost + 1

            # Calculate heuristic
            if heuristic_type == 'manhattan':
                new_h_cost = self.manhattan_distance(
                    next_pos,
                    goal_pos
                )

            elif heuristic_type == 'euclidean':
                new_h_cost = self.euclidean_distance(
                    next_pos,
                    goal_pos
                )

            else:
                new_h_cost = self.manhattan_distance(
                    next_pos,
                    goal_pos
                )

            # f(new) = g(new) + h(new)
            new_f_cost = new_g_cost + new_h_cost

            # Add new node to priority queue
            heapq.heappush(
                frontier,
                (
                    new_f_cost,
                    new_g_cost,
                    next_pos,
                    path_taken + [action]
                )
            )

    # No path found
    return []