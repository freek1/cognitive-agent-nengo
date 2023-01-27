import grid
import nengo
import nengo.spa as spa
import numpy as np 

#we can change the map here using # for walls and RGBMY for various colours
mymap="""
#######
#  M  #
# # # #
# #B# #
#G Y R#
#######
"""

linemap = """
###########
# G R G   #
###########
"""


#### Preliminaries - this sets up the agent and the environment ################ 
class Cell(grid.Cell):

    def color(self):
        if self.wall:
            return 'black'
        elif self.cellcolor == 1:
            return 'green'
        elif self.cellcolor == 2:
            return 'red'
        elif self.cellcolor == 3:
            return 'blue'
        elif self.cellcolor == 4:
            return 'magenta'
        elif self.cellcolor == 5:
            return 'yellow'
             
        return None

    def load(self, char):
        self.cellcolor = 0
        if char == '#':
            self.wall = True
            
        if char == 'G':
            self.cellcolor = 1
        elif char == 'R':
            self.cellcolor = 2
        elif char == 'B':
            self.cellcolor = 3
        elif char == 'M':
            self.cellcolor = 4
        elif char == 'Y':
            self.cellcolor = 5
            
            
world = grid.World(Cell, map=mymap, directions=int(4))

body = grid.ContinuousAgent()
world.add(body, x=1, y=2, dir=2)

#this defines the RGB values of the colours. We use this to translate the "letter" in 
#the map to an actual colour. Note that we could make some or all channels noisy if we
#wanted to
col_values = {
    0: [0.9, 0.9, 0.9], # White
    1: [0.2, 0.8, 0.2], # Green
    2: [0.8, 0.2, 0.2], # Red
    3: [0.2, 0.2, 0.8], # Blue
    4: [0.8, 0.2, 0.8], # Magenta
    5: [0.8, 0.8, 0.2], # Yellow
}

noise_val = 0.1 # how much noise there will be in the colour info

#You do not have to use spa.SPA; you can also do this entirely with nengo.Network()
model = spa.SPA()
with model:
    
    # create a node to connect to the world we have created (so we can see it)
    env = grid.GridNode(world, dt=0.005)

    ### Input and output nodes - how the agent sees and acts in the world ######

    #--------------------------------------------------------------------------#
    # This is the output node of the model and its corresponding function.     #
    # It has two values that define the speed and the rotation of the agent    #
    #--------------------------------------------------------------------------#
    def move(t, x):
        speed, rotation = x
        dt = 0.001
        max_speed = 20.0
        max_rotate = 10.0
        body.turn(rotation * dt * max_rotate)
        body.go_forward(speed * dt * max_speed)
        
    movement = nengo.Node(move, size_in=2)
    
    #--------------------------------------------------------------------------#
    # First input node and its function: 3 proximity sensors to detect walls   #
    # up to some maximum distance ahead                                        #
    #--------------------------------------------------------------------------#
    def detect(t):
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]
    proximity_sensors = nengo.Node(detect)

    #--------------------------------------------------------------------------#
    # Second input node and its function: the colour of the current cell of    #
    # agent                                                                    #
    #--------------------------------------------------------------------------#
    def cell2rgb(t):
        
        c = col_values.get(body.cell.cellcolor)
        noise = np.random.normal(0, noise_val,3)
        c = np.clip(c + noise, 0, 1)
        
        return c
        
    current_color = nengo.Node(cell2rgb)
     
    #--------------------------------------------------------------------------#
    # Final input node and its function: the colour of the next non-whilte     #
    # cell (if any) ahead of the agent. We cannot see through walls.           #
    #--------------------------------------------------------------------------#
    def look_ahead(t):
        
        done = False
        
        cell = body.cell.neighbour[int(body.dir)]
        if cell.cellcolor > 0:
            done = True 
            
        while cell.neighbour[int(body.dir)].wall == False and not done:
            cell = cell.neighbour[int(body.dir)]
            
            if cell.cellcolor > 0:
                done = True
        
        c = col_values.get(cell.cellcolor)
        noise = np.random.normal(0, noise_val,3)
        c = np.clip(c + noise, 0, 1)
        
        return c
        
    ahead_color = nengo.Node(look_ahead)    
    
    ### Agent functionality - your code adds to this section ###################
    #All input nodes should feed into one ensemble. Here is how to do this for
    #the radar, see if you can do it for the others
    
    walldist = nengo.Ensemble(n_neurons=480, dimensions=3, radius=4)
    nengo.Connection(proximity_sensors, walldist)

    #For now, all our agent does is wall avoidance. It uses values of the radar
    #to: a) turn away from walls on the sides and b) slow down in function of 
    #the distance to the wall ahead, reversing if it is really close
    
    # changed it below.
    
    
        #the movement function is only driven by information from the radar, so we
    #can connect the radar ensemble to the output node with this function 
    #directly. In the assignment, you will need intermediate steps
    
    # ---------------------------------
    # ---------------------------------
    ########## Freek's code ###########
    # ---------------------------------
    # ---------------------------------
    
    # ---------------------------------
    # Start of color recognition
    # ---------------------------------
    D = 16
    
    # Color conversion ensemble
    color_conversion = nengo.Ensemble(n_neurons=480, dimensions=3,\
        radius=2)
    
    '''
    The colors as provided:
    1: [0.2, 0.8, 0.2], # Green
    2: [0.8, 0.2, 0.2], # Red
    3: [0.2, 0.2, 0.8], # Blue
    4: [0.8, 0.2, 0.8], # Magenta
    5: [0.8, 0.8, 0.2], # Yellow
    '''
    
    # ---------------------------------
    # Mapping for higher dimension colors
    # ---------------------------------
    def mapping(x):
        '''
        Increasing the dimensionality of the color vectors
        to create more separated representations of the colors
        as vectors.
        Input x is assumed to be a list of length 3, representing rgb values.
        
        LIMITATION: due to substracting the rgb values from eachother
        to create higher dimensions, colors with the same values for r, g, and b
        (such as grey values: white, grey, black)) always have a value of 0
        after mapping.
        '''
        higher_dim = []
        higher_dim.append(x[0]-x[2])
        higher_dim.append(x[0]-x[1])
        higher_dim.append(x[1]-x[2])
        higher_dim.append(x[1]-x[0])
        higher_dim.append(x[2]-x[1])
        higher_dim.append(x[2]-x[0])
        
        for i in range(D-6):
            # Fill the other dimensions
            higher_dim.append(x[i%3]/10)
            # higher_dim.append(0)
        return higher_dim
    
    # ---------------------------------
    # Color vocabulary (mapped)
    # ---------------------------------
    # Variables for the colors
    GREEN = mapping(col_values[1])
    RED = mapping(col_values[2])
    BLUE = mapping(col_values[3])
    MAGENTA = mapping(col_values[4])
    YELLOW = mapping(col_values[5])
    
    colors = spa.Vocabulary(D)
    colors.add("GREEN", GREEN)
    colors.add("RED", RED)
    colors.add("BLUE", BLUE)
    colors.add("MAGENTA", MAGENTA)
    colors.add("YELLOW", YELLOW)
    colors.parse('YES + NO')

    # ---------------------------------
    # Vision state
    # ---------------------------------
    model.vision = spa.State(D, vocab=colors)
    
    # ---------------------------------
    # Individual color states
    # ---------------------------------
    
    # Setup to convert color to a higher dimension
    nengo.Connection(current_color, color_conversion)
    nengo.Connection(color_conversion, model.vision.input, function=mapping)
    
    model.mag_seen = spa.State(D, vocab=colors, feedback=1, \
        feedback_synapse=0.01)
    model.gre_seen = spa.State(D, vocab=colors, feedback=1, \
        feedback_synapse=0.01)
    model.yel_seen = spa.State(D, vocab=colors, feedback=1, \
        feedback_synapse=0.01)
    model.red_seen = spa.State(D, vocab=colors, feedback=1, \
        feedback_synapse=0.01)
    model.blu_seen = spa.State(D, vocab=colors, feedback=1, \
        feedback_synapse=0.01)
        
    # Error https://github.com/nengo/nengo/issues/805
    # to prevent biologically unplausible situations with passtrhough nodes
    # I use no real computation however, just a lower dimensional 
    # representation. (see integrate ensemble.)
    model.mag_seen.output.output = lambda t,x:x
    model.gre_seen.output.output = lambda t,x:x
    model.yel_seen.output.output = lambda t,x:x
    model.red_seen.output.output = lambda t,x:x
    model.blu_seen.output.output = lambda t,x:x
         
    # For color query
    model.answer = spa.State(D, vocab=colors)
    model.query = spa.State(D, vocab=colors)
        
    # ---------------------------------
    # Vision ahead
    # ---------------------------------

    ahd_color_conversion = nengo.Ensemble(n_neurons=480, dimensions=3,\
        radius=2)
    model.ahd_vision = spa.State(D, vocab=colors)
    
    # Setup to convert color to a higher dimension
    nengo.Connection(ahead_color, ahd_color_conversion)
    nengo.Connection(ahd_color_conversion, model.ahd_vision.input,\
        function=mapping)
    
    # ---------------------------------
    # Processing information for avoiding
    # ---------------------------------
    
    def isyes(x):
        '''returns 1 if the vector represents YES'''
        t = 0.8
        if np.dot(x, colors["YES"].v) > t:
            return 1
        else: return 0
    
    def iscolor(x):
        '''returns a vector representing one of the colors'''
        # threshold
        t = 0.65
        # keep track which color has the highest dot product
        maxi = 0
        # and the respective representation
        cols = [0,0,0,0,0]
        
        magsim = np.dot(x, colors["MAGENTA"].v)
        gresim = np.dot(x, colors["GREEN"].v)
        yelsim = np.dot(x, colors["YELLOW"].v)
        redsim = np.dot(x, colors["RED"].v)
        blusim = np.dot(x, colors["BLUE"].v)
        
        val = 5
        
        # if similarity passes threshold, and is the new higest,
        # then store new max value and update the return vector.
        if magsim > t and magsim > maxi:
            # print(magsim)
            maxi = magsim
            cols = [val, 0, 0, 0, 0]
        if gresim > t and gresim > maxi:
            maxi = gresim
            cols = [0, val, 0, 0, 0]
        if yelsim > t and yelsim > maxi:
            maxi = yelsim
            cols = [0, 0, val, 0, 0]
        if redsim > t and redsim > maxi:
            maxi = redsim
            cols = [0, 0, 0, val, 0]
        if blusim > t and blusim > maxi:
            maxi = blusim
            cols = [0, 0, 0, 0, val]
        return cols
  
    # readout of mapped ahead_color and interim ensemble to make ens_ahd_color
    # not a passthrough node, to enable functions on the connection.
    interim_ahd_color = nengo.Ensemble(n_neurons=400, dimensions=D)
    ens_ahd_color = nengo.Ensemble(n_neurons=400, dimensions=5, radius=1)
    # ens_ahd_color has five neurons which are 1 when a color is ahead.
    nengo.Connection(model.ahd_vision.output, interim_ahd_color)
    nengo.Connection(interim_ahd_color, ens_ahd_color, function=iscolor)
    
    # integrate ensemble to connect ahead color and current color information.
    # Has 5 neurons which are 1 when a color has been seen respectively.
    integrate = nengo.Ensemble(n_neurons=400, dimensions=5, radius=2)

    # connecting to integrate
    nengo.Connection(model.mag_seen.output, integrate[0], function=isyes)
    nengo.Connection(model.gre_seen.output, integrate[1], function=isyes)
    nengo.Connection(model.yel_seen.output, integrate[2], function=isyes)
    nengo.Connection(model.red_seen.output, integrate[3], function=isyes)
    nengo.Connection(model.blu_seen.output, integrate[4], function=isyes)

    # andgate ensemble to store 5 AND values
    andInterim = nengo.Ensemble(n_neurons=400, dimensions=5, radius=3)
    nengo.Connection(integrate, andInterim, transform=0.5)
    nengo.Connection(ens_ahd_color, andInterim, transform=0.5)
    
    def threshold(x):
        # and gate threshold
        return x > 0.7
    
    # process ensemble
    # has 5 values which are 1 when that color has been seen and is ahead ->
    # should avoid.
    # e.g. mag_seen AND ahd_color = magenta -> neuron[0] = 1
    process = nengo.Ensemble(n_neurons=500, dimensions=5, radius=1)
    nengo.Connection(andInterim, process, function=threshold)
    
    ########
    # inp = nengo.Node(size_in=5)
    # nengo.Connection(inp, process)
    
    def avoid(x):
        ''' Decide whether to avoid the upcoming color'''
        # input x is 5D vector of values. If one is about 1, the ahead square
        # should be avoided. 
        avoid = x > 0.65
        if avoid.any():
            # avoid
            return 1
        # else, dont avoid
        else:
            return 0
        
    def movement_func(x):
        ''' original movement function'''
        turn = x[2] - x[0]
        spd = x[1] - 0.35
        return spd, turn
        
    def act(x):
        '''input is 3D: [spd, trn, avoid?]
            output is 2D: [spd, trn] for movement ensemble
        '''
        if x[2] > 0.7:
            #avoid: stop and turn
            return [-1, -5]
        else:
            # just do object avoidance
            return x[0:2]
        
    # ensemble to control movement: [spd, trn, avoid?]
    motor = nengo.Ensemble(n_neurons=500, dimensions=3)
    
    # walldist sensors connect to first 2 dims for wall avoidance control
    # last dim is used for binary avoid? no/yes: 0 or 1
    nengo.Connection(walldist, motor[0:2], function=movement_func)
    nengo.Connection(process, motor[2], function=avoid)
    # act function controls the changing of action which passes on
    # a 2 dim vector of form: [spd, trn] again.
    nengo.Connection(motor, movement, function=act)
    
    # ---------------------------------
    # Input color sequence to follow
    # ---------------------------------
    
    # Input sequence of colors to search
    color_sequence = [GREEN, RED, BLUE]
    
    def check(x):
        '''input is 5D array of colors seen
        Function checks whether the color_sequence has been achieved.'''
        t = 0.8 # Threshold
        length = len(color_sequence) + 1
        for c in color_sequence:
            # color c is a D dimensional color vector
            # iscolor makes it the same 5d representation as x
            cols = iscolor(c)
            if np.dot(cols, x) > t:
                length -= 1
        if length == 0:
            # all asked colors have been seen
            return 1
        else: return 0
            
    isdone = nengo.Ensemble(500, 1)
    nengo.Connection(isdone, isdone, transform=1, synapse=0.01)
    nengo.Connection(integrate, isdone, function=check)
    
    # ---------------------------------
    # Spa actions
    # ---------------------------------
    actions = spa.Actions(
        # Store that the agent has seen a color
        "dot(vision, MAGENTA) --> mag_seen=YES",
        "dot(vision, GREEN) --> gre_seen=YES",
        "dot(vision, YELLOW) --> yel_seen=YES",
        "dot(vision, RED) --> red_seen=YES",
        "dot(vision, BLUE) --> blu_seen=YES",
        "0.5 --> 0",
        )
    
    query = spa.Actions(
        "answer = mag_seen * MAGENTA * ~query",
        "answer = yel_seen * YELLOW * ~query",
        "answer = gre_seen * GREEN * ~query",
        "answer = red_seen * RED * ~query",
        "answer = blu_seen * BLUE * ~query",
    )
    
    # ---------------------------------
    # Basal ganglia and thalamus
    # ---------------------------------
    
    model.bg = spa.BasalGanglia(actions)
    model.thalamus = spa.Thalamus(model.bg)
    model.cortical = spa.Cortical(query)
    
