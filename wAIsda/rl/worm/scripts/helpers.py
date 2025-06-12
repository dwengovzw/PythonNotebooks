
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy

# Maak een worm klasse die n segmenten heeft, een segment kan ofwel lang ofwel kort zijn.
class Worm:
    def __init__(self, n):
        self.segments = []
        self.segment_positions = [] # The position of the front of each segment on thet x-axis.
        # Create enum for expanded and contracted states
        self.states = {'EXPANDED': 1, 'CONTRACTED': 0}
        for i in range(n):
            self.segments.append(self.states['CONTRACTED']) # Initialize all segments to contracted
        
        # The last segment is at position 1 and the first segment is at position n    
        for i in range(1, n + 1):
            self.segment_positions.append(i)
            
    def maak_segment_langer(self, index):
        # Do nothing if the segment is already expanded
        if self.segments[index] == self.states['EXPANDED']:
            return
        # If the segment cannot be expanded because it is between two contracted segments, do nothing
        between_contracted, left_contracted, right_contracted = self.zit_segment_tussen_twee_korte_segmenten(index)
        if between_contracted:
            return
        # If there is a contracted segment to the left, increase the position of this segment and all segments to the right
        if left_contracted:
            for i in range(index, len(self.segments)):
                self.segment_positions[i] += 1
                
        # If there is a contracted segment to the right, decrease the position of all segments to the left, not the segment itself since the front of the segment is at the same position
        if right_contracted:
            for i in range(index):
                self.segment_positions[i] -= 1
                
        # If there are no other contracted segments, the segment can be expanded
        if (not left_contracted) and (not right_contracted):
            # all segments to the left of the segment will move half a position to the left
            for i in range(index):
                self.segment_positions[i] -= 0.5
                
            # all segments to the right of the segment and the segment itself will move half a position to the right
            for i in range(index, len(self.segments)):
                self.segment_positions[i] += 0.5
            
        if 0 <= index < len(self.segments):
            self.segments[index] = self.states['EXPANDED']
        else:
            raise IndexError("Segment index out of range")
        
    def maak_segment_korter(self, index):
        # Do nothing if segment is already contracted
        if self.segments[index] == self.states['CONTRACTED']:
            return
        # If the segment cannot be contracted because it is between two contracted segments, do nothing
        between_contracted, left_contracted, right_contracted = self.zit_segment_tussen_twee_korte_segmenten(index)
        if between_contracted:
            return
        # If there is a contracted segment to the left, decrease the position of this segment and all segments to the right
        if left_contracted:
            for i in range(index, len(self.segments)):
                self.segment_positions[i] -= 1
        # If there is a contracted segment to the right, increase the position of all segments to the left, not the segment itself since the front of the segment is at the same position
        if right_contracted:
            for i in range(index):
                self.segment_positions[i] += 1
                
        # If there are no other contracted segments, the segment can be contracted
        if (not left_contracted) and (not right_contracted):
            # all segments to the left of the segment will move half a position to the right
            for i in range(index):
                self.segment_positions[i] += 0.5
                
            # all segments to the right of the segment and the segment itself will move half a position to the left
            for i in range(index, len(self.segments)):
                self.segment_positions[i] -= 0.5
                
        # Set the segment to contracted
        if 0 <= index < len(self.segments):
            self.segments[index] = self.states['CONTRACTED']
        else:
            raise IndexError("Segment index out of range") 
        
    # Check if the contracting or expanding segment is in between any two contracted segments anywhere on the worm        
    def zit_segment_tussen_twee_korte_segmenten(self, index):
        if 0 <= index < len(self.segments):
            # Check if any segment to the left is contracted
            left_contracted = any(self.segments[i] == self.states['CONTRACTED'] for i in range(index))
            # Check if any segment to the right is contracted
            right_contracted = any(self.segments[i] == self.states['CONTRACTED'] for i in range(index + 1, len(self.segments)))
            return left_contracted and right_contracted, left_contracted, right_contracted
        else:
            raise IndexError("Segment index out of range")
        
    def segmenten(self):
        return self.segments
    
    def posities_van_de_segmenten(self):
        return self.segment_positions
    
    def positie_van_het_hoofd(self):
        if self.segment_positions:
            return self.segment_positions[len(self.segment_positions) - 1]  # Return the position of the last segment
        
    def toestand(self):
        string_state = ''.join(['L' if state == self.states['EXPANDED'] else 'K' for state in self.segments])
        tuple_state = tuple(string_state)
        return tuple_state
    
    def toestand_kopie(self):
        """
        Returns a copy of the current state of the worm.
        This is useful for saving the state before making changes.
        """
        return deepcopy(self.toestand())
    
    def maak_kopie(self):
        """
        Returns a deep copy of the current worm instance.
        This is useful for saving the state before making changes.
        """
        return deepcopy(self)
    
    # String representation of the worm
    def __str__(self):
        return ''.join(['L' if state == self.states['EXPANDED'] else 'K' for state in self.segments])
    
    # Representation of the worm with its segments
    def __repr__(self):
        return f"Worm({len(self.segments)}) with segments: {self.segmenten()}"
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, index):
        if 0 <= index < len(self.segments):
            return self.segments[index]
        else:
            raise IndexError("Segment index out of range")
        
        


    def teken_worm(self):
        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_aspect('equal')
        
        y_center = 1  # Vertical position of the worm
        for i, (state, pos) in enumerate(zip(self.segmenten(), self.posities_van_de_segmenten())):
            if state == self.states['CONTRACTED']:
                # Draw square 1x1 centered vertically
                rect = patches.Rectangle((pos - 1, y_center - 0.5), 1, 1, facecolor='pink', edgecolor='black')
            else:  # EXPANDED
                # Draw rectangle 2x0.5 centered vertically
                rect = patches.Rectangle((pos - 2, y_center - 0.25), 2, 0.5, facecolor='pink', edgecolor='black')
            ax.add_patch(rect)
            ax.text(pos-1, y_center + 0.7, f"{pos - 1}", ha='center', fontsize=8)  # Label with segment index
        
        # Set limits and labels
        min_pos = 0
        max_pos = 12
        ax.set_xlim(min_pos, max_pos)
        ax.set_ylim(0, 1.5)
        ax.axis('off')
        plt.show()


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from IPython.display import display
from copy import copy, deepcopy
from matplotlib.animation import FuncAnimation

class WormMatplotlibAnimator:
    def __init__(self, worm, scale=1.0):
        self.worm = Worm(len(worm))  # Create a new Worm instance with the same length
        self.worm.segments = deepcopy(worm.segmenten())
        self.worm.segment_positions = deepcopy(worm.posities_van_de_segmenten())
        self.scale = scale
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_ylim(0, 2)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_xlim(-1, max(worm.posities_van_de_segmenten()) * scale + 30)

        self.patches = []
        self.draw_initial_worm()

    def draw_initial_worm(self):
        for state, pos in zip(self.worm.segmenten(), self.worm.posities_van_de_segmenten()):
            rect = self.make_patch(state, pos)
            self.ax.add_patch(rect)
            self.patches.append(rect)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def make_patch(self, state, pos):
        x = pos * self.scale
        if state == self.worm.states["CONTRACTED"]:
            return Rectangle((x - self.scale, 0), self.scale, self.scale, color='pink')
        else:
            return Rectangle((x - 2 * self.scale, self.scale/4), 2 * self.scale, 0.5 * self.scale, color='pink')

    def update_patches(self):
        # Remove old patches
        for patch in self.patches:
            patch.remove()
        self.patches = []

        for state, pos in zip(self.worm.segmenten(), self.worm.posities_van_de_segmenten()):
            rect = self.make_patch(state, pos)
            self.ax.add_patch(rect)
            self.patches.append(rect)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        
    def update(self, worm):
        """
        Update the internal worm and redraw the patches.
        This method is called to update the worm's state and redraw it.
        """
        assert isinstance(worm, Worm), "Argument must be a Worm instance"
        assert len(worm) == len(self.worm), "Worm length mismatch"

        # Update internal worm's segments and segment_positions
        self.worm.segments = deepcopy(worm.segmenten())
        self.worm.segment_positions = deepcopy(worm.posities_van_de_segmenten())

        # Redraw the worm
        self.update_patches()
        
from IPython.display import Video

def create_worm_animation(worm_states, scale=1.0, interval=500, filename="worm_animation.mp4", fps=1):
    """
    Create an animation of the worm's states.
    
    Parameters:
    - worm_states: List of Worm instances representing the states of the worm.
    - scale: Scaling factor for the worm's size.
    - interval: Time interval between frames in milliseconds.
    """
    animator = WormMatplotlibAnimator(worm_states[0], scale)
    anim = FuncAnimation(animator.fig, animator.update, frames=worm_states, interval=interval, repeat=False)
    anim.save(filename, writer="ffmpeg", fps=fps)
    # Video(filename, width=640, height=360) 
    
from itertools import product
import numpy as np
def maak_q_tabel(aantal_segmenten):
    toestanden = list(product("KL", repeat= aantal_segmenten))
    acties = []
    for i in range(3):
        acties.append("L" + str(i + 1))
        acties.append("K" + str(i + 1))
    q_tabel = np.zeros((len(toestanden), len(acties)))
    
    return q_tabel, toestanden, acties

def print_q_tabel(q_tabel, toestanden, acties):
    print("Q-tabel:")
    print("Toestand\\actie\t" + "\t".join([''.join(map(str, actie)) for actie in acties]))
    for i, toestand in enumerate(toestanden):
        print(f"{''.join(map(str, toestand))}\t\t" + "\t".join(map(lambda x: str("%.2f" % round(x, 2)), q_tabel[i])))
        
        
def voer_policy_uit(q_tabel, toestanden, acties, bestandsnaam="worm_geleerde_policy.mp4"):
    worm_length = q_tabel.shape[1]//2 # Number of segments is half the number of actions
    # Execute the learned policy and add the states to the worm_states list
    worm_toestanden = []
    worm = Worm(worm_length)
    toestand_index = toestanden.index(worm.toestand())
    worm_toestanden.append(deepcopy(worm))
    for stap in range(100):
        actie_index = np.argmax(q_tabel[toestand_index])  # Choose the action with the highest Q-value
        action = acties[actie_index]
        index = int(action[1])
        if action[0] == 'E':
            worm.maak_segment_langer(index)
        else:
            worm.maak_segment_korter(index)
            
        nieuwe_toestand_index = toestanden.index(worm.toestand())
        worm_toestanden.append(deepcopy(worm))
        toestand_index = nieuwe_toestand_index
        
    # Create the animation of the learned policy
    create_worm_animation(worm_toestanden, scale=1.0, interval=500, filename=bestandsnaam)
    
def maak_animatie_van_worm(toestanden_van_de_worm, bestandsnaam="worm_animatie.mp4", interval=500, fps=1):
    return create_worm_animation(toestanden_van_de_worm, scale=1.0, interval=interval, filename=bestandsnaam, fps=fps)
    
    
def lees_bestaande_q_tabel(bestandsnaam):
    """
    Lees een bestaande Q-tabel uit een npy bestand.
    
    Parameters:
    - bestandsnaam: Naam van het bestand waarin de Q-tabel is opgeslagen.
    
    Returns:
    - q_tabel: De geladen Q-tabel.
    """
    try:
        q_tabel = np.load(bestandsnaam)
        return q_tabel
    except FileNotFoundError:
        print(f"Bestand {bestandsnaam} niet gevonden.")
        return None