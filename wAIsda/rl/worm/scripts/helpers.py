
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
