import matplotlib.pyplot as plt

def maak_staafdiagram_polariteiten(polariteiten, frequenties):
    """Staafdiagram plotten. """
    plt.xticks([0, 1], polariteiten) # benoem x-as
    plt.bar([0, 1], frequenties, align='center') # maak staafdiagram 
    plt.show() # toon grafiek