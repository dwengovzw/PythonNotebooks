
# Check if a graph is still connected, the graph is represente as an adjacency list
def is_graaf_verbonden(graaf):
    # Lege knopen zien we als verwijderd uit de graaf, filter de lege lijsten weg uit de graaf
    niet_lege_knopen = [index for index, knoop in enumerate(graaf) if len(knoop) > 0]
    aantal_knopen_in_graaf = len(niet_lege_knopen)
    if (len(niet_lege_knopen) == 0): # Als alle knopen leeg zijn, dan is de graaf nog steeds verbonden
        return True
    
    # We maken een lijst aan met alle knopen die we nog moeten bezoeken
    # We starten met de eerste niet lege knoop in de graaf
    te_bezoeken = [niet_lege_knopen[0]]
    
    # We maken een lijst aan met alle knopen die we al bezocht hebben
    bezocht = []
    
    # Zolang er nog knopen zijn die we moeten bezoeken
    while len(te_bezoeken) > 0:
        # We nemen de eerste knoop uit de lijst van knopen die we nog moeten bezoeken
        huidige_knoop = te_bezoeken.pop(0)
        # We voegen de huidige knoop toe aan de lijst van knopen die we al bezocht hebben
        bezocht.append(huidige_knoop)
        # We voegen alle knopen die nog niet bezocht zijn en die verbonden zijn met de huidige knoop toe aan de lijst van knopen die we nog moeten bezoeken
        for knoop in graaf[huidige_knoop]:
            if knoop not in bezocht and knoop not in te_bezoeken:
                te_bezoeken.append(knoop)
    # Als alle knopen bezocht zijn, dan is de graaf nog steeds verbonden
    return len(bezocht) == aantal_knopen_in_graaf

def boog_knipt_graaf_in_twee(graaf, van_knoop, naar_knoop):
    # Controleer of de naar_knoop meer dan 1 boog heeft (de boog terug naar de van_knoop)
    if (len(graaf[naar_knoop]) == 1):
        # Als van_knoop en naar_knoop de enige knopen zijn die nog overblijven in de graaf, dan knipt de boog de graaf niet in twee
        if (sum([len(knoop) for knoop in graaf])== 2) and (len(graaf[naar_knoop]) == 1) and (len(graaf[van_knoop]) == 1):
            return False
        else:
            # Als het niet de twee laatste knopen zijn dan knipt de boog de graaf in twee
            return True
    else:
        # Als de naar_knoop meer dan 1 boog heeft, verwijder de boog (van_knoop, naar_knoop) en controleer of de graaf nog steeds verbonden is
        verwijder_boog_uit_graaf(graaf, van_knoop, naar_knoop)
        if is_graaf_verbonden(graaf):
            voeg_boog_toe_aan_graaf(graaf, van_knoop, naar_knoop)
            return False
        else:
            # voeg de boog terug toe aan de graaf
            voeg_boog_toe_aan_graaf(graaf, van_knoop, naar_knoop)
            return True

def verwijder_boog_uit_graaf(graaf, van_knoop, naar_knoop):
    graaf[van_knoop].remove(naar_knoop)
    graaf[naar_knoop].remove(van_knoop)

def voeg_boog_toe_aan_graaf(graaf, van_knoop, naar_knoop):
    if graaf[van_knoop] == None:
        graaf[van_knoop] = []
    graaf[van_knoop].append(naar_knoop)
    if graaf[naar_knoop] == None:
        graaf[naar_knoop] = []
    graaf[naar_knoop].append(van_knoop)



