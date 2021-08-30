def tiensleutels(dict):
    """Tien sleutels van een dictionary."""
    lijst = list(dict)
    return(lijst[0:9])
    
def tienelementen(dict):
    """Tien elementen van dictionary tonen."""
    string = ""
    for sleutel in tiensleutels(dict):
        string +=  "\'"+ sleutel +  "\': " + str(dict[sleutel]) + "\n"
    return string

