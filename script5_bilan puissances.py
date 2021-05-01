# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt



def recuperationDonneesEclairement(nomFichier) :
    """
     Permet de récupérer un fichier d'eclairement
     Parametre d'entree :
        nomFichier : Nom du fichier contenant les donnees d'eclairement
     Parametres de sortie :
        temps : Instant des points de mesure [s]
        eclairement [W/m2]
    Usage :
        temps, eclairement = recuperationDonneesEclairement("eclairement_juin.txt")
    """

    with open(nomFichier, 'r') as fic:
        lines = [line.strip() for line in fic.readlines()]
    del lines[0]
    temps_brut        = np.zeros(np.shape(lines))
    eclairement_brut  = np.zeros(np.shape(lines))
    for k in range(len(lines)):
        colonnes         = lines[k].split(' ')
        temps_brut[k]    = float(colonnes[0])
        eclairement_brut[k] = float(colonnes[1])  
    
    return temps_brut, eclairement_brut






def recuperationDonneesParcours(nomFichier) :
    """
     Permet de récupérer un fichier issu du tracé GPS
     Parametre d'entree :
        nomFichier : Nom du fichier contenant les donnees du parcours
     Parametres de sortie :
        temps : Instant des points de mesure [s]
        distance : Distance parcourue [m]
        altitude : Altitude [m]
    Usage :
        temps_brut, altitude_brut, distance_brut = recuperationDonneesParcours("marmotte.txt")
    """

    with open(nomFichier, 'r') as fic:
        lines = [line.strip() for line in fic.readlines()]
    del lines[0]
    temps_brut     = np.zeros(np.shape(lines))
    altitude_brut  = np.zeros(np.shape(lines))
    distance_brut  = np.zeros(np.shape(lines))
    for k in range(len(lines)):
        colonnes         = lines[k].split(' ')
        temps_brut[k]    = float(colonnes[2])/1000
        altitude_brut[k] = float(colonnes[0])/1000       
        distance_brut[k] = float(colonnes[1])/1000    
    
    return temps_brut, altitude_brut, distance_brut


def denivele(y):
    """
    Calcul de la dénivelée positive
    """
    deniv = 0
    
    for i in range(len(y)-1):
        if y[i+1]>y[i]:
            deniv += (y[i+1]-y[i])
    
    print("Denivele =", deniv, "m")
    return(deniv)


def glissante(x,Nmoy):
    """ 
    s = glissante(x,Nmoy)
    retourne un vecteur s issu du filtrage par moyenne glissante de largeur 2*Nmoy+1 du vecteur x 
    """
    porte = np.ones(2*Nmoy+1)/(2*Nmoy+1)
    Nb    = len(x)
    foo   = np.concatenate((x,np.ones(2*Nmoy)*x[Nb-1]))-x[0] # on complete le signal par 2Nmoy fois la  derniere valeur puis on enleve la valeur initiale
    s     = np.convolve(porte,foo)+x[0]
    return s[Nmoy:Nb+Nmoy]


def integration(x,y, c):
    """
    Calcul de l'integrale d'un vecteur
    """
    Nech = len(x)
    z = np.zeros(Nech)
    
    
    for k in range(Nech) :
        if k==0:
            z[k] = c
        else:
            z[k] = z[k-1]+(x[k]-x[k-1])*y[k]
    
    return(z)


def derivation(x,y):
    """
    Calcul de la dérivée d'un vecteur (l[k+1]-l[k-1])
    """
    Nech = len(x)
    z = np.zeros(Nech)
    
    
    for k in range(Nech) :
        if k == 0:
            z[k] = (y[1]-y[0])/(x[1]-x[0])
        elif k<Nech-1 :
            z[k] = (y[k+1]-y[k-1])/(x[k+1]-x[k-1])
        elif k == Nech-2:
            z[k] = (y[k+1]-y[k])/(x[k+1]-x[k])
    
    return(z)



#Récupération données d'éclairement
temps_E,eclairement_juin = recuperationDonneesEclairement("Eclairement_juin.txt")
temps_j,eclairement_janvier = recuperationDonneesEclairement("Eclairement_janvier.txt")


#Récupération des donnnées de parcours
temps_brut, altitude_brut, distance_brut = recuperationDonneesParcours('marmotte.txt')

#Interpolation des courbes
temps = np.arange(temps_brut[0], temps_brut[len(temps_brut)-1], 0.1)
print(temps_brut[len(temps_brut)-1]-temps_brut[0])
altitude = np.interp(temps, temps_brut, altitude_brut)
distance = np.interp(temps, temps_brut, distance_brut)
#Interpolation de l'éclairement de juin
eclairement = np.interp(temps, temps_E, eclairement_juin)



#Filtrage de la distance
distance_m = glissante(distance, 5000)

#Calcul de la vitesse
vitesse = derivation(temps, distance_m)

#Calcul de l'acceleration
acceleration = derivation(temps, vitesse)



#Filtrage de l'altitude
altitude_m = glissante(altitude, 1000)

#Calcul de la pente
pente = derivation(distance_m, altitude_m)

#Calcul du denivele
denivele = denivele(altitude)



#Calcul de la force
f = 0.015
m = 100
g = 9.81
Scx = 0.23
Fres = f*m*g + Scx*(vitesse*vitesse)
force = m*acceleration+Fres+m*g*np.sin(pente)

#Calcul de la puissance
puissance = force*vitesse

#Calcul de l'energie
energie = integration(temps, puissance, 0)/3600






### DEBUT BILAN DES PUISSANCES ###

#Axes
plt.grid('on')
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')

#variables :
s_pv = 2   #en m2
n_pv = 0.15 #rendement Panneau
n_h1 = 0.85 #rendement hacheur 1
n_h2 = 0.85 #rendement hacheur 2
n_mot = 0.80 #rendement moteur
W_stock_init = 350 #charge initiale de la batterie
n_stock = 0.64 #rendement de stockage de la batterie
b_assist = 0.5  #coeff d'assistance - 1=beaucoup d'assistance, 0=pas beaucoup d'assistance
b_frein = 0.5  #coeff de récupération lors du freinage
r = 0.311 #rayon de la roue



# CHARGE :

#Panneau solaire :
P_pv = eclairement*s_pv*n_pv #puissance fournie par le panneau
#Affichage :
plt.plot(temps/3600, P_pv, color = "gray", linewidth = 0.5)



#1er hacheur :
P_h1 = n_h1*P_pv #puissance de charge
#Affichage :
#plt.plot(temps/3600, P_h1, color = "orange", linewidth = 0.5)




# DECHARGE :

#puissance du vélo :
#plt.plot(temps/3600, puissance, color = "green", linewidth = 0.5)
#energie du vélo :
#plt.plot(temps/3600, energie, color = "purple", linewidth = 0.5)

#Calcul de la puissance mécanique :
P_m = np.zeros(len(temps))
P_frein = np.zeros(len(temps))
P_cycliste = np.zeros(len(temps))


for i in range(len(temps)-1):
    if(puissance[i] < 0):
        P_m[i] = b_frein*puissance[i]
        P_cycliste[i] = 0
    else:
        P_frein[i] = 0
        
        P_m[i] = b_assist*puissance[i]
        P_cycliste[i] = puissance[i]-P_m[i]
        
# Affichage :
plt.plot(temps/3600, P_m, color = "red", linewidth = 0.5)

# Calcul de la puissance électrique
P_a = np.zeros(len(temps))

for j in range(len(temps)-1):
    if(P_m[j] < 0):
        P_a[j] = n_mot*P_m[j]
    else:
        P_a[j] = P_m[j]/n_mot

# Affichage :
#plt.plot(temps/3600, P_a, color = "blue", linewidth = 0.5)

# 2eme hacheur
P_var = np.zeros(len(temps))

for k in range(len(temps)-1):
    if(P_a[k] < 0):
        P_var[k] = P_a[k]*n_h2
    else:
        P_var[k] = P_a[k]/n_h2
        
# Affichage :
#plt.plot(temps/3600, P_var, color = "orange", linewidth = 0.5)


# STOCKAGE

# batterie
P_bat = P_h1-P_var

#Affichage :
#plt.plot(temps/3600, P_bat, color = "gray", linewidth = 0.5)


#pertes de stockage :

P_stock = np.zeros(len(temps))

for l in range(len(temps)-1):
    if(P_bat[l] < 0):
        P_stock[l] = P_bat[l]/n_stock
    else:
        P_stock[l] = P_bat[l]*n_stock

#Energie batterie :
W_stock = integration(temps/3600, P_bat, W_stock_init)
#Affichage :
plt.plot(temps/3600, W_stock, color = "black", linewidth = 0.5)



#Rapport couple/vitesse de rotation
omega = vitesse/r
C = P_m/omega

#Affichage :
#plt.plot(omega, C, color = "black", linewidth = 0.5)



plt.ylabel("Puissance (W), énergie (Wh)")
plt.xlabel("Temps (h)")


### FIN BILAN DES PUISSANCES ###








