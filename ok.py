
salaire_net = "3 364.85"

#Relevé de compte
file_txt = open("out/compte.txt","w+")
lines = file_txt.readlines()
fraude = True
for line in lines:
    print(line)
    if salaire_net in line:
        print("Il y a bien un virement qui correspond au montant affiché sur le buletin de salaire.")
        fraude = False
if fraude == True:
    print("ATTENTION ! Il n'y a pas de virement qui correspond au montant inscrit sur la fiche de paie.")

