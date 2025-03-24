# import time
# from recognition import detect_cube_accurate
# from movement import move_arm_to_cube, move_to_yellow_place, move_to_green_place, move_to_red_place, move_to_blue_place, take_cube, initialize_arm, look_at_platform



# def main():

#     # Initialisation du bras robotisé
#     initialize_arm()
#     look_at_platform()

#     # Boucle pour détecter et déplacer les cubes
#     while True:
#         # Détection du cube et prédiction de la couleur
#         predicted_class = detect_cube_accurate()

#         # Affichage du résultat dans la console
#         print(f"Cube détecté: {predicted_class}")

#         # Si un cube est détecté, il faut le déplacer
#         if predicted_class:
#             # Déplacement vers le cube
#             move_arm_to_cube(cube_position=(320, 240))  # Position hypothétique du cube au centre de l'image

#             # Prendre le cube
#             take_cube()

#             # Déplacer le cube au bon endroit en fonction de sa couleur
#             if predicted_class == 'cube_red':
#                 move_to_red_place()
#             elif predicted_class == 'cube_yellow':
#                 move_to_yellow_place()
#             elif predicted_class == 'cube_green':
#                 move_to_green_place()
#             elif predicted_class == 'cube_blue':
#                 move_to_blue_place()

#             # Attente avant de passer au prochain cube
#             time.sleep(2)

#         # # Quitter en appuyant sur 'q'
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break

# if __name__ == "__main__":
#     main()


import time
from recognition import detect_cube_accurate
from movement import move_arm_to_cube, move_to_yellow_place, move_to_green_place, move_to_red_place, move_to_blue_place, take_cube, initialize_arm, look_at_platform


import sys
import importlib

sys.path.append('/home/jetson/Dofbot/0.py_install')
Arm_Lib = importlib.import_module('Arm_Lib')
Arm_Device = Arm_Lib.Arm_Device
arm = Arm_Device()



def main():
    arm.Arm_serial_set_torque(1)

    # Initialisation du bras robotisé
    initialize_arm()
    look_at_platform()

    # Boucle pour détecter et déplacer les cubes
    while True:
        # Détection du cube et prédiction de la couleur
        predicted_class = detect_cube_accurate()

        # Affichage du résultat dans la console
        if predicted_class:
            print(f"Cube détecté: {predicted_class}")
        else:
            print("Aucun cube détecté")

        # Si un cube est détecté, il faut le déplacer
        if predicted_class:
            # Déplacement vers le cube
            move_arm_to_cube(cube_position=(320, 240))  # Position hypothétique du cube au centre de l'image

            # Prendre le cube
            take_cube()

            # Déplacer le cube au bon endroit en fonction de sa couleur
            if predicted_class == 'cube_red':
                move_to_red_place()
            elif predicted_class == 'cube_yellow':
                move_to_yellow_place()
            elif predicted_class == 'cube_green':
                move_to_green_place()
            elif predicted_class == 'cube_blue':
                move_to_blue_place()

            # Attente avant de passer au prochain cube
            time.sleep(2)

if __name__ == "__main__":
    main()
