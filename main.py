# -*- coding: utf-8 -*-

import time
from recognition import detect_cube_and_predict, VideoCamera
from movement import move_arm_to_cube, move_to_yellow_place, move_to_green_place, move_to_red_place, move_to_blue_place, take_cube

# Initialisation de la caméra et du robot
camera = VideoCamera()

def main():
    # Boucle pour capturer des images et détecter les cubes
    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Erreur : Impossible de capturer l'image")
            break

        # Détection du cube et prédiction de la couleur
        predicted_class = detect_cube_and_predict(frame)
        
        # Affichage du résultat sur l'image
        # cv2.putText(frame, f"Cube: {predicted_class}", (50, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Cube: {}".format(predicted_class), (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


        # Affichage en direct
        cv2.imshow("Cube Detection", frame)

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

        # Quitter en appuyant sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Fermer les fenêtres après utilisation
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
