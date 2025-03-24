# # -*- coding: utf-8 -*-


# import torch
# import cv2
# import numpy as np
# from torchvision import models, transforms
# from PIL import Image
# import subprocess

# # Charger le modèle complet
# # model = models.resnet18(pretrained=False)  # Créer une nouvelle instance de ResNet18
# # model.load_state_dict(torch.load('./test-save/resnet_model_best.pth'))  # Charger les poids
# # model.eval()


# # Charger le modèle complet
# model = models.resnet18(pretrained=False)  # Créer une nouvelle instance de ResNet18

# # Adapter la dernière couche pour 4 classes
# num_features = model.fc.in_features
# model.fc = torch.nn.Linear(num_features, 4)  # 4 classes : cube_red, cube_green, cube_blue, cube_yellow

# # Charger les poids
# model.load_state_dict(torch.load('./test-save/resnet_model_best.pth'))  # Charger les poids
# model.eval()



# # Transformation pour les images de la caméra
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# class VideoCamera:
#     def __init__(self):
#         self.release_camera_if_used()
#         self.video = cv2.VideoCapture(0)
#         self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
#         self.video.set(cv2.CAP_PROP_BRIGHTNESS, 30)
#         self.video.set(cv2.CAP_PROP_CONTRAST, 50)
#         self.video.set(cv2.CAP_PROP_EXPOSURE, 156)
#         self.video.set(3, 640)
#         self.video.set(4, 480)
#         self.video.set(5, 30)

#     def release_camera_if_used(self):
#         """Libère la caméra si un autre processus l'utilise."""
#         try:
#             result = subprocess.check_output("sudo fuser /dev/video0", shell=True).decode().strip()
#             if result:
#                 pid = result.split()[-1]
#                 # subprocess.run(f"sudo kill -9 {pid}", shell=True, check=True)
#                 subprocess.run("sudo kill -9 {}".format(pid), shell=True, check=True)
#         except subprocess.CalledProcessError:
#             pass

#     def get_frame(self):
#         success, image = self.video.read()
#         return image if success else None

#     def __del__(self):
#         self.video.release()

# def detect_cube_and_predict(frame):
#     """Détecte la couleur et fait la prédiction avec le modèle."""
#     # Convertir l'image BGR en RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_image = Image.fromarray(frame_rgb)

#     # Appliquer la transformation pour correspondre à l'entraînement
#     input_tensor = transform(pil_image).unsqueeze(0)  # Ajouter une dimension pour le batch

#     # Prédiction
#     with torch.no_grad():
#         outputs = model(input_tensor)
#         _, predicted_class = torch.max(outputs, 1)

#     # Liste des classes du modèle
#     class_names = ['cube_red', 'cube_green', 'cube_blue', 'cube_yellow']
#     predicted_class_name = class_names[predicted_class.item()]

#     return predicted_class_name

# # Initialisation de la caméra du robot
# camera = VideoCamera()

# while True:
#     frame = camera.get_frame()
#     if frame is None:
#         print("Erreur : Impossible de capturer l'image")
#         break

#     # Détection et prédiction du cube
#     predicted_class = detect_cube_and_predict(frame)

#     # Affichage du résultat sur l'image
#     # cv2.putText(frame, f"Cube: {predicted_class}", (50, 50),
#     #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#     cv2.putText(frame, "Cube: {}".format(predicted_class), (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#     # Affichage en direct
#     cv2.imshow("Cube Detection", frame)

#     # Quitter en appuyant sur 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()



# -*- coding: utf-8 -*-

import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import subprocess
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# Charger le modèle complet
# model = models.resnet18(pretrained=False)  # Créer une nouvelle instance de ResNet18

# # Adapter la dernière couche pour 4 classes
# num_features = model.fc.in_features
# model.fc = torch.nn.Linear(num_features, 4)  # 4 classes : cube_red, cube_green, cube_blue, cube_yellow

# # Charger les poids
# model.load_state_dict(torch.load('./test-save/resnet_model_best.pth'))  # Charger les poids
# model.eval()





# # Charger un modèle ResNet18 pré-entraîné
# model = models.resnet18(pretrained=True)

# # Adapter la dernière couche pour 4 classes
# num_features = model.fc.in_features
# model.fc = torch.nn.Linear(num_features, 4)  # 4 classes : cube_red, cube_green, cube_blue, cube_yellow

# # Charger les poids
# try:
#     # model.load_state_dict(torch.load('./test-save/resnet_model_best.pth', map_location=device))  # Charger les poids
#     model.load_state_dict(torch.load('./test-save/resnet_model_best.pth', map_location=torch.device('cpu')))

# except Exception as e:
#     print(f"Erreur lors du chargement des poids : {e}")

# model.eval()


# Créer une instance du modèle
model = models.resnet18(pretrained=False)  # S'assurer que les poids pré-entraînés ne sont pas chargés
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # Adapter la couche finale pour 4 classes

# Charger les poids
checkpoint = torch.load('./test-save/resnet_model_best.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint.state_dict())  # Utiliser state_dict() si nécessaire

model.eval()  # Passer le modèle en mode évaluation

# Déplacer le modèle sur le GPU s'il est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Transformation pour les images de la caméra
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class VideoCamera:
    def __init__(self):
        self.release_camera_if_used()
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.video.set(cv2.CAP_PROP_BRIGHTNESS, 30)
        self.video.set(cv2.CAP_PROP_CONTRAST, 50)
        self.video.set(cv2.CAP_PROP_EXPOSURE, 156)
        self.video.set(3, 640)
        self.video.set(4, 480)
        self.video.set(5, 30)

    def release_camera_if_used(self):
        """Libère la caméra si un autre processus l'utilise."""
        try:
            result = subprocess.check_output("sudo fuser /dev/video0", shell=True).decode().strip()
            if result:
                pid = result.split()[-1]
                subprocess.run("sudo kill -9 {}".format(pid), shell=True, check=True)
        except subprocess.CalledProcessError:
            pass

    def get_frame(self):
        success, image = self.video.read()
        return image if success else None

    def __del__(self):
        self.video.release()

def detect_cube_and_predict(frame):
    """Détecte la couleur et fait la prédiction avec le modèle."""
    # Convertir l'image BGR en RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Appliquer la transformation pour correspondre à l'entraînement
    input_tensor = transform(pil_image).unsqueeze(0).to(device)  # Ajouter une dimension pour le batch et déplacer vers GPU

    # Prédiction
    start_time = time.time()  # Capture du temps avant la prédiction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)
    end_time = time.time()  # Capture du temps après la prédiction

    # Affichage du temps de prédiction
    print(f"Temps de prédiction : {end_time - start_time:.4f} secondes")

    # Liste des classes du modèle
    class_names = ['cube_red', 'cube_green', 'cube_blue', 'cube_yellow']
    predicted_class_name = class_names[predicted_class.item()]

    return predicted_class_name

# Initialisation de la caméra du robot
camera = VideoCamera()

while True:
    frame = camera.get_frame()
    if frame is None:
        print("Erreur : Impossible de capturer l'image")
        break

    # Détection et prédiction du cube
    predicted_class = detect_cube_and_predict(frame)

    # Affichage du résultat sur l'image
    cv2.putText(frame, "Cube: {}".format(predicted_class), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Affichage en direct
    cv2.imshow("Cube Detection", frame)

    # Quitter en appuyant sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
