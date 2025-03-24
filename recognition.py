import torch
import cv2
import subprocess
import time
from PIL import Image
import numpy as np
from torchvision import models, transforms
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

class CubeClassifier:
    def __init__(self, model_path='./test-save/resnet_model_best.pth', num_classes=4, device=None):
        # Initialisation du modèle
        self.model = models.resnet18(pretrained=False)  # Ne pas charger les poids préentraînés
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)  # Adapter la couche finale pour 4 classes

        # Charger les poids du modèle
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint.state_dict())

        # Passer le modèle en mode évaluation
        self.model.eval()

        # Déplacer le modèle sur le GPU s'il est disponible
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Transformation pour les images de la caméra
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Liste des classes
        self.class_names = ['cube_blue', 'cube_green', 'cube_red', 'cube_yellow']

    def predict(self, frame):
        """Détecte la couleur et fait la prédiction avec le modèle."""
        # Convertir l'image BGR en RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Appliquer la transformation pour correspondre à l'entraînement
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)  # Ajouter une dimension pour le batch et déplacer vers GPU

        # Prédiction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted_class = torch.max(outputs, 1)

        predicted_class_name = self.class_names[predicted_class.item()]
        return predicted_class_name


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
        """Capture une image à partir de la caméra."""
        success, image = self.video.read()
        return image if success else None

    def __del__(self):
        """Libère la caméra."""
        self.video.release()


def run_cube_detection():
    """Méthode pour tester la détection des cubes."""
    # Initialisation du classificateur et de la caméra
    model_path = './test-save/resnet_model_best.pth'
    classifier = CubeClassifier(model_path=model_path)
    camera = VideoCamera()

    while True:
        # Capture d'une frame de la caméra
        frame = camera.get_frame()
        if frame is None:
            print("Erreur : Impossible de capturer l'image")
            break

        # Prédiction du cube
        predicted_class = classifier.predict(frame)

        # Affichage du résultat sur l'image
        cv2.putText(frame, "Cube: {}".format(predicted_class), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Affichage en direct
        cv2.imshow("Cube Detection", frame)

        # Quitter en appuyant sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



def detect_cube_with_picture(image_path):
    """Détecte la classe du cube à partir d'une image."""
    # Initialisation du classificateur
    model_path = './test-save/resnet_model_best.pth'
    classifier = CubeClassifier(model_path=model_path)

    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur : Impossible de charger l'image")
        return

    # Prédiction du cube
    predicted_class = classifier.predict(image)

    # Affichage du résultat
    print(f"Classe prédite : {predicted_class}")



def detect_cube_picture_camera():
    """Détecte la classe du cube à partir de la caméra avec une seule photo."""
    # Initialisation du classificateur et de la caméra
    model_path = './test-save/resnet_model_best.pth'
    classifier = CubeClassifier(model_path=model_path)
    camera = VideoCamera()

    # Capture d'une frame de la caméra
    frame = camera.get_frame()
    if frame is None:
        print("Erreur : Impossible de capturer l'image")
        return

    # Prédiction du cube
    predicted_class = classifier.predict(frame)

    # Affichage du résultat
    print(f"Classe prédite : {predicted_class}")


# def detect_cube_accurate():
#     """Détecte la classe du cube en lançant la détection 5 fois et en prenant la classe la plus fréquente."""
#     model_path = './test-save/resnet_model_best.pth'
#     classifier = CubeClassifier(model_path=model_path)
#     camera = VideoCamera()

#     # Initialisation des comptages pour chaque classe
#     class_counts = {class_name: 0 for class_name in classifier.class_names}

#     # Essayer 5 fois de détecter le cube
#     for _ in range(2):
#         frame = camera.get_frame()
#         if frame is None:
#             print("Erreur : Impossible de capturer l'image")
#             return None  # Retourner None si l'image ne peut pas être capturée

#         predicted_class = classifier.predict(frame)
        
#         # Vérification si la prédiction est valide
#         if predicted_class in class_counts:
#             class_counts[predicted_class] += 1
#         else:
#             print(f"Prédiction invalide : {predicted_class}")

#     # Trouver la classe la plus fréquente
#     most_frequent_class = max(class_counts, key=class_counts.get)

#     # Afficher la classe la plus fréquente
#     print(f"Classe prédite la plus fréquente : {most_frequent_class}")

#     return most_frequent_class  # Retourner la classe la plus fréquente


def detect_cube_accurate(classifier, camera):
    """Détecte la classe du cube en lançant la détection 2 fois et en prenant la classe la plus fréquente."""

    # Initialisation des comptages pour chaque classe
    class_counts = {class_name: 0 for class_name in classifier.class_names}

    # Essayer 2 fois de détecter le cube
    for _ in range(2):
        frame = camera.get_frame()
        if frame is None:
            print("Erreur : Impossible de capturer l'image")
            return None  # Retourner None si l'image ne peut pas être capturée

        predicted_class = classifier.predict(frame)
        
        # Vérification si la prédiction est valide
        if predicted_class in class_counts:
            class_counts[predicted_class] += 1
        else:
            print(f"Prédiction invalide : {predicted_class}")

    # Trouver la classe la plus fréquente
    most_frequent_class = max(class_counts, key=class_counts.get)

    print(f"Classe prédite la plus fréquente : {most_frequent_class}")

    return most_frequent_class  # Retourner la classe la plus fréquente




# if __name__ == '__main__':

    # detect_cube_picture_camera()
    # detect_cube_accurate()