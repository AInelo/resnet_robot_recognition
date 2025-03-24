# -*- coding: utf-8 -*-

import sys
import importlib
import time

sys.path.append('/home/jetson/Dofbot/0.py_install')
Arm_Lib = importlib.import_module('Arm_Lib')
Arm_Device = Arm_Lib.Arm_Device
arm = Arm_Device()

def initialize_arm():
    print("Initialisation du bras")

    # Position de départ
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 180, 250)
    time.sleep(0.5)
    print("Position de départ atteinte")

    # Position intermédiaire
    arm.Arm_serial_servo_write6(93, 87, 89, 2, 89, 178, 250)
    time.sleep(0.5)
    print("Position intermédiaire atteinte")

    

def look_at_platform():
    print("Positionnement du bras pour regarder la plateforme")
    arm.Arm_serial_servo_write(1, 89, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write(2, 95, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write(3, 1, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write(4, 6, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write(5, 89, 500)
    time.sleep(0.5)
    arm.Arm_serial_servo_write(6, 178, 500)
    time.sleep(0.5)
    print("Position mort atteinte")
    time.sleep(2)

def rotate_arm_horizontal(angle_1, angle_5):
    arm.Arm_serial_servo_write(1, angle_1, 500)
    arm.Arm_serial_servo_write(5, angle_5, 500)

def move_arm_vertical(angle_2, angle_3, angle_4):
    arm.Arm_serial_servo_write(2, angle_2, 500)
    arm.Arm_serial_servo_write(3, angle_3, 500)
    arm.Arm_serial_servo_write(4, angle_4, 500)

def operate_gripper(angle_6):
    arm.Arm_serial_servo_write(6, angle_6, 500)

def move_arm_to_cube(cube_position):
    if cube_position:
        x, y = cube_position
        angle_1 = int(x / 640 * 180)
        angle_2 = int(y / 480 * 180)
        
        arm.Arm_serial_servo_write(1, angle_1, 250)
        move_arm_vertical(angle_2, 60, 45)
        time.sleep(0.5)
        
        operate_gripper(30)  # Fermer la pince
        time.sleep(1)

        move_arm_vertical(45, 45, 10)
        time.sleep(1)

def take_cube():
    print("Position prêt à prendre l'objet")
    arm.Arm_serial_servo_write6(93, 82, 14, 32, 89, 178, 500)
    time.sleep(0.5)

    print("Approche de l'objet")
    arm.Arm_serial_servo_write6(94, 64, 11, 43, 89, 101, 500)
    time.sleep(0.5)

    print("Attrape l'objet")
    arm.Arm_serial_servo_write6(94, 32, 52, 20, 89, 153, 500)
    time.sleep(0.5)

    print("Soulève l'objet")
    arm.Arm_serial_servo_write6(93, 111, 2, 28, 89, 153, 500)
    time.sleep(0.5)

    print("Mise en l'air de l'objet")
    arm.Arm_serial_servo_write6(94, 87, 74, 74, 89, 153, 500)
    time.sleep(0.5)


def move_to_yellow_place():
    print("Start deposit on yellow box")
    arm.Arm_serial_servo_write6(54, 91, 46, 60, 89, 157, 500)
    time.sleep(0.5)

    print("Continuation de la position")
    arm.Arm_serial_servo_write6(67, 14, 77, 37, 89, 157, 500)
    time.sleep(0.5)

    print("Relâche l'objet")
    arm.Arm_serial_servo_write6(67, 14, 77, 37, 89, 107, 500)
    time.sleep(0.5)
    
    # Position intermédiaire
    arm.Arm_serial_servo_write6(93, 87, 89, 2, 89, 178, 500)
    time.sleep(0.5)
    print("Position intermédiaire atteinte")



def move_to_green_place():
    print("Start deposit on green box")
    arm.Arm_serial_servo_write6(141, 101, 42, 10, 89, 157, 500)
    time.sleep(0.5)

    print("Continuation de la position")
    arm.Arm_serial_servo_write6(141, 67, 31, 7, 89, 159, 500)
    time.sleep(0.5)

    print("Relâche l'objet")
    arm.Arm_serial_servo_write6(141, 67, 31, 7, 89, 109, 500)
    time.sleep(0.5)

     # Position intermédiaire
    arm.Arm_serial_servo_write6(93, 87, 89, 2, 89, 178, 500)
    time.sleep(0.5)
    print("Position intermédiaire atteinte")


def move_to_red_place():
    print("Start deposit on red box")
    arm.Arm_serial_servo_write6(129, 84, 66, 34, 89, 153, 500)
    time.sleep(0.5)

    print("Evolution de la position")
    arm.Arm_serial_servo_write6(120, 58, 51, 30, 89, 163, 500)
    time.sleep(0.5)

    print("positione bien l'objet")
    arm.Arm_serial_servo_write6(120, 19, 60, 53, 89, 163, 500)
    time.sleep(0.5)

    print("Relâche l'objet")
    arm.Arm_serial_servo_write6(120, 19, 60, 53, 89, 105, 500)
    time.sleep(0.5)

    # Position intermédiaire
    arm.Arm_serial_servo_write6(93, 87, 89, 2, 89, 178, 500)
    time.sleep(0.5)
    print("Position intermédiaire atteinte")

def move_to_blue_place():
    print("Start deposit on blue box")
    arm.Arm_serial_servo_write6(39, 108, 38, 32, 89, 179, 500)
    time.sleep(0.5)

    print("Continuation de la position")
    arm.Arm_serial_servo_write6(47, 83, 9, 21, 89, 179, 500)
    time.sleep(0.5)

    print("Relâche l'objet")
    arm.Arm_serial_servo_write6(47, 83, 9, 21, 89, 109, 500)
    time.sleep(0.5)
    # Position intermédiaire
    arm.Arm_serial_servo_write6(93, 87, 89, 2, 89, 178, 500)
    time.sleep(0.5)
    print("Position intermédiaire atteinte")
