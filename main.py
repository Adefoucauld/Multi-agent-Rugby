# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:27:46 2022

@author: Utilisateur
"""

import enum
import math
import random
import uuid
from enum import Enum

import mesa
import numpy as np
from collections import defaultdict

import mesa.space
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.visualization.ModularVisualization import VisualizationElement, ModularServer, UserSettableParameter
from mesa.visualization.modules import ChartModule

MAX_ITERATION = 100
PROBA_CHGT_ANGLE = 0.01


def move(x, y, speed, angle):
    return x + speed/2 * math.cos(angle), y + speed/2 * math.sin(angle)

# def go_to(x, y, speed, dest_x, dest_y):
#     if np.linalg.norm((x - dest_x, y - dest_y)) < speed:
#         return (dest_x, dest_y), 2 * math.pi * random.random()
#     else:
#         angle = math.acos((dest_x - x)/np.linalg.norm((x - dest_x, y - dest_y)))
#         if dest_y < y:
#             angle = - angle
#         return move(x, y, speed, angle), angle

def go_to(x, y, speed, dest_x, dest_y):
    if np.linalg.norm((x - dest_x, y - dest_y)) < speed:
        return (dest_x, dest_y)
    else:
        next_x = x + (dest_x - x)/(2*speed)
        next_y = y + (dest_y - y)/(2*speed)
        outside_environment = next_x < 0 or next_x > 600 or next_y < 0 or next_y > 600
        if outside_environment:
            return x,y
        else:
            return move(next_x,next_y,speed,np.sign(dest_y-y)*math.pi/2)


class ContinuousCanvas(VisualizationElement):
    local_includes = [
        "./js/simple_continuous_canvas.js",
        "./js/jquery.js"
    ]

    def __init__(self, canvas_height=500,
                 canvas_width=500, instantiate=True):
        VisualizationElement.__init__(self)
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.identifier = "space-canvas"
        if (instantiate):
            new_element = ("new Simple_Continuous_Module({}, {},'{}')".
                           format(self.canvas_width, self.canvas_height, self.identifier))
            self.js_code = "elements.push(" + new_element + ");"

    def portrayal_method(self, obj):
        return obj.portrayal_method()

    def render(self, model):
        representation = defaultdict(list)
        for obj in model.schedule.agents:#represent the agents
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        for obj in model.ball:#represent the ball
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        for obj in model.pitch_cones:#represent the cones
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        return representation

    
class Ball:  # La classe pour le ballon
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def portrayal_method(self):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": "red",
                     "r": 4}
        return portrayal

    
class PitchCones:
    def __init__(self, x,y):
        self.x = x
        self.y = y
        
    def portrayal_method(self):
        portrayal = {"Shape": "arrowHead", "s": 1, "Filled": "true", "Color": "Orange", "Layer": 3, 'x': self.x,
                     'y': self.y, "angle": math.pi/2}
        return portrayal

'''Agents Attacker : subsomption
Priority order : Se replacer (en x et en y), avancer avec balle et passer balle
'''

    
class Attacker(Agent):
    def __init__(self, unique_id: int, model: Model, x, y,speed,angle,r,pass_distance,ball= False):
        super().__init__(unique_id, model)
        self.x = x
        self.y = y
        self.speed = speed
        self.ball = ball
        self.angle = angle
        self.pass_done = False
        self.count_step_ball = 0
        self.pass_distance = pass_distance
        self.r = r
        self.attackers = [player for player in self.model.schedule.agents if player.__class__.__name__=="Attacker"]
        
    def make_pass(self, possible_receivers,x_prev_passeur):
        if self.model.count_pass>0 : #1 pass already done, we continue same sense
            sens_receiver = [player for player in possible_receivers if (self.x-player[0].x)*(x_prev_passeur-self.x) >0] #keep the sens of pass
            if sens_receiver:#reste joueur dans le sens
                closest_attacker = min(sens_receiver, key = lambda x:x[1])[0]
            else:#no more player in the same sens
                closest_attacker = min(possible_receivers, key = lambda x:x[1])[0]#pass to closest player
        else:#first to pass   
            closest_attacker = min(possible_receivers, key = lambda x:x[1])[0]#find to who I am going to pass the ball, choose the closest 
        self.model.ball[-1] = Ball(closest_attacker.x, closest_attacker.y+self.r) #replace position of the ball
        self.model.count_pass +=1 #add one to counter pass
        print("PASS DONE")#to control 
        return closest_attacker, self.x #return abscisse de celui qui a fait passe pour idee sens
    
    def replacement(self):
        carrier = [player for player in self.model.schedule.agents if player.__class__.__name__=="Attacker" and (self.x,self.y)!=(player.x, player.y) and player.ball==True ] #look qui a la balle
        x_ball, y_ball = carrier[0].x, carrier[0].y 
        other_players = [(player,np.linalg.norm(player.x-self.x)) for player in self.model.schedule.agents if player.__class__.__name__=="Attacker" and (self.x,self.y)!=(player.x, player.y)]
        close_mates = sorted(other_players, key = lambda x:x[1])[:2]#2 around me
        if y_ball-50 < self.y :#je suis devant le porteur.y-50 donc je recule et prend la profondeur 
            if self.y - self.speed*math.sin(- math.pi/2) <5 :#je verifie à pas sortir du terrain
                self.x,_  = move(self.x, self.y, self.speed, self.angle)#je bouge pas et j'attends que le porteur me dépasse
            else:
                _, self.y = move(self.x, self.y, self.speed*2, -self.angle) #je recule
        if abs(close_mates[0][0].x-x_ball)<abs(self.x-x_ball): #si un des mates est plus proche du porteur que je le suis
            _,self.y = go_to(self.x, self.y, self.speed, x_ball, close_mates[0][0].y -10) #je me mets en profondeur de lui 
        else:
            _,self.y =go_to(self.x, self.y, self.speed, self.x, close_mates[1][0].y -10) #je me mets en profondeur de lui 
        ##Following is for the pass_distance, not for now
        if abs(abs(close_mates[0][0].x - self.x) - self.pass_distance)<5 and close_mates[0][0].ball: #trop éloigné du porteur
            self.x,_ = go_to(self.x, self.y, self.speed,x_ball + self.pass_distance * (self.x-close_mates[0][0].x)/abs(close_mates[0][0].x-self.x+1) , self.y)
        elif abs(abs(close_mates[1][0].x - self.x) - self.pass_distance)<5 and close_mates[1][0].ball:
            self.x,_ = go_to(self.x, self.y, self.speed,x_ball + self.pass_distance * (self.x-close_mates[0][0].x)/abs(close_mates[0][0].x-self.x+1) , self.y)
        
        # elif abs(abs(close_mates[0][0].x - self.x) > self.pass_distance and close_mates[0][0].ball: #trop loin du porteur
        #     self.x,_ = go_to(self.x, self.y, self.speed,x_ball + self.pass_distance * (self.x-close_mates[0][0].x)/abs(close_mates[0][0].x-self.x+1), self.y)
        # elif abs(close_mates[1][0].x - self.x) > self.pass_distance and close_mates[1][0].ball:
        #     self.x,_ = go_to(self.x, self.y, self.speed,x_ball + self.pass_distance * (self.x-close_mates[0][0].x)/abs(close_mates[0][0].x-self.x+1) , self.y)
        
        # elif np.abs(x_ball-self.x)>self.pass_distance: #je suis derriere le porteur et mais trop loin pour une passe
        #     _,self.y  = move(self.x, self.y, self.speed, self.angle)
        #     self.x = self.x + (x_ball-self.x)/self.speed #je me rapproche de la balle 
        return self.x, self.y
        
    def step(self):
        
        '''1st priority: replacement if not ball '''
        if (self.model.ball[-1].x, self.model.ball[-1].y)!=(self.x,self.y+self.r): #si j'ai la balle
           self.x, self.y = self.replacement()
        
        '''2nd priority: avancer avec la balle tant que ya pas deux joueurs en jeu   '''
        
        if (self.model.ball[-1].x, self.model.ball[-1].y)==(self.x,self.y+self.r): #si j'ai la balle
            self.ball = True #bool to indicate I have the ball
            possible_receivers = [(attacker,np.linalg.norm((attacker.x - self.x, attacker.y - self.y))) for attacker in self.model.schedule.agents if attacker.__class__.__name__=="Attacker" and (self.x, self.y) != (attacker.x, attacker.y) and attacker.y < self.y ] #forced to pass behind
            if (self.y < self.model.pitch_cones[-1].y -self.speed*math.sin(self.angle)) and (self.count_step_ball < 20) : #on avance jusqu'aux plots avant de faire une passe ou on fait 10 steps avec la balle
                _, self.y = move(self.x, self.y, self.speed, self.angle) #j'avance en ligne droite au ralenti pour laisser temps de replacement  
                self.model.ball[-1] = Ball(self.x, self.y+self.r) #la balle nous suit
                self.count_step_ball+=1 #counter de pass avec la balle
            else:#je fais une passe
                if self.model.count_pass==0:#premier à passer, je choisis sens aleatoire
                    receiver, self.model.x_passeur = self.make_pass(possible_receivers,self.x+random.uniform(-1, 1)*0.1+0.01)#random.uniform(-1, 1)*0.1+0.01 : add small relatif integer to choose sens of pass
                else:
                    receiver, self.model.x_passeur = self.make_pass(possible_receivers,self.model.x_passeur)
                receiver.ball = True
                self.ball = False #j'ai fais la passe, je n'ai plus le ballon
                self.count_step_ball=0 #je remets à zero mes pas avec la balle
        
        '''1st scenario : WAVES OF PASS
        Player make 10 steps or reaches the cones before passing the ball. Keep the sense of pass
        When ball reaches end of lines, come back
        '''

    def portrayal_method(self):
        if self.ball:
            portrayal = {"Shape": "circle",
                         "Filled": "true",
                         "Layer": 2,
                         "Color": "yellow",
                         "r": self.r}
        else:
            portrayal = {"Shape": "circle",
                         "Filled": "true",
                         "Layer": 2,
                         "Color": "blue",
                         "r": self.r}
        return portrayal

class Nine(Attacker):
    def __init__(self,unique_id, model, x, y,speed,angle,r,pass_distance,ball= False):
        super().__init__(unique_id, model, x, y,speed,angle,r,pass_distance,ball= False)

    def portrayal_method(self):
        if self.ball:
            portrayal = {"Shape": "triangle",
                         "Filled": "true",
                         "Layer": 2,
                         "Color": "yellow",
                         "r": self.r}
        else:
            portrayal = {"Shape": "triangle",
                         "Filled": "false",
                         "Layer": 2,
                         "Color": "blue",
                         "r": self.r}
        return portrayal

class Defender(Agent):
    def __init__(self, unique_id: int, model: Model, x, y,speed,angle,r):
        super().__init__(unique_id, model)
        self.x = x
        self.y = y
        self.speed = speed
        self.angle = angle
        self.r = r
        self.set_ligne = False
    
    def step(self):
        '''Priority order; Se placer en ligne 5 pts derriere les plots, monter, se diriger vers porteur, plaquer'''
        if self.set_ligne==False:
            print(f'y :{self.y}')
            if abs(self.y - 560)>10:
                _,self.y = go_to(self.x, self.y, self.speed, self.x, 560)
            else:
                self.set_ligne = True
            
        ##On monte si on est tous sur la meme ligne
        defenders = [player for player in self.model.schedule.agents if player.__class__.__name__ =="Defender" and player.set_ligne ==True]
        if len(defenders)== len([player for player in self.model.schedule.agents if player.__class__.__name__ =="Defender"]): #tous le monde est en ligne
            self.y -= self.speed/4
        
    def portrayal_method(self):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 2,
                     "Color": "red",
                     "r": self.r}
        return portrayal
    
class RugbyPitch(Model):
    collector = DataCollector(
        model_reporters={"Nb_passes": lambda model: model.count_pass,
                          },
        agent_reporters={})

    def __init__(self,n_attackers,n_defenders,speed, pass_distance):
        Model.__init__(self)
        self.space = mesa.space.ContinuousSpace(600, 600, False)
        self.schedule = RandomActivation(self)
        self.ball=[]
        self.pitch_cones = []
        self.count_pass = 0 #counter for nb of passes
        self.x_passeur= 0
        self.r_player = 6
        self.pass_distance = pass_distance 
        
        for _ in range(n_attackers-1):
            x, y = random.random() * 500, random.randint(0,200) 
            self.schedule.add(Attacker(int(uuid.uuid1()), self, x, y,speed,math.pi/2, self.r_player,self.pass_distance ))
        x_ball, y_ball = random.random() * 500, random.randint(0,200)
        self.schedule.add(Attacker(int(uuid.uuid1()), self,x_ball , y_ball,speed,math.pi/2, self.r_player,self.pass_distance, ball = True))
        self.ball.append(Ball(x_ball, y_ball+self.r_player)) #add a ball for last attacker introduced, +4 to be displayed ahead of the player
        
        for _ in range(n_defenders):
            x, y = random.random() * 500, random.randint(450,600) 
            self.schedule.add(Defender(int(uuid.uuid1()), self, x, y,speed,math.pi/2, self.r_player))
        
        
        for x in range(1,601,50):
            self.pitch_cones.append(PitchCones(x, 450))#trace ligne de cones
            
        
        self.datacollector = self.collector

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if self.schedule.steps >= 1000:
            self.running = False
        

def run_single_server():
    chart = ChartModule([{"Label": "Nb_passes",
                          "Color": "blue"},
                         ],
                        data_collector_name='datacollector')
    server = ModularServer(RugbyPitch,
                           [ContinuousCanvas(),
                            chart],
                           "Play Rugby",
                           {"n_attackers": UserSettableParameter('slider', "Nb attackers", 4, 2,
                                                             15, 1),
                            "n_defenders": UserSettableParameter('slider', "Nb defenders", 3, 2,
                                                              15, 1),
                            "speed": UserSettableParameter('slider', "player speed", 10, 1, 50, 2),
                            "pass_distance": UserSettableParameter('slider', "Pass distance", 50, 10, 300, 10),
                            
                           })
    server.port = 8521
    server.launch()


if __name__ == "__main__":
    run_single_server()
