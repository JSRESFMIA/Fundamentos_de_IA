#================================================================
# Game Vibora Inteligente
#================================================================
# Serrano Ramirez Javier
# Fundamentos de Inteligencia Artificial
# IPN ESFM
# Junio 2025
#================================================================
# Importamos las librerías necesarias
import pygame
import random
from enum import Enum  # Para usar enumeraciones (valores constantes)
from collections import namedtuple  # Para crear objetos similares a tuplas

# Inicializamos pygame y la fuente que se usará para mostrar el puntaje
pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# Definimos las direcciones posibles como una enumeración
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Creamos una estructura llamada Point que contiene una coordenada x y y
Point = namedtuple('Point', 'x, y')

# Definimos los colores en formato RGB
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Tamaño del bloque de la serpiente y la comida, y velocidad del juego
BLOCK_SIZE = 20
SPEED = 20

# Clase principal del juego
class SnakeGame:
    
    def __init__(self, w=640, h=480):
        # Ancho y alto de la ventana
        self.w = w
        self.h = h
        # Inicializa la pantalla del juego
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')  # Título de la ventana
        self.clock = pygame.time.Clock()  # Control de la velocidad del juego
        
        # Estado inicial del juego
        self.direction = Direction.RIGHT  # Comienza moviéndose a la derecha
        
        # Coordenadas iniciales de la cabeza de la víbora
        self.head = Point(self.w/2, self.h/2)
        # Cuerpo inicial de la víbora (3 bloques)
        self.snake = [
            self.head, 
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        
        self.score = 0  # Puntaje inicial
        self.food = None
        self._place_food()  # Coloca la primera comida

    # Método privado para colocar la comida aleatoriamente en la pantalla
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        # Si la comida aparece donde está la víbora, se vuelve a colocar
        if self.food in self.snake:
            self._place_food()

    # Método principal que ejecuta un paso del juego
    def play_step(self):
        # 1. Leer eventos del teclado
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # Cambiar dirección con las flechas
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        # 2. Mover la víbora
        self._move(self.direction)
        self.snake.insert(0, self.head)  # Agrega la nueva cabeza al cuerpo

        # 3. Verificar colisiones (muros o a sí misma)
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # 4. Verificar si se comió la comida
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()  # Si no comió, eliminar la última parte del cuerpo

        # 5. Dibujar todo en pantalla y controlar velocidad
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Regresar si el juego terminó y el puntaje
        return game_over, self.score

    # Verifica si hubo una colisión
    def _is_collision(self):
        # Si choca contra los bordes
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or \
           self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # Si se choca contra sí misma
        if self.head in self.snake[1:]:
            return True
        return False

    # Dibuja la serpiente, la comida y el puntaje
    def _update_ui(self):
        self.display.fill(BLACK)  # Fondo negro
        # Dibujar cada parte del cuerpo de la víbora
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        # Dibujar la comida
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # Mostrar el puntaje
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()  # Actualiza la pantalla

    # Cambia la posición de la cabeza dependiendo de la dirección
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

# Ejecución del juego
if __name__ == '__main__':
    game = SnakeGame()
    
    # Ciclo principal del juego
    while True:
        game_over, score = game.play_step()
        if game_over:
            break  # Si el juego terminó, se sale del bucle

    print('Puntaje Final:', score)
    pygame.quit()  # Cierra el juego


