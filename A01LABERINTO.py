#================================================================
# LABERINTO
#================================================================
# Serrano Ramirez Javier
# Fundamentos de Inteligencia Artificial 
# IPN ESFM
# Junio 2025
#================================================================
import random, datetime, csv, os
from tkinter import *
from enum import Enum
from collections import deque

class COLOR(Enum):
    '''
    Clase para definir colores de fondo y de agentes en Tkinter.
    '''
    dark = ('gray11', 'white')
    light = ('white', 'black')
    black = ('black', 'dim gray')
    red = ('red3', 'tomato')
    cyan = ('cyan4', 'cyan4')
    green = ('green4', 'pale green')
    blue = ('DeepSkyBlue4', 'DeepSkyBlue2')
    yellow = ('yellow2', 'yellow2')

class agent:
    '''
    Clase que representa un agente en el laberinto. Puede ser cuadrado o flecha.
    '''
    def __init__(self, parentMaze, x=None, y=None, shape='square', goal=None, filled=False, footprints=False, color: COLOR = COLOR.blue):
        self._parentMaze = parentMaze
        self.color = color
        if isinstance(color, str):
            if color in COLOR.__members__:
                self.color = COLOR[color]
            else:
                raise ValueError(f'{color} no es un color válido!')
        self.filled = filled
        self.shape = shape
        self._orient = 0
        if x is None: x = parentMaze.rows
        if y is None: y = parentMaze.cols
        self.x = x
        self.y = y
        self.footprints = footprints
        self._parentMaze._agents.append(self)
        self.goal = goal if goal else self._parentMaze._goal
        self._body = []
        self.position = (self.x, self.y)
    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, newX):
        self._x = newX

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, newY):
        self._y = newY
        w = self._parentMaze._cell_width
        x = self.x * w - w + self._parentMaze._LabWidth
        y = self.y * w - w + self._parentMaze._LabWidth
        if self.shape == 'square':
            if self.filled:
                self._coord = (y, x, y + w, x + w)
            else:
                self._coord = (y + w / 2.5, x + w / 2.5, y + w / 2.5 + w / 4, x + w / 2.5 + w / 4)
        else:
            self._coord = (y + w / 2, x + 3 * w / 9, y + w / 2, x + 3 * w / 9 + w / 4)

        if hasattr(self, '_head'):
            if not self.footprints:
                self._parentMaze._canvas.delete(self._head)
            else:
                if self.shape == 'square':
                    self._parentMaze._canvas.itemconfig(self._head, fill=self.color.value[1], outline="")
                    self._parentMaze._canvas.tag_raise(self._head)
                    try:
                        self._parentMaze._canvas.tag_lower(self._head, 'ov')
                    except:
                        pass
                    if self.filled:
                        lll = self._parentMaze._canvas.coords(self._head)
                        oldcell = (
                            round(((lll[1] - 26) / w) + 1),
                            round(((lll[0] - 26) / w) + 1)
                        )
                        self._parentMaze._redrawCell(*oldcell, self._parentMaze.theme)
                else:
                    self._parentMaze._canvas.itemconfig(self._head, fill=self.color.value[1])
                    self._parentMaze._canvas.tag_raise(self._head)
                    try:
                        self._parentMaze._canvas.tag_lower(self._head, 'ov')
                    except:
                        pass
                self._body.append(self._head)

            if not self.filled or self.shape == 'arrow':
                if self.shape == 'square':
                    self._head = self._parentMaze._canvas.create_rectangle(
                        *self._coord, fill=self.color.value[0], outline=''
                    )
                    try:
                        self._parentMaze._canvas.tag_lower(self._head, 'ov')
                    except:
                        pass
                else:
                    self._head = self._parentMaze._canvas.create_line(
                        *self._coord, fill=self.color.value[0],
                        arrow=FIRST, arrowshape=(3 / 10 * w, 4 / 10 * w, 4 / 10 * w)
                    )
                    try:
                        self._parentMaze._canvas.tag_lower(self._head, 'ov')
                    except:
                        pass
                    o = self._orient % 4
                    if o == 1:
                        self._RCW(); self._orient -= 1
                    elif o == 3:
                        self._RCCW(); self._orient += 1
                    elif o == 2:
                        self._RCCW(); self._RCCW(); self._orient += 2
            else:
                self._head = self._parentMaze._canvas.create_rectangle(
                    *self._coord, fill=self.color.value[0], outline=''
                )
                try:
                    self._parentMaze._canvas.tag_lower(self._head, 'ov')
                except:
                    pass
                self._parentMaze._redrawCell(self.x, self.y, theme=self._parentMaze.theme)
        else:
            self._head = self._parentMaze._canvas.create_rectangle(
                *self._coord, fill=self.color.value[0], outline=''
            )
            try:
                self._parentMaze._canvas.tag_lower(self._head, 'ov')
            except:
                pass
            self._parentMaze._redrawCell(self.x, self.y, theme=self._parentMaze.theme)

    @property
    def position(self):
        return (self.x, self.y)

    @position.setter
    def position(self, newpos):
        self.x, self.y = newpos
        self._position = newpos
    def _RCCW(self):
        '''
        Gira el agente en sentido antihorario.
        '''
        def pointNew(p, origin):
            return (p[0] - origin[0], p[1] - origin[1])

        w = self._parentMaze._cell_width
        x = self.x * w - w + self._parentMaze._LabWidth
        y = self.y * w - w + self._parentMaze._LabWidth
        center = (y + w / 2, x + w / 2)

        p1 = pointNew((self._coord[0], self._coord[1]), center)
        p2 = pointNew((self._coord[2], self._coord[3]), center)

        p1_rot = (p1[1], -p1[0])
        p2_rot = (p2[1], -p2[0])

        p1 = p1_rot[0] + center[0], p1_rot[1] + center[1]
        p2 = p2_rot[0] + center[0], p2_rot[1] + center[1]

        self._coord = (*p1, *p2)
        self._parentMaze._canvas.coords(self._head, *self._coord)
        self._orient = (self._orient - 1) % 4

    def _RCW(self):
        '''
        Gira el agente en sentido horario.
        '''
        def pointNew(p, origin):
            return (p[0] - origin[0], p[1] - origin[1])

        w = self._parentMaze._cell_width
        x = self.x * w - w + self._parentMaze._LabWidth
        y = self.y * w - w + self._parentMaze._LabWidth
        center = (y + w / 2, x + w / 2)

        p1 = pointNew((self._coord[0], self._coord[1]), center)
        p2 = pointNew((self._coord[2], self._coord[3]), center)

        p1_rot = (-p1[1], p1[0])
        p2_rot = (-p2[1], p2[0])

        p1 = p1_rot[0] + center[0], p1_rot[1] + center[1]
        p2 = p2_rot[0] + center[0], p2_rot[1] + center[1]

        self._coord = (*p1, *p2)
        self._parentMaze._canvas.coords(self._head, *self._coord)
        self._orient = (self._orient + 1) % 4

    # Métodos para mover al agente con teclas
    def moveRight(self, event):
        if self._parentMaze.maze_map[self.x, self.y]['E']:
            self.y += 1

    def moveLeft(self, event):
        if self._parentMaze.maze_map[self.x, self.y]['W']:
            self.y -= 1

    def moveUp(self, event):
        if self._parentMaze.maze_map[self.x, self.y]['N']:
            self.x -= 1

    def moveDown(self, event):
        if self._parentMaze.maze_map[self.x, self.y]['S']:
            self.x += 1
class textLabel:
    '''
    Clase para crear etiquetas de texto que muestran resultados sobre la ventana.
    '''
    def __init__(self, parentMaze, title, value):
        self.title = title
        self._value = value
        self._parentMaze = parentMaze
        self._var = None
        self.drawLabel()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v
        self._var.set(f'{self.title} : {v}')

    def drawLabel(self):
        self._var = StringVar()
        self.lab = Label(
            self._parentMaze._canvas,
            textvariable=self._var,
            bg="white",
            fg="black",
            font=('Helvetica bold', 12),
            relief=RIDGE
        )
        self._var.set(f'{self.title} : {self.value}')
        self.lab.pack(expand=True, side=LEFT, anchor=NW)


class maze:
    '''
    Clase principal para crear y visualizar un laberinto.
    '''
    def __init__(self, rows=10, cols=10):
        '''
        rows: número de filas del laberinto.
        cols: número de columnas del laberinto.
        '''
        self.rows = rows
        self.cols = cols
        self.maze_map = {}
        self.grid = []
        self.path = {}
        self._cell_width = 50
        self._win = None
        self._canvas = None
        self._agents = []
        self.markCells = []

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, n):
        '''
        Inicializa la estructura de celdas y bloquea todas las paredes por defecto.
        '''
        self._grid = []
        y = 0
        for n in range(self.cols):
            x = 1
            y = 1 + y
            for m in range(self.rows):
                self.grid.append((x, y))
                self.maze_map[x, y] = {'E': 0, 'W': 0, 'N': 0, 'S': 0}
                x = x + 1
    def _Open_East(self, x, y):
        '''
        Abre la pared Este de la celda (x, y).
        '''
        self.maze_map[x, y]['E'] = 1
        if y + 1 <= self.cols:
            self.maze_map[x, y + 1]['W'] = 1

    def _Open_West(self, x, y):
        '''
        Abre la pared Oeste de la celda (x, y).
        '''
        self.maze_map[x, y]['W'] = 1
        if y - 1 > 0:
            self.maze_map[x, y - 1]['E'] = 1

    def _Open_North(self, x, y):
        '''
        Abre la pared Norte de la celda (x, y).
        '''
        self.maze_map[x, y]['N'] = 1
        if x - 1 > 0:
            self.maze_map[x - 1, y]['S'] = 1

    def _Open_South(self, x, y):
        '''
        Abre la pared Sur de la celda (x, y).
        '''
        self.maze_map[x, y]['S'] = 1
        if x + 1 <= self.rows:
            self.maze_map[x + 1, y]['N'] = 1

    def CreateMaze(self, x=1, y=1, pattern=None, loopPercent=0, saveMaze=False, loadMaze=None, theme: COLOR = COLOR.dark):
        '''
        Crea un laberinto nuevo o carga uno desde archivo CSV.

        Parámetros:
        - x, y: coordenadas de inicio.
        - pattern: 'h' o 'v' para sesgo horizontal o vertical.
        - loopPercent: porcentaje de caminos adicionales (0 = laberinto perfecto).
        - saveMaze: guarda el laberinto generado en archivo CSV.
        - loadMaze: nombre del archivo CSV a cargar (si se desea cargar).
        - theme: tema de colores.
        '''
        _stack = []
        _closed = []
        self.theme = theme
        self._goal = (x, y)

        if isinstance(theme, str):
            if theme in COLOR.__members__:
                self.theme = COLOR[theme]
            else:
                raise ValueError(f'{theme} no es un COLOR válido.')

        def blockedNeighbours(cell):
            '''
            Retorna una lista de vecinos bloqueados de la celda.
            '''
            n = []
            for d in self.maze_map[cell]:
                if self.maze_map[cell][d] == 0:
                    if d == 'E' and (cell[0], cell[1] + 1) in self.grid:
                        n.append((cell[0], cell[1] + 1))
                    elif d == 'W' and (cell[0], cell[1] - 1) in self.grid:
                        n.append((cell[0], cell[1] - 1))
                    elif d == 'N' and (cell[0] - 1, cell[1]) in self.grid:
                        n.append((cell[0] - 1, cell[1]))
                    elif d == 'S' and (cell[0] + 1, cell[1]) in self.grid:
                        n.append((cell[0] + 1, cell[1]))
            return n
        def removeWallinBetween(cell1, cell2):
            '''
            Quita la pared entre dos celdas adyacentes.
            '''
            if cell1[0] == cell2[0]:
                if cell1[1] > cell2[1]: cell1, cell2 = cell2, cell1
                self.maze_map[cell1]['E'] = 1
                self.maze_map[cell2]['W'] = 1
            else:
                if cell1[0] > cell2[0]: cell1, cell2 = cell2, cell1
                self.maze_map[cell1]['S'] = 1
                self.maze_map[cell2]['N'] = 1

        def BFS(cell):
            '''
            Búsqueda en anchura para hallar el camino más corto hacia el objetivo.
            '''
            frontier = deque([cell])
            path = {}
            visited = { (self.rows, self.cols) }

            while frontier:
                c = frontier.popleft()
                for dir, dx, dy in [('W', 0, -1), ('S', 1, 0), ('E', 0, 1), ('N', -1, 0)]:
                    if self.maze_map[c][dir]:
                        nc = (c[0] + dx, c[1] + dy)
                        if nc not in visited:
                            visited.add(nc)
                            frontier.append(nc)
                            path[nc] = c

            fwdPath = {}
            c = self._goal
            while c != (self.rows, self.cols):
                try:
                    fwdPath[path[c]] = c
                    c = path[c]
                except:
                    print('¡Camino no encontrado!')
                    return
            return fwdPath

        if not loadMaze:
            # Generación de laberinto aleatorio
            _stack.append((x, y))
            _closed.append((x, y))

            while _stack:
                cell = []
                if (x, y + 1) not in _closed and (x, y + 1) in self.grid:
                    cell.append("E")
                if (x, y - 1) not in _closed and (x, y - 1) in self.grid:
                    cell.append("W")
                if (x + 1, y) not in _closed and (x + 1, y) in self.grid:
                    cell.append("S")
                if (x - 1, y) not in _closed and (x - 1, y) in self.grid:
                    cell.append("N")

                if cell:
                    direction = random.choice(cell)
                    if direction == "E":
                        self._Open_East(x, y)
                        self.path[x, y + 1] = (x, y)
                        y += 1
                    elif direction == "W":
                        self._Open_West(x, y)
                        self.path[x, y - 1] = (x, y)
                        y -= 1
                    elif direction == "N":
                        self._Open_North(x, y)
                        self.path[x - 1, y] = (x, y)
                        x -= 1
                    elif direction == "S":
                        self._Open_South(x, y)
                        self.path[x + 1, y] = (x, y)
                        x += 1
                    _closed.append((x, y))
                    _stack.append((x, y))
                else:
                    x, y = _stack.pop()

            if loopPercent != 0:
                self.path = BFS((self.rows, self.cols))
        else:
            # Carga laberinto desde archivo CSV
            with open(loadMaze, 'r') as f:
                last = list(f.readlines())[-1]
                c = last.split(',')
                c[0] = int(c[0].lstrip('"('))
                c[1] = int(c[1].rstrip(')"'))
                self.rows = c[0]
                self.cols = c[1]
                self.grid = []

            with open(loadMaze, 'r') as f:
                r = csv.reader(f)
                next(r)
                for i in r:
                    c = i[0].split(',')
                    c[0] = int(c[0].lstrip('('))
                    c[1] = int(c[1].rstrip(')'))
                    self.maze_map[tuple(c)] = {
                        'E': int(i[1]), 'W': int(i[2]),
                        'N': int(i[3]), 'S': int(i[4])
                    }
            self.path = BFS((self.rows, self.cols))
        # Dibuja el laberinto y crea el agente objetivo en el punto de inicio
        self._drawMaze(self.theme)
        agent(self, *self._goal, shape='square', filled=True, color=COLOR.green)

        if saveMaze:
            dt_string = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
            with open(f'maze--{dt_string}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['  cell  ', 'E', 'W', 'N', 'S'])
                for k, v in self.maze_map.items():
                    entry = [k] + list(v.values())
                    writer.writerow(entry)
                f.seek(0, os.SEEK_END)
                f.seek(f.tell() - 2, os.SEEK_SET)
                f.truncate()

    def _drawMaze(self, theme):
        '''
        Crea la ventana de Tkinter y dibuja las líneas del laberinto.
        '''
        self._LabWidth = 26
        self._win = Tk()
        self._win.state('zoomed')
        self._win.title('PYTHON MAZE WORLD by Learning Orbis')

        scr_width = self._win.winfo_screenwidth()
        scr_height = self._win.winfo_screenheight()
        self._win.geometry(f"{scr_width}x{scr_height}+0+0")

        self._canvas = Canvas(width=scr_width, height=scr_height, bg=theme.value[0])
        self._canvas.pack(expand=YES, fill=BOTH)

        k = 3.25
        if self.rows >= 95 and self.cols >= 95:
            k = 0
        elif self.rows >= 80 and self.cols >= 80:
            k = 1
        elif self.rows >= 70 and self.cols >= 70:
            k = 1.5
        elif self.rows >= 50 and self.cols >= 50:
            k = 2
        elif self.rows >= 35 and self.cols >= 35:
            k = 2.5
        elif self.rows >= 22 and self.cols >= 22:
            k = 3

        self._cell_width = round(min(
            (scr_height - self.rows - k * self._LabWidth) / self.rows,
            (scr_width - self.cols - k * self._LabWidth) / self.cols,
            90
        ), 3)

        for cell in self.grid:
            x, y = cell
            w = self._cell_width
            x = x * w - w + self._LabWidth
            y = y * w - w + self._LabWidth
            if self.maze_map[cell]['E'] == 0:
                self._canvas.create_line(y + w, x, y + w, x + w, width=2, fill=theme.value[1], tag='line')
            if self.maze_map[cell]['W'] == 0:
                self._canvas.create_line(y, x, y, x + w, width=2, fill=theme.value[1], tag='line')
            if self.maze_map[cell]['N'] == 0:
                self._canvas.create_line(y, x, y + w, x, width=2, fill=theme.value[1], tag='line')
            if self.maze_map[cell]['S'] == 0:
                self._canvas.create_line(y, x + w, y + w, x + w, width=2, fill=theme.value[1], tag='line')

    def _redrawCell(self, x, y, theme):
        '''
        Redibuja una celda específica para mantener las líneas visibles.
        '''
        w = self._cell_width
        cell = (x, y)
        x = x * w - w + self._LabWidth
        y = y * w - w + self._LabWidth
        if self.maze_map[cell]['E'] == 0:
            self._canvas.create_line(y + w, x, y + w, x + w, width=2, fill=theme.value[1])
        if self.maze_map[cell]['W'] == 0:
            self._canvas.create_line(y, x, y, x + w, width=2, fill=theme.value[1])
        if self.maze_map[cell]['N'] == 0:
            self._canvas.create_line(y, x, y + w, x, width=2, fill=theme.value[1])
        if self.maze_map[cell]['S'] == 0:
            self._canvas.create_line(y, x + w, y + w, x + w, width=2, fill=theme.value[1])

    def enableArrowKey(self, a):
        '''
        Permite mover el agente con flechas del teclado.
        '''
        self._win.bind('<Left>', a.moveLeft)
        self._win.bind('<Right>', a.moveRight)
        self._win.bind('<Up>', a.moveUp)
        self._win.bind('<Down>', a.moveDown)

    def enableWASD(self, a):
        '''
        Permite mover el agente con teclas W, A, S y D.
        '''
        self._win.bind('<a>', a.moveLeft)
        self._win.bind('<d>', a.moveRight)
        self._win.bind('<w>', a.moveUp)
        self._win.bind('<s>', a.moveDown)
    _tracePathList = []

    def _tracePathSingle(self, a, p, kill, showMarked, delay):
        '''
        Método interno para animar el recorrido del agente a lo largo del camino.
        '''
        def killAgent(a):
            # Borra visualmente al agente del canvas
            for i in range(len(a._body)):
                self._canvas.delete(a._body[i])
            self._canvas.delete(a._head)

        w = self._cell_width
        if ((a.x, a.y) in self.markCells and showMarked):
            x = a.x * w - w + self._LabWidth
            y = a.y * w - w + self._LabWidth
            self._canvas.create_oval(
                y + w / 2.5 + w / 20, x + w / 2.5 + w / 20,
                y + w / 2.5 + w / 4 - w / 20, x + w / 2.5 + w / 4 - w / 20,
                fill='red', outline='red', tag='ov'
            )
            self._canvas.tag_raise('ov')

        if (a.x, a.y) == a.goal:
            del maze._tracePathList[0][0][a]
            if maze._tracePathList[0][0] == {}:
                del maze._tracePathList[0]
                if maze._tracePathList:
                    self.tracePath(*maze._tracePathList[0])
            if kill:
                self._win.after(300, killAgent, a)
            return

        if isinstance(p, dict):
            if not p:
                del maze._tracePathList[0][0][a]
                return
            a.x, a.y = p[(a.x, a.y)]
        elif isinstance(p, str):
            if not p:
                del maze._tracePathList[0][0][a]
                if maze._tracePathList[0][0] == {}:
                    del maze._tracePathList[0]
                    if maze._tracePathList:
                        self.tracePath(*maze._tracePathList[0])
                if kill:
                    self._win.after(300, killAgent, a)
                return
            move = p[0]
            if move == 'E' and a.y + 1 <= self.cols:
                a.y += 1
            elif move == 'W' and a.y - 1 > 0:
                a.y -= 1
            elif move == 'N' and a.x - 1 > 0:
                a.x -= 1
            elif move == 'S' and a.x + 1 <= self.rows:
                a.x += 1
            p = p[1:]
        elif isinstance(p, list):
            if not p:
                del maze._tracePathList[0][0][a]
                if maze._tracePathList[0][0] == {}:
                    del maze._tracePathList[0]
                    if maze._tracePathList:
                        self.tracePath(*maze._tracePathList[0])
                if kill:
                    self._win.after(300, killAgent, a)
                return
            a.x, a.y = p[0]
            del p[0]

        self._win.after(delay, self._tracePathSingle, a, p, kill, showMarked, delay)

    def tracePath(self, d, kill=False, delay=300, showMarked=False):
        '''
        Anima la ruta que deben seguir uno o varios agentes en el laberinto.

        Parámetros:
        - d: diccionario con pares {agente: camino}
        - kill: si True, borra al agente al finalizar
        - delay: retardo entre pasos (ms)
        - showMarked: muestra marcas visuales en las celdas recorridas
        '''
        self._tracePathList.append((d, kill, delay))
        if maze._tracePathList[0][0] == d:
            for a, p in d.items():
                if a.goal != (a.x, a.y) and len(p) != 0:
                    self._tracePathSingle(a, p, kill, showMarked, delay)
    def run(self):
        '''
        Inicia el ciclo principal de la interfaz Tkinter (ventana gráfica).
        '''
        self._win.mainloop()
if __name__ == '__main__':
    m = maze(10, 10)              # Crea un laberinto de 10x10
    m.CreateMaze(loopPercent=30)  # Crea el laberinto con 30% de caminos extra
    a = agent(m, filled=True)     # Crea un agente en la posición inicial
    m.tracePath({a: m.path})      # Hace que siga el camino hacia la meta
    m.enableArrowKey(a)           # Habilita flechas para moverlo manualmente
    m.run()                       # Lanza la interfaz
#TIENE FALLAS, SE TRATA DE SOLUCONAR ACORTANDO RUTAS.