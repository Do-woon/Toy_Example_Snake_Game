from pygame.locals import *
from random import randint
import pygame
import time


class Apple:
    x = 0
    y = 0
    step = 44

    def __init__(self, x, y):
        self.x = x * self.step
        self.y = y * self.step

    def draw(self, surface, image):
        surface.blit(image, (self.x, self.y))


class Player:
    step = 44

    def __init__(self, length):
        self.length = length
        self.direction = 0
        self.x = []
        self.y = []

        for i in range(0, 2000):
            self.x.append(-100)
            self.y.append(-100)

        # initial positions, no collision.
        for i in range(length):
            self.x[i] = (length-i)*44
            self.y[i] = 0

    def update(self):
        # update previous positions
        for i in range(self.length - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]

        # update position of head of snake
        if self.direction == 0:
            self.x[0] = self.x[0] + self.step
        if self.direction == 1:
            self.x[0] = self.x[0] - self.step
        if self.direction == 2:
            self.y[0] = self.y[0] - self.step
        if self.direction == 3:
            self.y[0] = self.y[0] + self.step

    def move_right(self):
        if self.direction != 1:
            self.direction = 0

    def move_left(self):
        if self.direction != 0:
            self.direction = 1

    def move_up(self):
        if self.direction != 3:
            self.direction = 2

    def move_down(self):
        if self.direction != 2:
            self.direction = 3

    def draw(self, surface, image):
        for i in range(0, self.length):
            surface.blit(image, (self.x[i], self.y[i]))


class Game:
    def is_collision(self, x1, y1, x2, y2, bsize):
        if x1 >= x2 and x1 <= x2 + bsize:
            if y1 >= y2 and y1 <= y2 + bsize:
                return True
        return False

    def is_outsize(self, x, y, win_width, win_height, bsize):
        if x > win_width - bsize or x < 0:
            return True
        elif y > win_height - bsize or y < 0:
            return True
        else:
            return False


class App:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.game = Game()
        self.width = 6
        self.height = 6
        self.windowWidth = 44 * self.width
        self.windowHeight = 44 * self.height
        self.player = Player(3)
        self.apple = Apple(randint(1, self.width-1), randint(1, self.height-1))

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.windowWidth, self.windowHeight), pygame.HWSURFACE)

        if self.verbose:
            pygame.display.set_caption('Pygame pythonspot.com example')
        self._running = True
        self._image_surf = pygame.image.load("./game_img/block_body.jpg").convert()
        self._apple_surf = pygame.image.load("./game_img/block_apple.jpg").convert()

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def on_loop(self):
        self.player.update()
        reward = 0

        # does snake go outside? (should check before apple eating)
        for i in range(0, self.player.length):
            if self.game.is_outsize(self.player.x[i], self.player.y[i], self.windowWidth, self.windowHeight, 43):
                if self.verbose:
                    print("You go outside!")
                self._running = False

        # does snake eat apple?
        for i in range(0, self.player.length):
            if self.game.is_collision(self.apple.x, self.apple.y, self.player.x[i], self.player.y[i], 43):

                self.apple.x = randint(1, self.width-1) * 44
                self.apple.y = randint(1, self.height-1) * 44
                while (self.apple.x, self.apple.y) in zip(self.player.x, self.player.y):
                    self.apple.x = randint(1, self.width - 1) * 44
                    self.apple.y = randint(1, self.height - 1) * 44

                self.player.length = self.player.length + 1
                reward = self.player.length * 10
                break

        # does snake collide with itself?
        for i in range(2, self.player.length):
            if self.game.is_collision(self.player.x[0], self.player.y[0], self.player.x[i], self.player.y[i], 40):
                if self.verbose:
                    print("You lose! Collision: ")
                    print("x[0] (" + str(self.player.x[0]) + "," + str(self.player.y[0]) + ")")
                    print("x[" + str(i) + "] (" + str(self.player.x[i]) + "," + str(self.player.y[i]) + ")")
                self._running = False

        return reward

    def on_render(self):
        self._display_surf.fill((0, 0, 0))
        self.player.draw(self._display_surf, self._image_surf)
        self.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def is_running(self):
        return self._running

    def do_action(self, action, do_render=True):
        if isinstance(action, tuple):
            if action[K_RIGHT]:
                self.player.move_right()
            if action[K_LEFT]:
                self.player.move_left()
            if action[K_UP]:
                self.player.move_up()
            if action[K_DOWN]:
                self.player.move_down()
            if action[K_ESCAPE]:
                self._running = False
        else:
            if action==0:
                self.player.move_right()
            elif action==1:
                self.player.move_left()
            elif action==2:
                self.player.move_up()
            elif action==3:
                self.player.move_down()

        reward = self.on_loop()
        if do_render:
            self.on_render()
            if isinstance(action, tuple):
                time.sleep(100.0/1000.0)
            else:
                time.sleep(100.0/1000.0)

        if self.player.x[self.player.length-1] == -100:
            return self.player.x[:self.player.length-1], self.player.y[:self.player.length-1], \
                   self.player.direction, self.player.length, self.apple.x, self.apple.y, reward
        else:
            return self.player.x[:self.player.length], self.player.y[:self.player.length], \
                   self.player.direction, self.player.length, self.apple.x, self.apple.y, reward

    def reset(self):
        self._running = True
        self.player = Player(3)
        self.apple = Apple(randint(1, self.width-1), randint(1, self.height-1))
        return self.player.x[:self.player.length], self.player.y[:self.player.length], \
               self.player.direction, self.player.length, self.apple.x, self.apple.y, 0


if __name__ == "__main__":
    theApp = App()

    theApp.on_init()
    for n in range(10):
        while theApp.is_running():
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            print(theApp.do_action(keys))
        theApp.reset()

    theApp.on_cleanup()
