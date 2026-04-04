import pygame
import numpy as np

# ------------------- CONSTANTS -------------------
WIDTH, HEIGHT = 1000, 500
G = 6.67430e-11
SCALE = 1 / 1e9
DT = 50000

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("N-Body Simulation")

# ------------------- CAMERA -------------------
zoom = 1.0
camera_pos = np.array([0.0, 0.0])

def world_to_screen(pos):
    x = (pos[0] - camera_pos[0]) * SCALE * zoom + WIDTH / 2
    y = (pos[1] - camera_pos[1]) * SCALE * zoom + HEIGHT / 2
    return int(x), int(y)

# ------------------- BODY -------------------
class Body:
    def __init__(self, mass, pos, vel, color, radius):
        self.mass = mass
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.acc = np.zeros(2)
        self.color = color
        self.radius = radius
        self.trail = []

    def update_acc(self, bodies):
        total_force = np.zeros(2)
        for other in bodies:
            if self == other:
                continue
            diff = other.pos - self.pos
            dist = np.linalg.norm(diff) + 1e-5
            force = G * self.mass * other.mass / dist**2
            total_force += force * (diff / dist)

        self.acc = total_force / self.mass

    def verlet_step(self, bodies):
        # 1. Update position
        self.pos += self.vel * DT + 0.5 * self.acc * DT**2

        # 2. Store old acceleration
        old_acc = self.acc.copy()

        # 3. Recompute acceleration
        self.update_acc(bodies)

        # 4. Update velocity
        self.vel += 0.5 * (old_acc + self.acc) * DT

        # Trail
        self.trail.append(self.pos.copy())
        if len(self.trail) > 2000:
            self.trail.pop(0)

    def draw(self, screen):
        x, y = world_to_screen(self.pos)

        # Fading trail
        for i, point in enumerate(self.trail):
            tx, ty = world_to_screen(point)
            fade = i / len(self.trail)
            color = (
                int(self.color[0] * fade),
                int(self.color[1] * fade),
                int(self.color[2] * fade)
            )
            pygame.draw.circle(screen, color, (tx, ty), 2)

        # Planet
        pygame.draw.circle(screen, self.color, (x, y), max(1, int(self.radius * zoom)))

        # Velocity vector
        vx = self.vel[0] * SCALE * 2000 * zoom
        vy = self.vel[1] * SCALE * 2000 * zoom
        pygame.draw.line(screen, (255, 255, 255), (x, y), (x + vx, y + vy), 1)

# ------------------- BARYCENTER -------------------
def barycenter(bodies):
    total_mass = sum(b.mass for b in bodies)
    com_velocity = sum(b.mass * b.vel for b in bodies) / total_mass
    for b in bodies:
        b.vel -= com_velocity

def get_barycenter(bodies):
    total_mass = sum(b.mass for b in bodies)
    return sum(b.mass * b.pos for b in bodies) / total_mass

# ------------------- SYSTEM -------------------

star = Body(1.989e30, [0, 0], [0, 0], (255, 200, 0), 12)

mercury = Body(3.3e23, [5.7e10, 0], [0, 47000], (150, 150, 150), 4)
venus   = Body(4.87e24, [1.08e11, 0], [0, 35000], (200, 180, 120), 6)
earth   = Body(5.972e24, [1.5e11, 0], [0, 30000], (100, 150, 255), 7)
mars    = Body(6.39e23, [2.2e11, 0], [0, 24000], (200, 100, 80), 6)

jupiter = Body(1.898e27, [7.7e11, 0], [0, 13000], (230, 200, 160), 14)
saturn  = Body(5.683e26, [1.4e12, 0], [0, 9600], (210, 180, 140), 12)
uranus  = Body(8.681e25, [2.9e12, 0], [0, 6800], (170, 220, 230), 10)
neptune = Body(1.024e26, [4.5e12, 0], [0, 5400], (100, 130, 255), 10)

bodies = [star, mercury, venus, earth, mars, jupiter,saturn,uranus,neptune]

barycenter(bodies)

# Initialize acceleration
for b in bodies:
    b.update_acc(bodies)

# ------------------- INPUT -------------------
dragging = False
last_mouse_pos = None

# ------------------- LOOP -------------------
clock = pygame.time.Clock()
running = True

while running:
    clock.tick(120)
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Zoom (cursor centered)
        if event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()

            world_x = (mx - WIDTH / 2) / (SCALE * zoom) + camera_pos[0]
            world_y = (my - HEIGHT / 2) / (SCALE * zoom) + camera_pos[1]

            if event.y > 0:
                zoom *= 1.2
            else:
                zoom /= 1.2

            zoom = max(0.1, min(zoom, 50))

            camera_pos[0] = world_x - (mx - WIDTH / 2) / (SCALE * zoom)
            camera_pos[1] = world_y - (my - HEIGHT / 2) / (SCALE * zoom)

        # Mouse drag panning
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            dragging = True
            last_mouse_pos = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            dragging = False
            last_mouse_pos = None

        if event.type == pygame.MOUSEMOTION and dragging:
            mx, my = pygame.mouse.get_pos()
            if last_mouse_pos is None:
                last_mouse_pos = (mx, my)
                continue

            dx = mx - last_mouse_pos[0]
            dy = my - last_mouse_pos[1]

            camera_pos[0] -= dx / (SCALE * zoom)
            camera_pos[1] -= dy / (SCALE * zoom)

            last_mouse_pos = (mx, my)

    # Physics 
    for b in bodies:
        b.verlet_step(bodies)

    # Draw bodies
    for b in bodies:
        b.draw(screen)

    # Draw barycenter
    bary = get_barycenter(bodies)
    bx, by = world_to_screen(bary)
    pygame.draw.circle(screen, (255, 255, 255), (bx, by), 4)

    pygame.display.flip()

pygame.quit()