import pygame, sys
import numpy as np
import matplotlib.pyplot as plt

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 100, 255)
GREEN = (50, 150, 50)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (130, 0, 130)
GRAY = (128, 128, 128)

BACKGROUND = WHITE


class Agent(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, color=BLACK, radius=5, velocity=[0, 0], immunity_level=None,
                 randomize=False, age=None, health_status=None):
        super().__init__()
        self.image = pygame.Surface([radius * 2, radius * 2])
        self.image.fill(BACKGROUND)
        pygame.draw.circle(self.image, color, (radius, radius), radius)
        self.rect = self.image.get_rect()
        self.pos = np.array(([x, y]), dtype=np.float64)
        self.vel = np.array(velocity, dtype=np.float64)
        self.killswitch_on = False
        self.recovered = False
        self.randomize = randomize
        self.ready_to_die = False
        self.age = age or np.random.choice(['child', 'adult', 'elderly'], p=[0.2, 0.6, 0.2])
        self.health_status = health_status or np.random.choice(['healthy', 'compromised'], p=[0.8, 0.2])
        self.immunity_level = immunity_level if immunity_level is not None else np.random.uniform(0.01, 0.2)
        self.WIDTH = width
        self.HEIGHT = height
        # Additional initialization for the new attributes

    def handle_collision(self):
        # Simple collision handling: reverse direction
        self.vel *= -1

    def update(self):

        self.pos += self.vel

        x, y = self.pos

        # Periodic boundary conditions
        if x < 0:
            self.pos[0] = self.WIDTH
            x = self.WIDTH
        if x > self.WIDTH:
            self.pos[0] = 0
            x = 0
        if y < 0:
            self.pos[1] = self.HEIGHT
            y = self.HEIGHT
        if y > self.HEIGHT:
            self.pos[1] = 0
            y = 0

        self.rect.x = x
        self.rect.y = y

        vel_norm = np.linalg.norm(self.vel)
        if vel_norm > 3:
            self.vel /= vel_norm

        if self.randomize:
            self.vel += np.random.rand(2) * 2 - 1

        if self.recovered:
            # Increase immunity upon recovery
            self.immunity_level += 0.2
            self.immunity_level = min(1.0, self.immunity_level)  # Max immunity capped at 1

        # Check boundary and obstacle collisions in the environment
        if self.rect.x < 0 or self.rect.x > self.WIDTH - self.rect.width:
            self.vel[0] *= -1
        if self.rect.y < 0 or self.rect.y > self.HEIGHT - self.rect.height:
            self.vel[1] *= -1

        if self.killswitch_on:
            self.cycles_to_fate -= 1
            if self.cycles_to_fate <= 0:
                self.killswitch_on = False
                mortality_risk = self.mortality_rate  # Base mortality rate
                if self.age == 'elderly':
                    mortality_risk *= 1.5  # Increased risk for elderly
                if self.health_status == 'compromised':
                    mortality_risk *= 1.5  # Increased risk for health-compromised individuals
                some_number = np.random.rand()
                if mortality_risk > some_number:
                    self.ready_to_die = True
                else:
                    self.recovered = True

    def respawn(self, color, radius=5):
        return Agent(
            self.rect.x,
            self.rect.y,
            self.WIDTH,
            self.HEIGHT,
            color=color,
            velocity=self.vel,
        )

    def killswitch(self, cycles_to_fate=20, mortality_rate=0.2):
        self.killswitch_on = True
        self.cycles_to_fate = cycles_to_fate
        self.mortality_rate = mortality_rate


    def apply_boids(self, neighbors, separation_weight=2.2, alignment_weight=2.1, cohesion_weight=1):
        # Assuming neighbors is a list of nearby agents of the same type
        if not neighbors:
            return
        separation = np.array([0.0, 0.0])
        for neighbor in neighbors:
            distance = np.linalg.norm(neighbor.pos - self.pos)
            if distance < 100:  # Adjusted distance threshold for separation
                separation += (self.pos - neighbor.pos) / distance  # Adjust direction and normalize

        # Alignment
        average_velocity = np.mean([neighbor.vel for neighbor in neighbors], axis=0)
        alignment = average_velocity - self.vel

        # Cohesion
        average_position = np.mean([neighbor.pos for neighbor in neighbors], axis=0)
        cohesion = average_position - self.pos

        # Combine the behaviors with weights
        self.vel += separation_weight * separation + alignment_weight * alignment + cohesion_weight * cohesion

        # Ensure velocity stays within bounds
        if np.linalg.norm(self.vel) > 3:
            self.vel = (self.vel / np.linalg.norm(self.vel)) * 1

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, color=BLACK):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class Simulation:
    def __init__(self, width=800, height=680, include_obstacles=False):
        self.WIDTH = width
        self.HEIGHT = height
        self.include_obstacles = include_obstacles
        # Data collection
        self.data = {
            'susceptible': [],
            'infected': [],
            'recovered': [],
            'vaccinated': [],
            'dead': [],
            'virulence': [],
            'average_immunity': [],
            'mortality_rate': [],
            'transmissibility': []
        }
        self.susceptible_container = pygame.sprite.Group()
        self.infected_container = pygame.sprite.Group()
        self.recovered_container = pygame.sprite.Group()
        self.vaccinated_container = pygame.sprite.Group()
        self.vaccinator_container = pygame.sprite.Group()
        self.dead_container = pygame.sprite.Group()
        self.all_container = pygame.sprite.Group()

        self.n_susceptible = 20
        self.n_infected = 1
        self.n_quarantined = 0
        self.n_vaccinated = 10
        self.n_vaccinator = 10
        self.death_count = 0  # Add a death counter
        self.T = 1500
        self.cycles_to_fate = 20
        self.mortality_rate = 0.2
        self.transmissibility = 0.1
        self.virulence = 0.2
        self.average_immunity = 0.2  # New attribute to track population immunity

        # Adjust virulence based on average immunity
        # Example: higher immunity leads to higher virulence
        self.virulence += (self.average_immunity - 0.5) * 0.05
        self.virulence = max(0.1, min(self.virulence, 1.0))  # Keep virulence within bounds

        self.obstacle_container = pygame.sprite.Group()

        self.infection_eradicated_shown = False
        self.herd_immunity_achieved_shown = False
        self.last_printed_stats = None

        if self.include_obstacles:
            self.create_obstacles()

        pygame.font.init()  # Initialize font module
        self.font = pygame.font.SysFont('Arial', 20)  # Create a Font object from the system fonts

    def record_data(self):
        self.data['susceptible'].append(len(self.susceptible_container))
        self.data['infected'].append(len(self.infected_container))
        self.data['recovered'].append(len(self.recovered_container))
        self.data['vaccinated'].append(len(self.vaccinated_container))
        self.data['dead'].append(self.death_count)
        self.data['virulence'].append(self.virulence)
        self.data['average_immunity'].append(self.average_immunity)
        self.data['mortality_rate'].append(self.mortality_rate)
        self.data['transmissibility'].append(self.transmissibility)

    def create_obstacles(self):
        # Define the number and dimensions of obstacles
        for _ in range(80):  # Example: 10 obstacles
            x = np.random.randint(0, self.WIDTH - 10)  # Avoid placing obstacles at the very edge
            y = np.random.randint(0, self.HEIGHT - 20)
            obstacle = Obstacle(x, y, np.random.randint(20, 100), np.random.randint(20, 100), GRAY)
            self.obstacle_container.add(obstacle)
        # Do not add obstacles to the all_container if it is used for agents only

    def calculate_mortality_rate(self, virulence):
        # Linear scaling parameters
        min_mortality = 0.01  # Minimum mortality rate at the lowest virulence
        max_mortality = 0.9  # Maximum mortality rate at the highest virulence

        # Linear interpolation between min and max mortality based on virulence
        mortality_rate = min_mortality + (max_mortality - min_mortality) * (virulence - 0.1) / (1.0 - 0.1)
        return mortality_rate

    def calculate_transmissibility(self):
        # Ensuring there's no division by zero and keeping the value within realistic bounds
        if self.mortality_rate > 0:
            # Example inverse relationship, adjust as needed
            transmissibility = 1 / (10 * self.mortality_rate)
        else:
            transmissibility = 0.9  # Maximum transmissibility when mortality rate is extremely low

        # Ensure the transmissibility is within a realistic range
        transmissibility = max(0.01, min(transmissibility, 0.9))
        return transmissibility

    def update_virulence(self):
        # Existing logic to update virulence...
        self.virulence += (self.average_immunity - 0.5) * 0.05
        self.virulence = max(0.1, min(self.virulence, 1.0))

        # After updating virulence, update the mortality rate and transmissibility
        self.update_mortality_rate()
        self.update_transmissibility()
        self.record_data()
        # Print updated statistics
        print(f"Updated Virulence: {self.virulence:.2f}, "
              f"Mortality Rate: {self.mortality_rate:.2f}, Transmissibility: {self.transmissibility:.2f}")

    def update_mortality_rate(self):
        base_rate = 0.01  # Set the minimum base rate for mortality
        virulence_coefficient = 0.85  # This determines the impact of virulence on mortality rate

        # Calculate mortality rate using a direct linear relationship
        self.mortality_rate = base_rate + virulence_coefficient * self.virulence

        # Ensure the mortality rate does not exceed a logical maximum or fall below a minimum
        self.mortality_rate = max(0.01, min(self.mortality_rate, 0.9))

    def update_transmissibility(self):
        # Calculate transmissibility based on the mortality rate
        if self.mortality_rate > 0:
            self.transmissibility = 1 / (10 * self.mortality_rate)
        else:
            self.transmissibility = 0.9  # Use a default value if mortality rate is zero

        # Ensure the transmissibility is within a logical range
        self.transmissibility = max(0.01, min(self.transmissibility, 0.9))

    # methods to extract statistics

    def get_number_susceptible(self):
        return len(self.susceptible_container)
    def get_number_infected(self):
        return len(self.infected_container)

    def get_number_recovered(self):
        return len(self.recovered_container)

    def get_number_dead(self):
        return self.death_count

    def get_number_vaccinated(self):
        return len(self.vaccinated_container)

    def get_number_vaccinators(self):
        return len(self.vaccinator_container)

    def get_current_stats(self):
        return (
            self.get_number_infected(),
            self.get_number_recovered(),
            self.get_number_dead(),
            self.get_number_vaccinated()
        )

    def draw_statistics(self, screen):
        # Calculate average immunity level for reporting
        total_immunity = sum(agent.immunity_level for agent in self.all_container.sprites())
        average_immunity = total_immunity / len(self.all_container.sprites()) if self.all_container.sprites() else 0

        # Update the statistics text to include comprehensive stats
        stats_text = (
            #f"Susceptible: {self.get_number_susceptible()}, "
            #f"Infected: {self.get_number_infected()}, "
            #f"Recovered: {self.get_number_recovered()}, "
            #f"Dead: {self.get_number_dead()}, "
            #f"Vaccinated: {self.get_number_vaccinated() + self.get_number_vaccinators()}, "
            f"Virulence: {self.virulence:.2f}, "
            f"Average Immunity: {average_immunity:.2f}, "
            f"Mortality Rate: {self.mortality_rate:.2f}, "
            f"Transmissibility: {self.calculate_transmissibility():.2f}"
        )
        text_surface = self.font.render(stats_text, True, BLACK)
        # Adjust the positioning if needed to ensure the text fits on the screen
        screen.blit(text_surface, (10, 10))  # Position the text at the top left corner

    def draw_graphs(self, screen):
        # Define some parameters for the bar graph
        bar_width = 40
        max_bar_height = 150  # Maximum height of a bar
        spacing = 80  # Spacing between bars

        # Data for drawing
        categories = ['Susceptible ', '  Infected', 'Recovered', 'Vaccinated', 'Dead']
        values = [
            len(self.susceptible_container),
            len(self.infected_container),
            len(self.recovered_container),
            len(self.vaccinated_container),
            self.death_count  # Include the death count
        ]
        max_value = max(values) if max(values) > 0 else 1  # Avoid division by zero
        colors = [BLUE, RED, PURPLE, GREEN, BLACK]  # Add a color for the death count

        # Starting position for the first bar
        start_x = 50
        start_y = self.HEIGHT - 50  # Adjust as necessary

        # Draw each bar
        for i, value in enumerate(values):
            # Calculate the height of the bar based on the value
            bar_height = (value / max_value) * max_bar_height
            # Draw the bar
            pygame.draw.rect(
                screen,
                colors[i],
                (start_x + i * spacing, start_y - bar_height, bar_width, bar_height)
            )
            # Draw the category name or value below each bar
            text_surface = self.font.render(categories[i], True, BLACK)
            screen.blit(text_surface, (start_x + i * spacing, start_y + 10))

            # Drawing the value above each bar for clarity
            value_surface = self.font.render(str(value), True, BLACK)
            screen.blit(value_surface, (start_x + i * spacing, start_y - bar_height - 20))

    def start(self, randomize=False):

        self.N = self.n_susceptible + self.n_infected + self.n_quarantined + self.n_vaccinated
        countdown = None  # Initialize countdown timer to None
        mortality_rate = self.calculate_mortality_rate(self.virulence)

        pygame.init()
        screen = pygame.display.set_mode(
            [self.WIDTH, self.HEIGHT]
        )

        for i in range(self.n_susceptible):
            x = np.random.randint(0, self.WIDTH + 1)
            y = np.random.randint(0, self.HEIGHT + 1)
            vel = np.random.rand(2) * 2 - 1
            person = Agent(
                x,
                y,
                self.WIDTH,
                self.HEIGHT,
                color=BLUE,
                velocity=vel,
                randomize=randomize,
            )
            self.susceptible_container.add(person)
            self.all_container.add(person)

        for i in range(self.n_quarantined):
            x = np.random.randint(0, self.WIDTH + 1)
            y = np.random.randint(0, self.HEIGHT + 1)
            vel = [0, 0]
            person = Agent(
                x,
                y,
                self.WIDTH,
                self.HEIGHT,
                color=BLUE,
                velocity=vel,
                randomize=False,
            )
            self.susceptible_container.add(person)
            self.all_container.add(person)

        for i in range(self.n_vaccinator):
            x = np.random.randint(0, self.WIDTH + 1)
            y = np.random.randint(0, self.HEIGHT + 1)
            vel = np.random.rand(2) * 2 - 1
            person = Agent(
                x,
                y,
                self.WIDTH,
                self.HEIGHT,
                color=YELLOW,
                velocity=vel,
                randomize=randomize,
            )
            self.vaccinator_container.add(person)
            self.all_container.add(person)

        for i in range(self.n_infected):
            x = np.random.randint(0, self.WIDTH + 1)
            y = np.random.randint(0, self.HEIGHT + 1)
            vel = np.random.rand(2) * 0.000001
            person = Agent(
                x,
                y,
                self.WIDTH,
                self.HEIGHT,
                color=RED,
                velocity=vel,
                randomize=randomize,
            )
            mortality_rate = self.calculate_mortality_rate(self.virulence)
            person.killswitch(self.cycles_to_fate, mortality_rate)
            self.infected_container.add(person)
            self.all_container.add(person)

        for i in range(self.n_vaccinated):
            x = np.random.randint(0, self.WIDTH + 1)
            y = np.random.randint(0, self.HEIGHT + 1)
            vel = np.random.rand(2) * 2 - 1
            person = Agent(
                x,
                y,
                self.WIDTH,
                self.HEIGHT,
                color=GREEN,
                velocity=vel,
                randomize=False,
            )
            self.vaccinated_container.add(person)
            self.all_container.add(person)

        clock = pygame.time.Clock()

        for i in range(self.T):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            self.all_container.update()

            screen.fill(BACKGROUND)

            # Collision detection with obstacles
            for agent in self.all_container:
                # Ensure that handle_collision is only called on agents
                if isinstance(agent, Agent) and pygame.sprite.spritecollideany(agent, self.obstacle_container):
                    agent.handle_collision()

            # Combined vaccination and infection logic
            for susceptible in self.susceptible_container.sprites():
                # Check for collision with vaccinators
                if pygame.sprite.spritecollideany(susceptible, self.vaccinator_container):
                    # Vaccinate the susceptible agent
                    self.susceptible_container.remove(susceptible)
                    self.all_container.remove(susceptible)
                    vaccinated = susceptible.respawn(GREEN)
                    # Setting high immunity for vaccinated agents
                    vaccinated.immunity_level = np.random.uniform(0.8, 1.0)
                    self.vaccinated_container.add(vaccinated)
                    self.all_container.add(vaccinated)
                # If not vaccinated, then check for collision with infected agents
                elif pygame.sprite.spritecollideany(susceptible, self.infected_container):
                    transmissibility = self.calculate_transmissibility()
                    # Adjust transmissibility based on the susceptible's immunity
                    adjusted_transmissibility = transmissibility * (1 - susceptible.immunity_level)

                    if np.random.rand() < adjusted_transmissibility:
                        # Only infect the susceptible agent if the random check passes
                        self.susceptible_container.remove(susceptible)
                        self.all_container.remove(susceptible)
                        new_person = susceptible.respawn(RED)
                        new_person.vel *= -1
                        new_person.killswitch(self.cycles_to_fate, self.mortality_rate)
                        self.infected_container.add(new_person)
                        self.all_container.add(new_person)

            # Any recoveries?

            recovered = []
            for person in self.infected_container:
                if person.recovered:
                    new_person = person.respawn(PURPLE)
                    self.recovered_container.add(new_person)
                    self.all_container.add(new_person)
                    recovered.append(person)

            if len(recovered) > 0:
                self.infected_container.remove(*recovered)
                self.all_container.remove(*recovered)

            for agent in self.all_container:
                if self.use_boids_algorithm:
                    # Filter neighbors based on proximity
                    neighbors = [other for other in self.all_container if other != agent and np.linalg.norm(
                        other.pos - agent.pos) < 100]  # Example radius of 100
                    agent.apply_boids(neighbors)
                agent.update()

            for agent in self.all_container:
                if getattr(agent, 'ready_to_die', False):
                    self.death_count += 1
                    agent.kill()  # Now calling kill after incrementing the death_count

                # Check for "Infection Eradicated"
                if self.get_number_infected() == 0 and not self.infection_eradicated_shown:
                    print("Infection Eradicated")
                    self.infection_eradicated_shown = True

                # Check for "Herd Immunity Achieved"
                total_population = self.N
                immune_population = self.get_number_vaccinated() + self.get_number_recovered()
                if immune_population / total_population >= 0.7 and not self.herd_immunity_achieved_shown:
                    print("Herd Immunity Achieved")
                    self.herd_immunity_achieved_shown = True

            self.all_container.draw(screen)
            self.obstacle_container.draw(screen)
            self.draw_statistics(screen)  # Draw the statistics on the window
            self.draw_graphs(screen)

            if self.get_number_infected() == 0 and countdown is None:
                countdown = 8 * 60  # 8 seconds * 60 FPS = 480 frames

            if countdown is not None:
                countdown -= 1  # Decrement the countdown timer
                if countdown <= 0:
                    break  # Exit the loop when countdown reaches zero

            if i % 15 == 0:  # Example: update every 10 cycles
                self.update_virulence()
                self.record_data()

            pygame.display.flip()

            clock.tick(40)

        pygame.quit()

def plot_data(data):
    plt.figure(figsize=(8, 6))
    plt.plot(data['susceptible'], label='Susceptible')
    plt.plot(data['infected'], label='Infected')
    plt.plot(data['recovered'], label='Recovered')
    plt.plot(data['vaccinated'], label='Vaccinated')
    plt.plot(data['dead'], label='Dead')
    plt.title('Simulation Results')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Number of Agents')
    plt.legend()
    plt.show()

from scipy.signal import savgol_filter

def smooth_data(data, window_size=51, poly_order=3):
    if len(data) < window_size:
        return data  # Not enough data to smooth
    return savgol_filter(data, window_size, poly_order)

def plot_parameters(data):
    # Check lengths of data arrays
    print("Lengths:", len(data['virulence']), len(data['average_immunity']), len(data['mortality_rate']),
          len(data['transmissibility']))
    plt.figure(figsize=(8, 6))
    plt.plot(smooth_data(data['virulence']), label='Virulence')
    plt.plot(smooth_data(data['average_immunity']), label='Average Immunity')
    plt.plot(smooth_data(data['mortality_rate']), label='Mortality Rate')
    plt.plot(smooth_data(data['transmissibility']), label='Transmissibility')
    plt.title('Disease Dynamics Over Time')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Parameter Values')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    disease = Simulation(width=1280, height=720, include_obstacles=False)
    disease.n_susceptible = 150
    disease.n_quarantined = 0
    disease.n_infected = 4
    disease.n_vaccinated = 0
    disease.n_vaccinator = 4
    disease.cycles_to_fate = 350
    disease.virulence=0.1
    disease.use_boids_algorithm = False
    disease.start(randomize=True)
    # Plot the results after the simulation
    plot_data(disease.data)  # Plot agent counts
    plot_parameters(disease.data)  # Plot disease dynamics
