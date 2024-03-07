import random
import numpy as np
import time

class CarbonTaxEnvironment:
    def __init__(self, max_days, initial_revenue, initial_expenses, initial_money_in_bank, initial_max_car, initial_ev, initial_petrol_car, initial_power_consumption, initial_power_generation, initial_solar_panel, initial_electricity_price, max_electricity, initial_carbon_tax):
        self.num_day = max_days  # 5 years
        self.current_day = 0
        self.observation_space_dim = 10  # Adjust if needed for your state
        self.action_space_dim = 3  # 3 possible actions

        self.revenue = initial_revenue
        self.initial_expenses = initial_expenses
        self.initial_money_in_bank = initial_money_in_bank
        self.max_car = initial_max_car
        self.initial_ev = initial_ev
        self.initial_petrol_car = initial_petrol_car
        self.initial_power_consumption = initial_power_consumption
        self.initial_power_generation = initial_power_generation
        self.initial_solar_panel = initial_solar_panel
        self.electricity_price = initial_electricity_price
        self.carbon_tax = initial_carbon_tax
        self.max_electricity = max_electricity
        self.initial_state = self._initialize_state()
    def _initialize_state(self):
        # Define the starting values for:
        # All information are for day
        # unit are in dollars, kwh, km, kg
        self.expenses = self.initial_expenses
        self.money_in_bank = self.initial_money_in_bank

        self.ev = self.initial_ev
        self.petrol_car = self.initial_petrol_car
        self.car_mile = 200
        self.pertrol_per_km = 0.06
        self.car_carbon_per_km = 0.2
        self.ev_per_km = 0.147

        self.power_consumption = self.initial_power_consumption
        self.carbon_per_kwh_generated = 0.4057
        self.power_generation = self.initial_power_generation
        self.solar_panel = self.initial_solar_panel

        _,_,_,_,_,self.init_carbon_emitted = self.calculate_expenses()

        state = [self.revenue, self.expenses, self.money_in_bank, self.ev, self.petrol_car,
                 self.power_consumption, self.power_generation, self.solar_panel, self.electricity_price, self.carbon_tax]

        return state
    
    def calculate_expenses(self):
        power_by_ev = self.ev * self.ev_per_km * self.car_mile
        power_from_grid = self.power_consumption + power_by_ev - self.power_generation
        carbon_from_power = power_from_grid * self.carbon_per_kwh_generated
        cost_of_electricity = power_from_grid * self.electricity_price
        petrol_cost = self.petrol_car * self.car_mile * self.pertrol_per_km
        carbon_emitted = self.petrol_car * self.car_mile * self.car_carbon_per_km + carbon_from_power
        return power_by_ev, power_from_grid, carbon_from_power, cost_of_electricity, petrol_cost, carbon_emitted

    def step(self, action):
        # 1. Apply the effect of current carbon tax rate and the 'action'
        reward = 0
        bankrupcy = False

        if action == 0:  # Buy EV
            if self.ev < self.max_car:
                self.ev += 1
                self.petrol_car = max(0, self.petrol_car - 1)
                self.money_in_bank -= 30000
            else:
                reward = -50
        elif action == 1:  # Buy solar panel
            if self.power_generation <= self.max_electricity:
                self.solar_panel += 1
                self.money_in_bank -= 900
                self.power_generation += 2
            else:
                reward = -10
        elif action == 2:  # Do nothing
            reward -= 1


        # 2. Simulate company behaviors (randomness might be involved)
        # generate a number based on normal distribution
        # self.revenue = int(self.revenue * (1 + random.gauss(0, 0.1))) # 10% fluctuation
        # if self.revenue <= 0:
        #     self.revenue = 1
        # self.car_mile = int(self.car_mile * (1 + random.gauss(0, 0.2))) # 20% fluctuation
        # self.power_consumption = int(self.power_consumption * (1 + random.gauss(0, 0.2))) # 20% fluctuation

        # 3. Update your state variables:
        power_by_ev, power_from_grid, carbon_from_power, cost_of_electricity, petrol_cost, carbon_emitted = self.calculate_expenses()

        if power_from_grid < 0:
            surplus_power = -power_from_grid
        else:
            surplus_power = 0

        if carbon_emitted < 0:
            carbon_surplus = -carbon_emitted
        else:
            carbon_surplus = 0
            
        carbon_tax_cost = carbon_emitted * self.carbon_tax

        self.expenses = carbon_tax_cost + cost_of_electricity + petrol_cost + self.initial_expenses  # Simplifying other expenses

        self.money_in_bank += self.revenue - self.expenses

        # reward = self.money_in_bank * 0.00000001 - carbon_emitted * 0.0001

        reward = reward + ((self.money_in_bank - self.initial_money_in_bank) / self.initial_money_in_bank * 10 - (carbon_emitted - self.init_carbon_emitted) / self.init_carbon_emitted * 10)
        # print(f'reward: {reward}')

        # reward = reward / self.revenue  * 10
        money_in_bank = self.money_in_bank

        if self.money_in_bank < 0:
            done = True
            bankrupcy = True
            reward = -100
            self.reset()
        elif self.current_day >= self.num_day:
            done = True
            self.reset()
        else:
            done = False

        # Normalize to -3 to 3 range
        # reward = (reward - 1e7) / 1e7
        # print(f'Current Day: {self.current_day}, Action: {action} Reward: {reward}, Money in Bank: {self.money_in_bank}, EV: {self.ev}, Solar Panel: {self.solar_panel}, carbon emitted: {carbon_emitted}, carbon surplus: {carbon_surplus}, carbon tax cost: {carbon_tax_cost}, petrol cost: {petrol_cost}, bankrupcy: {bankrupcy}')
        # time.sleep(1)

        # 5-6. Check for end of episode and get new state
        self.current_day += 1
        new_state = self._get_current_state()  # Get the updated state

        data = [carbon_emitted, surplus_power, carbon_tax_cost, petrol_cost, carbon_surplus, bankrupcy, money_in_bank]
        visualization  = [self.ev, self.petrol_car, self.solar_panel, self.power_generation, self.power_consumption, self.electricity_price, self.carbon_tax, self.revenue, self.expenses, self.money_in_bank, self.current_day, self.max_car, self.max_electricity]

        return new_state, reward, done, data, visualization

    def reset(self):
        self.current_day = 0
        state = self._initialize_state()
        return state

    def _get_current_state(self):  # Helper to get the current state as an array
        return np.array([self.revenue, self.expenses, self.money_in_bank, self.ev, self.petrol_car,
                 self.power_consumption, self.power_generation, self.solar_panel,
                 self.electricity_price, self.carbon_tax])

    @property  # Add a property for observation_space
    def observation_space(self):
        return np.zeros(self.observation_space_dim)

    @property
    def action_space(self):
        return np.zeros(self.action_space_dim)
